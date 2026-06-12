use crate::custom_family::{
    AdditiveBlockJacobian, BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix,
    fit_custom_family_with_rho_prior,
};
use crate::estimate::{
    AdaptiveRegularizationOptions, EstimationError, FitOptions, FittedLinkState, UnifiedFitResult,
};
use crate::families::bms::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    fit_bernoulli_marginal_slope_terms,
};
use crate::families::gamlss::{
    BinomialLocationScaleFitResult, BinomialLocationScaleTermSpec, BlockwiseTermFitResult,
    BlockwiseTermFitResultParts, BlockwiseTermWiggleFitResult, DispersionFamilyKind,
    DispersionGlmLocationScaleTermSpec, GaussianLocationScaleFitResult,
    GaussianLocationScaleTermSpec, fit_binomial_location_scale_terms,
    fit_binomial_location_scale_terms_with_selected_wiggle,
    fit_binomial_mean_wiggle_terms_with_selected_basis, fit_dispersion_glm_location_scale_terms,
    fit_gaussian_location_scale_terms, fit_gaussian_location_scale_terms_with_selected_wiggle,
    select_binomial_location_scale_link_wiggle_basis_from_pilot,
    select_binomial_mean_link_wiggle_basis_from_pilot,
    select_gaussian_location_scale_link_wiggle_basis_from_pilot,
};
use crate::families::latent_survival::{
    LatentBinaryTermFitResult, LatentBinaryTermSpec, LatentSurvivalTermFitResult,
    LatentSurvivalTermSpec, fit_latent_binary_terms, fit_latent_survival_terms,
    latent_hazard_loading,
};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::survival_location_scale::{
    SurvivalLocationScaleTermFitResult, SurvivalLocationScaleTermSpec,
    fit_survival_location_scale_terms, fit_survival_location_scale_terms_with_selected_wiggle,
    select_survival_link_wiggle_basis_from_pilot,
};
use crate::families::survival_marginal_slope::{
    SurvivalMarginalSlopeFitResult, SurvivalMarginalSlopeTermSpec,
    fit_survival_marginal_slope_terms,
};
use crate::families::transformation_normal::{
    TransformationNormalConfig, TransformationNormalFitResult, TransformationWarmStart,
    fit_transformation_normal,
};
use crate::families::wiggle::WiggleBlockConfig;
use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, CoefficientGroupSpec, LinearTermSpec,
    SpatialLengthScaleOptimizationOptions, StandardLatentCoordConfig, TermCollectionDesign,
    TermCollectionSpec, build_term_collection_design,
    fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors,
    fit_term_collectionwith_latent_coord_optimization,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use crate::solver::latent_cache::LatentRetractionRegistry;
use crate::solver::riemannian_retraction::{ProductRetraction, RetractionKind};
use crate::survival::PenaltyBlock;
use crate::terms::latent_coord::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkFunction, MixtureLinkSpec,
    ResponseColumnKind, ResponseFamily, SasLinkSpec, StandardLink, WigglePenaltyConfig,
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde_json::Value as JsonValue;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Shared analytic-penalty descriptor parser. Both this in-process workflow
/// pipeline and the Python FFI (`gam-pyffi`) build their analytic-penalty
/// registries through [`descriptors::build_analytic_penalty_registry_from_descriptors`],
/// so the descriptor schema, defaults, shape checks, and error messages are
/// identical for every caller.
pub mod descriptors;

trait WorkflowCauseCountResult {
    fn into_workflow_result(self) -> Result<usize, String>;
}

impl WorkflowCauseCountResult for usize {
    fn into_workflow_result(self) -> Result<usize, String> {
        Ok(self)
    }
}

impl<E: ToString> WorkflowCauseCountResult for Result<usize, E> {
    fn into_workflow_result(self) -> Result<usize, String> {
        self.map_err(|err| err.to_string())
    }
}

/// Typed error category for the `solver::workflow` materialization and
/// fitting pipeline.
///
/// Every variant's `Display` impl is byte-equivalent to the original
/// `format!(...)`/`.to_string()` text the module emitted before the typed
/// migration. The category split lets internal callers reason about the
/// failure kind without parsing strings; public entry points keep their
/// `Result<_, String>` signatures and rely on `From<WorkflowError> for
/// String` at the boundary.
#[derive(Debug, Clone)]
pub enum WorkflowError {
    /// Fit configuration is internally inconsistent or selects an
    /// unsupported combination (conflicting `family`/`link`, unsupported
    /// `linkwiggle(...)`/`link(...)` placement, `frailty` requested for a
    /// family that does not implement it, duplicate or out-of-range
    /// hyperpriors, etc.).
    InvalidConfig { reason: String },
    /// Saved-model or runtime block dimensions disagree with what the
    /// rebuilt designs / penalties expect (initial beta length, penalty
    /// block shape vs range width, time-basis column count, response
    /// support mismatch).
    SchemaMismatch { reason: String },
    /// A required input column, frailty parameter, baseline target, or
    /// cause count is missing for the requested mode (e.g. cause-specific
    /// fit with one cause, latent-cloglog without a fixed sigma).
    MissingDependency { reason: String },
    /// An underlying numerical step (PIRLS / smoothing-parameter
    /// optimizer / profile-cost evaluation) failed to converge or
    /// produced a non-finite value that downstream code cannot consume.
    IntegrationFailed { reason: String },
    /// Formula parsing / term-resolution failed before materialization; the
    /// source retains the parser-layer category and argument context.
    FormulaDsl {
        context: &'static str,
        source: crate::inference::formula_dsl::FormulaDslError,
    },
    /// A formula referenced a column that does not exist in the input data.
    /// Carries the structured payload through to the FFI boundary so the
    /// Python side can raise `gamfit.ColumnNotFoundError` with `column`,
    /// `role`, `available`, `similar`, and `tsv_hint` attributes — issue
    /// #305 / #343 (typed-dispatch migration; no string classification at
    /// the boundary).
    ColumnNotFound {
        name: String,
        role: Option<String>,
        available: Vec<String>,
        similar: Vec<String>,
        tsv_hint: bool,
    },
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowError::InvalidConfig { reason }
            | WorkflowError::SchemaMismatch { reason }
            | WorkflowError::MissingDependency { reason }
            | WorkflowError::IntegrationFailed { reason } => f.write_str(reason),
            WorkflowError::FormulaDsl { context, source } => write!(f, "{context}: {source}"),
            // Reconstruct the display text from the structured payload so
            // CLI / `to_string()` consumers see the same prose the legacy
            // `missing_column_message` produced. The text is a function of
            // the typed fields — not parsed back out anywhere.
            WorkflowError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => {
                let label = match role {
                    Some(r) => format!("{r} column '{name}'"),
                    None => format!("column '{name}'"),
                };
                let tsv_suffix = if *tsv_hint {
                    " — your file appears to be tab-separated; gam expects comma-separated CSV. \
         Replace tabs with commas, or pre-convert with `tr '\\t' ',' < file.tsv > file.csv`."
                } else {
                    ""
                };
                if similar.is_empty() {
                    write!(
                        f,
                        "{label} not found in data. Available columns: [{}]{tsv_suffix}",
                        available.join(", ")
                    )
                } else {
                    write!(
                        f,
                        "{label} not found in data. Did you mean one of [{}]? Full list: [{}]{tsv_suffix}",
                        similar.join(", "),
                        available.join(", ")
                    )
                }
            }
        }
    }
}

impl std::error::Error for WorkflowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WorkflowError::FormulaDsl { source, .. } => Some(source),
            WorkflowError::InvalidConfig { .. }
            | WorkflowError::SchemaMismatch { .. }
            | WorkflowError::MissingDependency { .. }
            | WorkflowError::IntegrationFailed { .. }
            | WorkflowError::ColumnNotFound { .. } => None,
        }
    }
}

impl From<WorkflowError> for String {
    fn from(err: WorkflowError) -> String {
        err.to_string()
    }
}

/// Catchall lift for legacy `Result<_, String>` chains that flow into a
/// `WorkflowError`-returning function via `?`. Maps to `InvalidConfig` since
/// the upstream call sites that still hand out bare strings are
/// configuration / setup helpers (FitConfig parsing, payload assembly, etc.)
/// that pre-date the typed-error migration. Specific leaves that carry
/// structured payload (`DataError`, `FormulaDslError`, `EstimationError`,
/// …) have their own dedicated `From` impls and bypass this fallback.
impl From<String> for WorkflowError {
    fn from(reason: String) -> Self {
        Self::InvalidConfig { reason }
    }
}

impl From<&str> for WorkflowError {
    fn from(reason: &str) -> Self {
        Self::InvalidConfig {
            reason: reason.to_string(),
        }
    }
}

/// Cross-module cascade: a `FormulaDslError` raised inside `materialize` /
/// `fit_from_formula` (via `parse_formula`, `parse_surv_response`, etc.) flows
/// up with its parser-layer source attached instead of stringifying into a
/// generic workflow configuration bucket.
impl From<crate::inference::formula_dsl::FormulaDslError> for WorkflowError {
    fn from(err: crate::inference::formula_dsl::FormulaDslError) -> Self {
        Self::FormulaDsl {
            context: "workflow formula materialization",
            source: err,
        }
    }
}

/// Typed lift from term-builder errors. `TermBuilderError::ColumnNotFound`
/// preserves the structured fields (name, role, available, similar,
/// tsv_hint) through to the FFI boundary so `gam-pyffi` can raise a
/// `gamfit.ColumnNotFoundError` with attributes set from the payload —
/// not from re-parsed prose. Other variants degrade into the closest
/// generic workflow bucket; the dedicated typed channels for those
/// failure classes can be added incrementally as their dispatch arrives.
impl From<crate::terms::term_builder::TermBuilderError> for WorkflowError {
    fn from(err: crate::terms::term_builder::TermBuilderError) -> Self {
        use crate::terms::term_builder::TermBuilderError;
        match err {
            TermBuilderError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => Self::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            },
            TermBuilderError::MissingColumn { reason }
            | TermBuilderError::MalformedFormula { reason } => Self::SchemaMismatch { reason },
            TermBuilderError::IncompatibleConfig { reason }
            | TermBuilderError::InvalidOption { reason }
            | TermBuilderError::UnsupportedFeature { reason }
            | TermBuilderError::DegenerateData { reason } => Self::InvalidConfig { reason },
        }
    }
}

/// Typed lift from leaf data-layer errors. `DataError::ColumnNotFound` is
/// the variant of immediate interest — it preserves the structured fields
/// so `gam-pyffi` can dispatch to `ColumnNotFoundError` without parsing
/// human text. Other `DataError` variants degrade to the appropriate
/// workflow bucket (`SchemaMismatch` for row/column shape problems,
/// `InvalidConfig` for parse / encoding / empty / invalid-value sources)
/// since they don't have a dedicated structured destination yet.
impl From<crate::inference::data::DataError> for WorkflowError {
    fn from(err: crate::inference::data::DataError) -> Self {
        use crate::inference::data::DataError;
        match err {
            DataError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => Self::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            },
            DataError::SchemaMismatch { reason } => Self::SchemaMismatch { reason },
            DataError::ParseError { reason }
            | DataError::EncodingFailure { reason }
            | DataError::EmptyInput { reason }
            | DataError::InvalidValue { reason } => Self::InvalidConfig { reason },
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinkWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

/// Configuration for the second-stage binomial-mean wiggle fit appended to a
/// standard pilot. The blockwise refit options live inside this struct so the
/// pilot config (`link_kind` + `wiggle`) and its required `refit_options` can
/// never disagree: either the whole standard-wiggle request is `Some`, or it
/// is `None`. The previous shape had two sibling `Option` fields on
/// `StandardFitRequest`, which allowed the materialize path to construct an
/// inconsistent state (#320: linkwiggle config without blockwise options).
#[derive(Clone)]
pub struct StandardBinomialWiggleConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleConfig,
    pub refit_options: BlockwiseFitOptions,
}

pub struct StandardFitRequest<'a> {
    pub data: Array2<f64>,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodSpec,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleConfig>,
    pub coefficient_groups: Vec<CoefficientGroupSpec>,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
    pub latent_coord: Option<StandardLatentCoordConfig>,
    #[doc(hidden)]
    pub _marker: std::marker::PhantomData<&'a ()>,
}

pub struct GaussianLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct BinomialLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BinomialLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct DispersionLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: DispersionGlmLocationScaleTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct SurvivalLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub optimize_inverse_link: bool,
    /// See [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_location_scale_model`.
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
}

pub struct SurvivalTransformationFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalTransformationTermSpec,
    /// See [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_transformation_model`.
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
}

#[derive(Clone)]
pub struct SurvivalTransformationTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub covariate_offset: Array1<f64>,
    pub baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    pub likelihood_mode: crate::families::survival_construction::SurvivalLikelihoodMode,
    pub time_anchor: f64,
    pub time_build: crate::families::survival_construction::SurvivalTimeBuildOutput,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub weibull_seed: Option<(f64, f64)>,
    pub ridge_lambda: f64,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
}

pub(crate) fn survival_inverse_link_has_free_parameters(link: &InverseLink) -> bool {
    match link {
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
        InverseLink::Mixture(state) => !state.rho.is_empty(),
        InverseLink::LatentCLogLog(_) | InverseLink::Standard(_) => false,
    }
}

fn recover_converged_survival_inverse_link<R>(
    result: crate::solver::outer_strategy::OuterResult,
    context: &str,
    recover: R,
) -> Result<InverseLink, String>
where
    R: FnOnce(&Array1<f64>) -> Option<InverseLink>,
{
    if !result.converged {
        return Err(WorkflowError::IntegrationFailed {
            reason: format!(
                "{context} did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
                result.iterations,
                result.final_value,
                result.final_grad_norm_report(),
            ),
        }
        .into());
    }
    recover(&result.rho).ok_or_else(|| {
        format!(
            "{context} produced an invalid inverse-link state at rho={:?}",
            result.rho.to_vec()
        )
    })
}

pub struct BernoulliMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub policy: crate::resource::ResourcePolicy,
}

pub struct SurvivalMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

/// Lower floor applied before taking `ln(λ)` when mapping a smoothing parameter
/// into the log-λ optimization coordinate. `λ` is non-negative by construction;
/// flooring at the smallest positive normal `f64` keeps `ln` finite for an
/// exactly-zero (fully-relaxed) penalty without perturbing any λ above the
/// denormal range.
const LOG_LAMBDA_UNDERFLOW_FLOOR: f64 = 1e-300;

/// Inner-PIRLS controls shared by the survival-transformation baseline and
/// smoothing-coordinate eval closures. The baseline geometry is mildly
/// nonlinear, so the iteration budget is generous; the convergence/step floors
/// match the working-model PIRLS contract used throughout the survival path.
const SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS: usize = 400;
const SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL: f64 = 1e-6;
const SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING: usize = 40;
const SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE: f64 = 1e-12;

pub struct LatentSurvivalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentSurvivalTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct LatentBinaryFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentBinaryTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct TransformationNormalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub response: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub config: TransformationNormalConfig,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub warm_start: Option<TransformationWarmStart>,
}

pub enum FitRequest<'a> {
    Standard(StandardFitRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleFitRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleFitRequest<'a>),
    DispersionLocationScale(DispersionLocationScaleFitRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    SurvivalTransformation(SurvivalTransformationFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest<'a>),
    LatentSurvival(LatentSurvivalFitRequest<'a>),
    LatentBinary(LatentBinaryFitRequest<'a>),
    TransformationNormal(TransformationNormalFitRequest<'a>),
}

/// Mechanical surface every `FitRequest` variant must expose. Centralising
/// these here makes the per-variant logic live with the variant's data and
/// turns the `FitRequest` impl block's case-analysis into a single dispatch
/// macro, so adding a new family is one struct + one trait impl + one
/// variant arm in [`family_dispatch!`]; the compiler then refuses to
/// silently leave a ladder unupdated.
pub trait FamilyFitRequest {
    /// Stable, short identifier ("standard", "survival-marginal-slope", …).
    /// Used as a `family={tag}` segment of the persistent warm-start key
    /// and as the structured tag in `[CACHE] …` log lines.
    const TAG: &'static str;

    /// Instance accessor for [`Self::TAG`] so callers holding a `&dyn` /
    /// generic reference (or going through [`family_dispatch!`]) can avoid
    /// the turbofish.
    fn tag(&self) -> &'static str {
        Self::TAG
    }

    /// Row count of the response / design (n).
    fn n_obs(&self) -> usize;

    /// Column count of the user-facing design matrix (p_data, before
    /// per-family basis expansion). Used in the cache-key `dims=N×D`
    /// segment for the shape-prefix lookup.
    fn n_cols(&self) -> usize;

    /// Family-shape hash inputs (link kind, baseline, frailty, etc.) —
    /// everything that changes the coefficient layout. Feeds the exact
    /// cache key.
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter);

    /// Data-independent seed-key hash inputs. Same structural identifiers
    /// as `write_shape_hash` but with the row count deliberately *excluded*
    /// so cross-validation folds / hyperparameter sweeps share a prefix.
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter);

    /// Attach the primary persistent warm-start session. Variants like
    /// `Standard` that open their own session inside the outer optimizer
    /// (see `solver/estimate.rs:2701`) implement this as a `drop(session)`
    /// no-op to avoid double-checkpointing on the same fingerprint.
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>);

    /// Attach a mirror session that receives a broadcast copy of the final
    /// `finalize` write under the seed-prefix keyspace. Variants without
    /// a mirror channel implement this as a `drop(mirror)` no-op.
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>);
}

/// Enumerates every `FitRequest` variant in **one** place. Use as
/// `family_dispatch!(expr, r => expr_using_r)`; expands to the full
/// 10-arm match. The compiler enforces exhaustiveness here too, so adding
/// a new variant produces a single hard error at this site (rather than
/// silently routing through an `_` arm of a hand-written ladder).
macro_rules! family_dispatch {
    ($scrutinee:expr, $req:ident => $body:expr) => {
        match $scrutinee {
            FitRequest::Standard($req) => $body,
            FitRequest::GaussianLocationScale($req) => $body,
            FitRequest::BinomialLocationScale($req) => $body,
            FitRequest::DispersionLocationScale($req) => $body,
            FitRequest::SurvivalLocationScale($req) => $body,
            FitRequest::SurvivalTransformation($req) => $body,
            FitRequest::BernoulliMarginalSlope($req) => $body,
            FitRequest::SurvivalMarginalSlope($req) => $body,
            FitRequest::LatentSurvival($req) => $body,
            FitRequest::LatentBinary($req) => $body,
            FitRequest::TransformationNormal($req) => $body,
        }
    };
}

impl<'a> FamilyFitRequest for StandardFitRequest<'a> {
    const TAG: &'static str = "standard";
    fn n_obs(&self) -> usize {
        self.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("standard");
        h.write_str(&format!("{:?}", self.family));
        h.write_usize(self.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869): raw `data.ncols()` is blind to the smooth
        // basis, so `s(..., type=AUTO)` candidates fit on the same data would
        // otherwise share one warm-start key and cross-seed each other with
        // incompatible β/ρ. Fold the term-collection structural shape in so
        // each candidate keys distinctly while same-topology refits still hit.
        self.spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("standard-seed");
        h.write_str(&format!("{:?}", self.family));
        h.write_usize(self.data.ncols());
        // Same topology disambiguation for the data-independent seed key: a
        // sphere candidate must not seed a torus candidate across folds.
        self.spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // The standard REML path opens its own session inside the outer
        // optimizer via `reml_state.outer_cache_session()` (see
        // `solver/estimate.rs:2701`). Accepting another one here would
        // double-checkpoint the same fingerprint — drop the redundant arc.
        drop(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        // Same reasoning: the outer optimizer handles its own finalize.
        drop(mirror);
    }
}

impl<'a> FamilyFitRequest for GaussianLocationScaleFitRequest<'a> {
    const TAG: &'static str = "gaussian-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("gauss-ls");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): the location-scale outer
        // cache session keys on this hash; an `ExactFinal` hit on it
        // short-circuits the whole fit (see
        // `outer_strategy::classify_cache_entry_for_outer`). Raw
        // `data.ncols()` is blind to the smooth basis, so two AUTO-topology
        // candidates (sphere vs euclidean) on the same data with the same
        // penalty count would collide on one key and one would return the
        // other's converged ρ as its own result. Fold each block's
        // term-collection structural shape in so each candidate keys
        // distinctly while same-topology refits still hit.
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("gauss-ls-seed");
        h.write_usize(self.data.ncols());
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for BinomialLocationScaleFitRequest<'a> {
    const TAG: &'static str = "binomial-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("binom-ls");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.link_kind));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("binom-ls-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.link_kind));
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for DispersionLocationScaleFitRequest<'a> {
    const TAG: &'static str = "dispersion-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("disp-ls");
        h.write_str(self.spec.kind.family_tag());
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_dispspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("disp-ls-seed");
        h.write_str(self.spec.kind.family_tag());
        h.write_usize(self.data.ncols());
        self.spec.meanspec.write_structural_shape_hash(h);
        self.spec.log_dispspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for SurvivalLocationScaleFitRequest<'a> {
    const TAG: &'static str = "survival-location-scale";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ls");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.inverse_link));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ls-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.inverse_link));
        self.spec.thresholdspec.write_structural_shape_hash(h);
        self.spec.log_sigmaspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // Request-level slot is mirrored into the spec slot so the family
        // fit fn sees the session when it constructs its internal
        // BlockwiseFitOptions.
        if self.cache_session.is_none() {
            self.cache_session = Some(session.clone());
        }
        if self.spec.cache_session.is_none() {
            self.spec.cache_session = Some(session);
        }
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.spec.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for SurvivalTransformationFitRequest<'a> {
    const TAG: &'static str = "survival-transformation";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-tn");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.likelihood_mode));
        h.write_str(&self.spec.time_build.basisname);
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.covariate_spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-tn-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.likelihood_mode));
        h.write_str(&self.spec.time_build.basisname);
        self.spec.covariate_spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        // SurvivalTransformation uses WorkingModelPirlsOptions (different
        // mechanism) for its inner solve; the cache session is parked at
        // the request level so future wiring through
        // `persistent_survival_transformation_key` can be unified. For now
        // that path's own exact-match warm-start fires independently.
        self.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        // The path's own `persistent_survival_transformation_key`
        // mechanism handles exact-match warm-start; mirror finalize would
        // be a duplicate — drop the unused arc.
        drop(mirror);
    }
}

impl<'a> FamilyFitRequest for BernoulliMarginalSlopeFitRequest<'a> {
    const TAG: &'static str = "bernoulli-marginal-slope";
    fn n_obs(&self) -> usize {
        self.spec.y.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("bern-ms");
        h.write_usize(self.spec.y.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("bern-ms-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for SurvivalMarginalSlopeFitRequest<'a> {
    const TAG: &'static str = "survival-marginal-slope";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ms");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        h.write_str(&format!("{:?}", self.spec.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
        match self.spec.logslopespecs.as_ref() {
            Some(specs) => {
                h.write_bool(true);
                h.write_usize(specs.len());
                for spec in specs {
                    spec.write_structural_shape_hash(h);
                }
            }
            None => h.write_bool(false),
        }
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("surv-ms-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.spec.base_link));
        h.write_str(&format!("{:?}", self.spec.frailty));
        self.spec.marginalspec.write_structural_shape_hash(h);
        self.spec.logslopespec.write_structural_shape_hash(h);
        match self.spec.logslopespecs.as_ref() {
            Some(specs) => {
                h.write_bool(true);
                h.write_usize(specs.len());
                for spec in specs {
                    spec.write_structural_shape_hash(h);
                }
            }
            None => h.write_bool(false),
        }
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for LatentSurvivalFitRequest<'a> {
    const TAG: &'static str = "latent-survival";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-surv");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-surv-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for LatentBinaryFitRequest<'a> {
    const TAG: &'static str = "latent-binary";
    fn n_obs(&self) -> usize {
        self.spec.age_entry.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-bin");
        h.write_usize(self.spec.age_entry.len());
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("lat-bin-seed");
        h.write_usize(self.data.ncols());
        h.write_str(&format!("{:?}", self.frailty));
        self.spec.meanspec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FamilyFitRequest for TransformationNormalFitRequest<'a> {
    const TAG: &'static str = "transformation-normal";
    fn n_obs(&self) -> usize {
        self.response.len()
    }
    fn n_cols(&self) -> usize {
        self.data.ncols()
    }
    fn write_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("tn");
        h.write_usize(self.response.len());
        h.write_usize(self.data.ncols());
        // Topology identity (#869, extended): see GaussianLocationScale.
        self.covariate_spec.write_structural_shape_hash(h);
    }
    fn write_seed_hash(&self, h: &mut crate::cache::Fingerprinter) {
        h.write_str("tn-seed");
        h.write_usize(self.data.ncols());
        self.covariate_spec.write_structural_shape_hash(h);
    }
    fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_session.get_or_insert(session);
    }
    fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        self.options.cache_mirror_sessions.push(mirror);
    }
}

impl<'a> FitRequest<'a> {
    /// Short stable string identifying the family variant. Used as one
    /// segment of the warm-start cache key — every fit of this family
    /// shape shares the same family tag, so a hierarchical lookup can
    /// drop trailing key segments to find near-match seeds from prior
    /// fits of the same family on related data.
    pub fn family_tag(&self) -> &'static str {
        family_dispatch!(self, r => r.tag())
    }

    /// Deterministic warm-start cache key for this request.
    ///
    /// Layout (`/`-delimited so the suffix can be peeled for graceful
    /// degradation in future hierarchical lookups):
    ///   `v{lib}/family={tag}/dims=N×D/shape={family-shape-hash}`
    ///
    /// Inputs included: library version, family tag, dataset shape, and a
    /// family-specific shape hash covering anything that changes the basis
    /// dimension or coefficient layout (formula skeleton, baseline kind,
    /// link kind, etc.). Inputs deliberately *excluded*: anything that
    /// shifts the optimum but doesn't reshape the parameter vector
    /// (convergence tolerances, ridge, etc.) — those go in the
    /// finer-grained suffix so a near-match can still be a useful seed.
    pub fn cache_key(&self) -> String {
        // Family-shape hash: a 64-bit truncation of the canonical
        // SHA-256 `Fingerprinter` digest that
        // captures dimensions / formula structure unique to this family
        // variant. Two fits with identical family-shape hashes have
        // compatible θ shapes and can warm-start from each other.
        let mut shape = crate::cache::Fingerprinter::new();
        family_dispatch!(self, r => r.write_shape_hash(&mut shape));
        let shape_hash = shape.finish_hex();
        let (nrows, ncols) = family_dispatch!(self, r => (r.n_obs(), r.n_cols()));
        format!(
            "{}/family={}/dims={}x{}/shape={}",
            crate::solver::persistent_warm_start::cache_schema_tag(),
            self.family_tag(),
            nrows,
            ncols,
            shape_hash,
        )
    }

    /// Data-independent cache key suitable as a *seed* lookup when the
    /// exact-key cache misses. Two fits with the same family / link /
    /// baseline / column shape share this key even if their row counts
    /// differ (cross-validation folds, hyperparameter sweeps, anything
    /// re-fitting structurally identical models on related data).
    ///
    /// Save always goes to [`Self::cache_key`] (exact). The dispatcher
    /// looks up this prefix only when the exact key has no entry — so a
    /// near-match never silently overrides a perfect match.
    pub fn cache_seed_key(&self) -> String {
        let mut shape = crate::cache::Fingerprinter::new();
        family_dispatch!(self, r => r.write_seed_hash(&mut shape));
        format!(
            "{}/family={}/seed/{}",
            crate::solver::persistent_warm_start::cache_schema_tag(),
            self.family_tag(),
            shape.finish_hex(),
        )
    }

    /// Attach a mirror cache session that receives a broadcast copy of
    /// the final-result `finalize` write. Used by the dispatcher to write
    /// the converged ρ to the data-independent seed prefix keyspace so a
    /// future fit with related-but-not-identical structure can warm-start
    /// from this run. Checkpoints are mirrored through the same rate-limited
    /// session path as the primary cache, so interrupted runs can still seed
    /// later related fits instead of waiting for a successful final result.
    pub fn attach_cache_mirror(&mut self, mirror: std::sync::Arc<crate::cache::Session>) {
        family_dispatch!(self, r => <_ as FamilyFitRequest>::attach_cache_mirror(r, mirror))
    }

    /// Attach a warm-start cache session to this request.
    ///
    /// Threads the session into the variant's BlockwiseFitOptions
    /// (`request.options.cache_session`) or top-level `cache_session`
    /// field, whichever is the variant's natural slot. Idempotent —
    /// existing sessions are not overwritten so callers can pre-attach.
    pub fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        family_dispatch!(self, r => <_ as FamilyFitRequest>::attach_cache_session(r, session))
    }
}

pub struct StandardFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub saved_link_state: FittedLinkState,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

pub struct SurvivalLocationScaleFitResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub inverse_link: InverseLink,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

pub struct SurvivalTransformationFitResult {
    pub fit: UnifiedFitResult,
    pub resolvedspec: TermCollectionSpec,
    pub baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    pub likelihood_mode: crate::families::survival_construction::SurvivalLikelihoodMode,
    /// Persistable snapshot of the time basis used during the fit. Replaces
    /// six previously flat fields (basisname / degree / knots / keep_cols /
    /// smooth_lambda / anchor) so the FFI save path consumes a single
    /// source-of-truth value rather than threading siblings independently.
    pub time_basis: crate::families::survival_construction::SavedSurvivalTimeBasis,
    pub time_base_ncols: usize,
    pub baseline_timewiggle: Option<TimeWiggleBlockInput>,
}

struct SurvivalLocationScaleProfile {
    fit: SurvivalLocationScaleTermFitResult,
    inverse_link: InverseLink,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
}

impl SurvivalLocationScaleProfile {
    fn objective(&self) -> f64 {
        self.fit.fit.reml_score
    }

    fn into_result(self) -> SurvivalLocationScaleFitResult {
        SurvivalLocationScaleFitResult {
            fit: self.fit,
            inverse_link: self.inverse_link,
            wiggle_knots: self.wiggle_knots,
            wiggle_degree: self.wiggle_degree,
        }
    }
}

pub enum FitResult {
    Standard(StandardFitResult),
    GaussianLocationScale(GaussianLocationScaleFitResult),
    BinomialLocationScale(BinomialLocationScaleFitResult),
    DispersionLocationScale(DispersionLocationScaleFitResult),
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    SurvivalTransformation(SurvivalTransformationFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitResult),
    LatentSurvival(LatentSurvivalTermFitResult),
    LatentBinary(LatentBinaryTermFitResult),
    TransformationNormal(TransformationNormalFitResult),
}

/// Result of a dispersion-channel GAMLSS location-scale fit (#913). Wraps the
/// shared two-block [`BlockwiseTermFitResult`] (mean + log-precision designs
/// and coefficients) plus the family kind so the save path can stamp the right
/// likelihood. These families have no link-wiggle and no response
/// standardization, so the result is a thin wrapper.
pub struct DispersionLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub kind: DispersionFamilyKind,
}

fn resolved_wiggle_inverse_link(
    spec: &LikelihoodSpec,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, String> {
    let resolved = match fit.fitted_link_state(spec).map_err(|e| e.to_string())? {
        FittedLinkState::Standard(Some(link)) => InverseLink::Standard(link),
        FittedLinkState::Standard(None) => fallback.clone(),
        FittedLinkState::LatentCLogLog { state } => InverseLink::LatentCLogLog(state),
        FittedLinkState::Sas { state, .. } => InverseLink::Sas(state),
        FittedLinkState::BetaLogistic { state, .. } => InverseLink::BetaLogistic(state),
        FittedLinkState::Mixture { state, .. } => InverseLink::Mixture(state),
    };
    require_inverse_link_supports_joint_wiggle(&resolved, "standard link wiggle")?;
    Ok(resolved)
}

fn deviation_block_config_from_formula_linkwiggle(
    wiggle: &LinkWiggleFormulaSpec,
) -> DeviationBlockConfig {
    let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
    DeviationBlockConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_order: *wiggle.penalty_orders.iter().max().unwrap_or(&2),
        penalty_orders: wiggle.penalty_orders.clone(),
        double_penalty: wiggle.double_penalty,
        monotonicity_eps: defaults.monotonicity_eps,
    }
}

struct MarginalSlopeDeviationRouting {
    score_warp: Option<DeviationBlockConfig>,
    link_dev: Option<DeviationBlockConfig>,
}

fn route_marginal_slope_deviation_blocks(
    main_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    logslope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
) -> Result<MarginalSlopeDeviationRouting, String> {
    Ok(MarginalSlopeDeviationRouting {
        score_warp: logslope_linkwiggle.map(deviation_block_config_from_formula_linkwiggle),
        link_dev: main_linkwiggle.map(deviation_block_config_from_formula_linkwiggle),
    })
}

fn fixed_gaussian_shift_frailty_from_spec(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<FrailtySpec, String> {
    match frailty {
        FrailtySpec::None => Ok(FrailtySpec::None),
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        } => Ok(FrailtySpec::GaussianShift {
            sigma_fixed: Some(*sigma),
        }),
        FrailtySpec::GaussianShift { sigma_fixed: None } => Err(WorkflowError::MissingDependency {
            reason: format!("{context} currently requires a fixed GaussianShift sigma"),
        }
        .into()),
        FrailtySpec::HazardMultiplier { .. } => Err(WorkflowError::MissingDependency {
            reason: format!("{context} requires FrailtySpec::GaussianShift or no frailty"),
        }
        .into()),
    }
}

fn fit_standard_model(request: StandardFitRequest<'_>) -> Result<StandardFitResult, String> {
    let fitted = if let Some(latent_coord) = request.latent_coord.as_ref() {
        if !request.coefficient_groups.is_empty() || !request.penalty_block_gamma_priors.is_empty()
        {
            return Err("latent-coordinate standard fits do not support coefficient_groups or penalty_block_gamma_priors in the same request".to_string());
        }
        fit_term_collectionwith_latent_coord_optimization(
            request.data.view(),
            request.y.clone(),
            request.weights.clone(),
            request.offset.clone(),
            &request.spec,
            latent_coord,
            request.family.clone(),
            &request.options,
        )
        .map_err(|e| e.to_string())?
    } else if !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        let fitted = fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors(
            request.data.view(),
            request.y.view(),
            request.weights.view(),
            request.offset.view(),
            &request.spec,
            &request.coefficient_groups,
            &request.penalty_block_gamma_priors,
            request.family.clone(),
            &request.options,
        )
        .map_err(|e| e.to_string())?;
        crate::terms::smooth::FittedTermCollectionWithSpec {
            fit: fitted.fit,
            design: fitted.design,
            resolvedspec: request.spec.clone(),
            adaptive_diagnostics: fitted.adaptive_diagnostics,
        }
    } else {
        fit_term_collectionwith_spatial_length_scale_optimization(
            request.data.view(),
            request.y.clone(),
            request.weights.clone(),
            request.offset.clone(),
            &request.spec,
            request.family.clone(),
            &request.options,
            &request.kappa_options,
        )
        .map_err(|e| e.to_string())?
    };

    let result = StandardFitResult {
        saved_link_state: fitted.fit.fitted_link.clone(),
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec: fitted.resolvedspec,
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        wiggle_knots: None,
        wiggle_degree: None,
    };

    let Some(wiggle) = request.wiggle else {
        return Ok(result);
    };
    // `StandardBinomialWiggleConfig` now carries `refit_options` directly, so
    // the previous "pilot config present, blockwise options missing" failure
    // state (#320) is unrepresentable at the type level.
    let wiggle_options = wiggle.refit_options.clone();
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(&request.family, &result.fit, &wiggle.link_kind)?;
    let selected_wiggle_basis = select_binomial_mean_link_wiggle_basis_from_pilot(
        &result.design,
        &result.fit,
        &WiggleBlockConfig {
            degree: wiggle.wiggle.degree,
            num_internal_knots: wiggle.wiggle.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle.wiggle.double_penalty,
        },
        &wiggle.wiggle.penalty_orders,
    )?;

    // A penalized, monotone-constrained link-offset spline shrinks to zero at
    // large smoothing, so the no-wiggle pilot fit (`result`) is the *exact*
    // large-`λ` limit of the wiggle model — the wiggle model contains the
    // baseline as a limiting case. The wiggle refit is a coupled joint
    // Newton solve (`BinomialMeanWiggleFamily`) on top of that pilot; on the
    // hardest binomial regimes it can still fail to certify KKT convergence:
    // the I-spline warp `q = η + B(η)·β_w` can drive the linear predictor
    // toward link saturation, where the per-cycle data curvature collapses
    // and the joint trust region shrinks faster than the active-set QP can
    // pin the binding monotonicity rows (gam#872). Aborting the entire fit
    // there is wrong: a model that *contains* a fittable baseline must never
    // be less fittable than that baseline. Fall back to the pilot — a valid,
    // finite-deviance fit no worse than the wiggle model's own limit — rather
    // than surfacing an `IntegrationError` to the caller. (The separate
    // divergence failure mode, where the unconditional Jeffreys/Firth
    // augmentation blew the augmented objective up to ~1e9 on this path, is
    // fixed at the root by `BinomialMeanWiggleFamily::joint_jeffreys_term_required
    // = false`; this fallback only catches the residual trust-region/active-set
    // non-convergence.)
    let solved = match fit_binomial_mean_wiggle_terms_with_selected_basis(
        request.data.view(),
        &result.resolvedspec,
        &result.design,
        &result.fit,
        &request.y,
        &request.weights,
        wiggle_link_kind,
        selected_wiggle_basis,
        &wiggle_options,
        &request.kappa_options,
    ) {
        Ok(solved) => solved,
        Err(e) => {
            log::warn!(
                "[linkwiggle] binomial mean link-wiggle joint solve did not converge ({e}); \
                 falling back to the no-wiggle baseline fit (the large-smoothing limit of the \
                 penalized wiggle model, which contains it as a limiting case)"
            );
            return Ok(result);
        }
    };

    Ok(StandardFitResult {
        saved_link_state: result.saved_link_state,
        fit: solved.fit,
        design: solved.design,
        resolvedspec: solved.resolvedspec,
        adaptive_diagnostics: result.adaptive_diagnostics,
        wiggle_knots: Some(solved.wiggle_knots),
        wiggle_degree: Some(solved.wiggle_degree),
    })
}

/// Broken-out pieces of a location-scale fit request, family-agnostic.
///
/// Both the Gaussian and binomial location-scale requests are structurally the
/// same — a borrowed data matrix, a family-specific term spec, an optional link
/// wiggle config, and the two option bundles. The shared wiggle-pilot engine
/// ([`fit_location_scale_with_optional_wiggle`]) consumes these parts; each
/// family's request type lowers itself into them via
/// [`LocationScaleWorkflowAdapter::into_parts`].
struct LocationScaleWorkflowParts<'a, S> {
    data: ArrayView2<'a, f64>,
    spec: S,
    wiggle: Option<LinkWiggleConfig>,
    options: BlockwiseFitOptions,
    kappa_options: SpatialLengthScaleOptimizationOptions,
}

/// Family-specific glue for the shared location-scale wiggle-pilot workflow.
///
/// The workflow policy (pilot fit — which also enforces any family wiggle
/// compatibility guard — → select link wiggle basis from the pilot → refit with
/// the selected wiggle → extract `beta_link_wiggle` and assemble; otherwise the
/// plain non-wiggle fit) is identical across Gaussian and binomial
/// location-scale models — only the family fit/select/refit functions and result
/// type differ (#430). An adapter supplies exactly those family-specific
/// operations; the engine owns the policy.
trait LocationScaleWorkflowAdapter {
    /// The owned term spec for this family (`GaussianLocationScaleTermSpec` /
    /// `BinomialLocationScaleTermSpec`).
    type Spec;
    /// The borrowed request type the public model entry point receives.
    type Request<'a>;
    /// The family-specific fit result the engine assembles.
    type Result;

    /// Lower the borrowed request into the family-agnostic workflow parts.
    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec>;

    /// Pilot fit on the bare (non-wiggle) spec, used to seed the wiggle-basis
    /// selector. This is the first work the wiggle path performs, so any
    /// family-specific wiggle compatibility guard (e.g. the binomial inverse
    /// link must support a joint wiggle refit) is enforced here before fitting.
    /// The adapter clones whatever spec fields the pilot consumes so the caller
    /// retains ownership of `spec` for the subsequent refit.
    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String>;

    /// Select the link-wiggle basis from the pilot, then refit the full model
    /// with that selected wiggle block. Consumes `spec`.
    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String>;

    /// Plain non-wiggle fit, used when no wiggle config is present. Consumes
    /// `spec`.
    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String>;

    /// Assemble the family result from a non-wiggle fit (knots/degree/wiggle
    /// coefficients all absent).
    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result;

    /// Assemble the family result from a wiggle refit, carrying the selected
    /// knots/degree and the extracted `beta_link_wiggle` block.
    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result;
}

/// Shared wiggle-pilot workflow for Gaussian and binomial location-scale models
/// (#430). The single source of truth for the policy; families differ only via
/// their [`LocationScaleWorkflowAdapter`].
fn fit_location_scale_with_optional_wiggle<A: LocationScaleWorkflowAdapter>(
    request: A::Request<'_>,
) -> Result<A::Result, String> {
    let LocationScaleWorkflowParts {
        data,
        spec,
        wiggle,
        options,
        kappa_options,
    } = A::into_parts(request);

    let Some(wiggle_cfg) = wiggle else {
        let fit = A::fit_plain(data, spec, &options, &kappa_options)?;
        return Ok(A::assemble_plain(fit));
    };

    let pilot = A::fit_pilot(data, &spec, &options, &kappa_options)?;
    let solved =
        A::refit_with_selected_wiggle(data, spec, &pilot, &wiggle_cfg, &options, &kappa_options)?;

    // The selected link-wiggle basis is appended as the third blockwise term
    // (after the mean/threshold and log-σ blocks), so its coefficients live in
    // block 2 of the refit.
    let fit = solved.fit.fit;
    let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
    let assembled_fit = BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
        fit,
        meanspec_resolved: solved.fit.meanspec_resolved,
        noisespec_resolved: solved.fit.noisespec_resolved,
        mean_design: solved.fit.mean_design,
        noise_design: solved.fit.noise_design,
    })?;
    Ok(A::assemble_with_wiggle(
        assembled_fit,
        solved.wiggle_knots,
        solved.wiggle_degree,
        beta_link_wiggle,
    ))
}

/// Gaussian location-scale adapter for the shared wiggle-pilot workflow.
struct GaussianLocationScaleWorkflow;

impl LocationScaleWorkflowAdapter for GaussianLocationScaleWorkflow {
    type Spec = GaussianLocationScaleTermSpec;
    type Request<'a> = GaussianLocationScaleFitRequest<'a>;
    type Result = GaussianLocationScaleFitResult;

    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec> {
        LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        }
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        // Gaussian location-scale uses an identity mean link; the joint wiggle
        // refit is always admissible, so the pilot fits with no extra guard.
        fit_gaussian_location_scale_terms(
            data,
            GaussianLocationScaleTermSpec {
                y: spec.y.clone(),
                weights: spec.weights.clone(),
                meanspec: spec.meanspec.clone(),
                log_sigmaspec: spec.log_sigmaspec.clone(),
                mean_offset: spec.mean_offset.clone(),
                log_sigma_offset: spec.log_sigma_offset.clone(),
            },
            options,
            kappa_options,
        )
    }

    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String> {
        let selected_wiggle_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        fit_gaussian_location_scale_terms_with_selected_wiggle(
            data,
            spec,
            selected_wiggle_basis,
            options,
            kappa_options,
        )
    }

    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        fit_gaussian_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result {
        GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
            // The wiggle-pilot workflow fits in standardized response units; the
            // Gaussian model wrapper (`fit_gaussian_location_scale_model`) maps
            // the coefficients back to raw units and overwrites this with the
            // applied factor. `1.0` here is the identity (no standardization).
            response_scale: 1.0,
        }
    }

    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result {
        GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: Some(wiggle_knots),
            wiggle_degree: Some(wiggle_degree),
            beta_link_wiggle,
            // See `assemble_plain`: raw-unit remapping happens in the Gaussian
            // model wrapper, which overwrites this with the applied factor.
            response_scale: 1.0,
        }
    }
}

/// Binomial location-scale adapter for the shared wiggle-pilot workflow.
struct BinomialLocationScaleWorkflow;

impl LocationScaleWorkflowAdapter for BinomialLocationScaleWorkflow {
    type Spec = BinomialLocationScaleTermSpec;
    type Request<'a> = BinomialLocationScaleFitRequest<'a>;
    type Result = BinomialLocationScaleFitResult;

    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec> {
        LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        }
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        // Binomial location-scale requires an inverse link that supports the
        // joint link-wiggle refit; gate it before any fitting work (the pilot
        // runs only on the wiggle path).
        require_inverse_link_supports_joint_wiggle(
            &spec.link_kind,
            "binomial location-scale link wiggle",
        )?;
        fit_binomial_location_scale_terms(
            data,
            BinomialLocationScaleTermSpec {
                y: spec.y.clone(),
                weights: spec.weights.clone(),
                link_kind: spec.link_kind.clone(),
                thresholdspec: spec.thresholdspec.clone(),
                log_sigmaspec: spec.log_sigmaspec.clone(),
                threshold_offset: spec.threshold_offset.clone(),
                log_sigma_offset: spec.log_sigma_offset.clone(),
            },
            options,
            kappa_options,
        )
    }

    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String> {
        let selected_wiggle_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        fit_binomial_location_scale_terms_with_selected_wiggle(
            data,
            spec,
            selected_wiggle_basis,
            options,
            kappa_options,
        )
    }

    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        fit_binomial_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result {
        BinomialLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        }
    }

    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result {
        BinomialLocationScaleFitResult {
            fit,
            wiggle_knots: Some(wiggle_knots),
            wiggle_degree: Some(wiggle_degree),
            beta_link_wiggle,
        }
    }
}

/// Population standard deviation of a response column (divide by `n`, not
/// `n-1`).
///
/// This is the single response-standardization factor for the Gaussian
/// location-scale path, so the standardized fit is identical whether the
/// request arrives from the library (`fit_from_formula` →
/// `materialize_location_scale`), the FFI marshaller, or the CLI.
fn gaussian_response_sample_std(v: ArrayView1<'_, f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().copied().sum::<f64>() / n;
    let var = v
        .iter()
        .copied()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1.0);
    var.max(0.0).sqrt()
}

/// Map a Gaussian location-scale fit fitted in *standardized* response units
/// (`y / response_scale`) back to **raw** response units, in place.
///
/// The internal fit solves with `y_internal = y / s` where `s = response_scale`.
/// Reconstructing raw outputs requires
///
///   μ_raw  = s · μ_internal           ⇒ scale every Location/Mean coefficient by `s`,
///   σ_raw  = s · σ_internal           ⇒ since σ = b + exp(η_σ), shifting the
///                                         log-σ **intercept** by `+ln(s)` turns
///                                         `b + exp(η)` into `b + s·exp(η)`; the
///                                         multiplicative `exp(η)` part is now
///                                         correct, but the **floor must also be
///                                         scaled** to `s·b` so the reconstructed
///                                         σ = s·b + exp(η_raw) = s·σ_internal is
///                                         response-scale-equivariant (#884). The
///                                         floor cannot ride the intercept shift
///                                         (it sits outside the exp), so consumers
///                                         reconstruct with floor `s·LOGB_SIGMA_FLOOR`
///                                         (see `GaussianLocationScalePredictor`).
///
/// The link-wiggle lives on the mean (identity) channel, so its knots and
/// coefficients scale by `s` exactly like the Location block. Doing the remap
/// here — once, inside the single Gaussian model entry point — makes every
/// caller (library `fit_from_formula`, the FFI marshaller, the CLI save path)
/// observe raw-unit coefficients with **no** additional per-call rescaling,
/// which is what keeps the σ-floor scale-relative (κ ≈ 1) without leaving the
/// reconstruction half-applied in any one path.
fn rescale_gaussian_location_scale_to_raw(
    result: &mut GaussianLocationScaleFitResult,
    response_scale: f64,
) {
    use crate::estimate::BlockRole;

    let s = response_scale;
    let ln_s = s.ln();
    // Intercept columns of the log-σ (Scale) design, expressed as offsets into
    // the Scale block's coefficient vector (the block β is laid out in noise
    // design column order). These are the only constant directions in η_σ
    // (smooths are sum-to-zero), so shifting them adds `ln(s)` to η_σ uniformly.
    let scale_intercept_range = result.fit.noise_design.intercept_range.clone();

    // Per-block coefficient surgery. `blocks` is authoritative for
    // `block_by_role` (predict, the FFI payload, and the reference tests all
    // read it), and the joint `beta` / `block_states` mirror it.
    let mut joint_offset = 0usize;
    for (block_idx, block) in result.fit.fit.blocks.iter_mut().enumerate() {
        let block_len = block.beta.len();
        match block.role {
            BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => {
                block.beta.mapv_inplace(|v| v * s);
                if result.fit.fit.beta.len() >= joint_offset + block_len {
                    for i in 0..block_len {
                        result.fit.fit.beta[joint_offset + i] *= s;
                    }
                }
                if let Some(state) = result.fit.fit.block_states.get_mut(block_idx) {
                    state.beta.mapv_inplace(|v| v * s);
                }
            }
            BlockRole::Scale => {
                for col in scale_intercept_range.clone() {
                    if col < block.beta.len() {
                        block.beta[col] += ln_s;
                    }
                    let joint_col = joint_offset + col;
                    if joint_col < result.fit.fit.beta.len() {
                        result.fit.fit.beta[joint_col] += ln_s;
                    }
                    if let Some(state) = result.fit.fit.block_states.get_mut(block_idx)
                        && col < state.beta.len()
                    {
                        state.beta[col] += ln_s;
                    }
                }
            }
            BlockRole::Time | BlockRole::Threshold => {
                // Survival-only roles are never produced by the Gaussian
                // location-scale path; leave them untouched if ever present.
            }
        }
        joint_offset += block_len;
    }

    // The link-wiggle knots/coefficients live on the mean (identity) channel.
    if let Some(knots) = result.wiggle_knots.as_mut() {
        knots.mapv_inplace(|v| v * s);
    }
    if let Some(beta_w) = result.beta_link_wiggle.as_mut() {
        for coef in beta_w.iter_mut() {
            *coef *= s;
        }
    }

    // Conditional/corrected covariances were computed in standardized units.
    // Var(s·β_loc) = s²·Var(β_loc); the Scale block only had a constant added to
    // its intercept, which does not change its (co)variance. Cross terms between
    // a Location and the Scale block pick up one factor of `s`. This is exactly
    // a per-coefficient diagonal scaling D·Σ·D with D = s on Location/Mean/Wiggle
    // rows and D = 1 on Scale rows.
    let mut row_factors: Vec<f64> = Vec::new();
    for block in &result.fit.fit.blocks {
        let f = match block.role {
            BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => s,
            BlockRole::Scale | BlockRole::Time | BlockRole::Threshold => 1.0,
        };
        row_factors.extend(std::iter::repeat_n(f, block.beta.len()));
    }
    let rescale_cov = |cov: &mut Array2<f64>| {
        let m = cov.nrows().min(cov.ncols()).min(row_factors.len());
        for i in 0..m {
            for j in 0..m {
                cov[[i, j]] *= row_factors[i] * row_factors[j];
            }
        }
    };
    if let Some(cov) = result.fit.fit.covariance_conditional.as_mut() {
        rescale_cov(cov);
    }
    if let Some(cov) = result.fit.fit.covariance_corrected.as_mut() {
        rescale_cov(cov);
    }

    // The residual-scale summary `standard_deviation` is a response-units
    // quantity; the internal fit reports it in standardized units, so map it
    // back. `max_abs_eta` is the mean-channel η magnitude (raw μ = s·μ_internal).
    result.fit.fit.standard_deviation *= s;
    result.fit.fit.max_abs_eta *= s;

    // Change-of-variables correction for the likelihood-scale summaries. The
    // internal fit maximizes the density of y_internal = y/s; the raw-response
    // density is p_raw(y) = p_internal(y/s)/s, so per observation
    // log p_raw = log p_internal − ln(s). The deviance (−2·loglik) and the
    // REML/LAML objective (which carries the data log-likelihood) shift
    // accordingly. This keeps reported log-likelihood / deviance / REML in raw
    // response units, matching what an un-standardized fit would report.
    // The number of observations is the fitted eta length for any parameter
    // block. Use the first block state instead of optional geometry so the
    // public objective fields stay in one unit system even when covariance or
    // ALO geometry was not retained.
    if let Some(n_obs) = result
        .fit
        .fit
        .block_states
        .first()
        .map(|state| state.eta.len() as f64)
        .filter(|&n| n > 0.0)
    {
        let ln_s = s.ln();
        result.fit.fit.log_likelihood -= n_obs * ln_s;
        result.fit.fit.deviance += 2.0 * n_obs * ln_s;
        result.fit.fit.reml_score += n_obs * ln_s;
        result.fit.fit.penalized_objective += n_obs * ln_s;
    }

    result.response_scale = s;
}

fn fit_gaussian_location_scale_model(
    mut request: GaussianLocationScaleFitRequest<'_>,
) -> Result<GaussianLocationScaleFitResult, String> {
    // Standardize the response so the fixed log-σ soft floor
    // `LOGB_SIGMA_FLOOR = 0.01` is scale-relative (≈ 1 % of the response
    // spread) rather than absolute. Without this the link σ = 0.01 + exp(η)
    // gives κ = dlogσ/dη = exp(η)/(0.01+exp(η)) < 1 whenever the raw σ is small,
    // and the scale-block Fisher information 2κ²a is strictly below gamlss's
    // floorless 2a, systematically over-smoothing the log-σ envelope
    // (#686 #688 #684 #685 #687). Fitting on y/s restores κ ≈ 1.
    let response_scale = gaussian_response_sample_std(request.spec.y.view()).max(1e-6);
    if response_scale != 1.0 {
        request.spec.y.mapv_inplace(|v| v / response_scale);
        // The mean (identity-link) offset rides in the same units as y; the
        // log-σ offset is on the log-scale axis and is unaffected by the
        // multiplicative response rescale.
        request
            .spec
            .mean_offset
            .mapv_inplace(|v| v / response_scale);
    }

    let mut result =
        fit_location_scale_with_optional_wiggle::<GaussianLocationScaleWorkflow>(request)?;

    rescale_gaussian_location_scale_to_raw(&mut result, response_scale);
    Ok(result)
}

fn fit_dispersion_location_scale_model(
    request: DispersionLocationScaleFitRequest<'_>,
) -> Result<DispersionLocationScaleFitResult, String> {
    let kind = request.spec.kind;
    let fit = fit_dispersion_glm_location_scale_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )?;
    Ok(DispersionLocationScaleFitResult { fit, kind })
}

fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleFitRequest<'_>,
) -> Result<BinomialLocationScaleFitResult, String> {
    fit_location_scale_with_optional_wiggle::<BinomialLocationScaleWorkflow>(request)
}

fn survival_working_reml_score(state: &crate::pirls::WorkingState) -> f64 {
    0.5 * (state.deviance + state.penalty_term)
}

/// Recover the fitted Weibull baseline config from the anchor-CENTERED linear
/// `[1, log t]` time-basis coefficients.
///
/// The fit centers the time basis at the survival time anchor
/// (`center_survival_time_designs_at_anchor`), which zeroes the constant column,
/// so the constant-column coefficient `beta[0]` is UNIDENTIFIED (left at its
/// stale seed). The identified baseline the model carries is
/// `eta(t) = beta[1] * (log t - log anchor)`, exactly the Weibull form
/// `eta(t) = shape * (log t - log scale)` with `shape = beta[1]` and
/// `scale = anchor`. Reconstructing `scale` from `beta[0]` (the old
/// `exp(-beta[0]/shape)`) reads the stale constant column and produces a wrong
/// saved scale, misleading every consumer that rebuilds `H0(t) = (t/scale)^shape`
/// from the saved scale (e.g. competing-risks CIF). Recover `scale` from the
/// identified anchor instead (issue #899).
fn fitted_weibull_baseline_from_linear_time_beta(
    beta: &Array1<f64>,
    anchor: f64,
) -> Option<crate::families::survival_construction::SurvivalBaselineConfig> {
    if beta.len() < 2 {
        return None;
    }
    let shape = beta[1];
    if !shape.is_finite() || shape <= 0.0 {
        return None;
    }
    if !anchor.is_finite() || anchor <= 0.0 {
        return None;
    }
    let scale = anchor;
    Some(
        crate::families::survival_construction::SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(scale),
            shape: Some(shape),
            rate: None,
            makeham: None,
        },
    )
}

/// Penalized effective degrees of freedom for a survival transformation fit.
///
/// Uses exactly the mgcv definition `edf_total = p − Σ_k λ_k·tr(H⁻¹ S_k)`, where
/// `H` is the converged penalized Hessian `X'W_HX + S(λ) + ridge·I` (held in
/// `state.hessian`) and `S_k` is the penalty matrix of block `k` (without its
/// `λ_k` factor, which is applied here). The per-block edf is
/// `edf_k = block_cols_k − λ_k·tr(H⁻¹ S_k)`, clamped to `[0, block_cols_k]`.
///
/// Returned alongside the dense penalized Hessian so the caller can populate the
/// inference block (`edf_total`, `edf_by_block`, `penalized_hessian`). This is the
/// same trace formula `estimate.rs` uses for the standard GAM path; the survival
/// path runs its own `runworking_model_pirls` optimizer and therefore never
/// reached that block, leaving edf uncomputed (issue #565).
fn survival_transformation_edf(
    state: &crate::pirls::WorkingState,
    penalty_blocks: &[PenaltyBlock],
) -> Result<(f64, Vec<f64>, Array2<f64>), String> {
    let h_dense = state.hessian.to_dense();
    let p = h_dense.nrows();
    let h_sym = crate::linalg::matrix::SymmetricMatrix::Dense(h_dense.clone());
    // Sparse-aware factorization with ridge retry (mirrors estimate.rs) so a
    // marginally indefinite Hessian at a boundary-constrained optimum still
    // yields a usable trace rather than aborting the whole fit.
    let factor = {
        let scale = h_sym.max_abs_diag();
        let min_step = scale * 1e-10;
        let mut ridge = 0.0_f64;
        let mut attempts = 0_usize;
        loop {
            let candidate = if ridge > 0.0 {
                h_sym.addridge(ridge).unwrap_or_else(|_| h_sym.clone())
            } else {
                h_sym.clone()
            };
            if let Ok(f) = candidate.factorize() {
                break f;
            }
            attempts += 1;
            if attempts >= 8 {
                return Err("survival edf: penalized Hessian could not be factorized".to_string());
            }
            ridge = if ridge <= 0.0 { min_step } else { ridge * 10.0 };
        }
    };
    let mut edf_by_block = vec![0.0_f64; penalty_blocks.len()];
    let mut total_trace = 0.0_f64;
    for (kk, block) in penalty_blocks.iter().enumerate() {
        let block_cols = block.range.end - block.range.start;
        if block.lambda <= 0.0 || block_cols == 0 {
            edf_by_block[kk] = block_cols as f64;
            continue;
        }
        // RHS = S_k embedded into the full p×block_cols layout: column j holds
        // column j of S_k placed in the block rows. Solving H Z = RHS gives the
        // block columns of H⁻¹ S_full, whose block-diagonal entries sum to
        // tr(H⁻¹ S_k).
        let mut rhs = Array2::<f64>::zeros((p, block_cols));
        for c in 0..block_cols {
            for r in 0..block_cols {
                rhs[[block.range.start + r, c]] = block.matrix[[r, c]];
            }
        }
        let sol = factor
            .solvemulti(&rhs)
            .map_err(|e| format!("survival edf trace solve failed: {e}"))?;
        let mut trace = 0.0_f64;
        for j in 0..block_cols {
            trace += sol[[block.range.start + j, j]];
        }
        let lam_trace = block.lambda * trace;
        total_trace += lam_trace;
        edf_by_block[kk] = (block_cols as f64 - lam_trace).clamp(0.0, block_cols as f64);
    }
    let edf_total = (p as f64 - total_trace).clamp(0.0, p as f64);
    if !edf_total.is_finite() || edf_by_block.iter().any(|v| !v.is_finite()) {
        return Err("survival edf: non-finite effective degrees of freedom".to_string());
    }
    Ok((edf_total, edf_by_block, h_dense))
}

/// REML/LAML smoothing-parameter selection for the single-cause transformation
/// survival baseline (issue #563).
///
/// The transformation path solves a constrained PIRLS (`γ ≥ 0` I-spline box) at
/// a fixed time-penalty `λ`, which oversmooths: with `λ` pinned at its seed the
/// monotone baseline collapses toward an affine log-cumulative-hazard and cannot
/// recover real curvature (e.g. Gompertz convexity). This routine wraps that
/// inner solve in a proper outer LAML optimization over `ρ = log λ` for the
/// `num_smoothing` time-penalty blocks (the trailing stabilization ridge is held
/// fixed), exactly as the standard GAM path and mgcv/scam do. The inner solve
/// still honors the structural box at every candidate `λ`, so the constrained
/// optimum stays valid; only the outer `λ` becomes data-adaptive.
///
/// `model` is the working model at the seed `λ`; it is cloned per candidate so
/// the proposal never corrupts the warm model. The returned vector has one
/// `λ_k` per penalty block (smoothing blocks at REML-selected values, the ridge
/// at its fixed seed). Returns `None` when there are no smoothing blocks to
/// select (e.g. the Weibull linear-time path), so the caller keeps the seed.
fn optimize_survival_transformation_smoothing(
    model: &crate::families::survival::WorkingModelSurvival,
    penalty_blocks: &[PenaltyBlock],
    num_smoothing: usize,
    beta0: &Array1<f64>,
    structural_lower_bounds: Option<&Array1<f64>>,
) -> Result<Option<Vec<f64>>, String> {
    use crate::solver::outer_strategy::{
        Derivative, HessianResult, OuterEval, OuterProblem, SolverClass,
    };
    if num_smoothing == 0 {
        return Ok(None);
    }
    // Full λ vector (smoothing blocks + fixed ridge), used to rebuild each
    // candidate model. The ridge entries (indices >= num_smoothing) are frozen.
    let seed_lambdas: Vec<f64> = penalty_blocks.iter().map(|b| b.lambda).collect();
    let seed_rho = Array1::from_iter(
        seed_lambdas
            .iter()
            .take(num_smoothing)
            .map(|&l| l.max(1e-12).ln()),
    );

    // Evaluate the LAML objective and ρ-gradient at a smoothing-ρ proposal:
    // set the smoothing λ, re-run the constrained inner PIRLS, evaluate the
    // unified survival LAML, and project the gradient onto the smoothing
    // coordinates (the trailing ridge gradient component is discarded since the
    // ridge is fixed).
    let eval_at = |rho_smooth: &Array1<f64>| -> Result<(f64, Array1<f64>), String> {
        let mut candidate = model.clone();
        let mut lambdas = seed_lambdas.clone();
        for k in 0..num_smoothing {
            lambdas[k] = rho_smooth[k].exp();
        }
        candidate
            .set_penalty_lambdas(&lambdas)
            .map_err(|e| e.to_string())?;
        let opts = crate::pirls::WorkingModelPirlsOptions {
            max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
            convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
            adaptive_kkt_tolerance: None,
            max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
            min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
            firth_bias_reduction: false,
            coefficient_lower_bounds: structural_lower_bounds.cloned(),
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };
        let summary = crate::pirls::runworking_model_pirls(
            &mut candidate,
            crate::types::Coefficients::new(beta0.clone()),
            &opts,
            |_| {},
        )
        .map_err(|err| format!("survival smoothing PIRLS failed: {err}"))?;
        let beta = summary.beta.as_ref().to_owned();
        let state = candidate
            .update_state(&beta)
            .map_err(|err| format!("survival smoothing state eval failed: {err}"))?;
        // Active-penalty ρ over ALL active blocks (smoothing + fixed ridge), in
        // block order, as the unified survival LAML evaluator requires. The
        // candidate's λ are exactly `lambdas` (smoothing entries from the
        // proposal, ridge entries frozen), so build ρ from that vector directly.
        let full_rho = Array1::from_iter(lambdas.iter().filter(|&&l| l > 0.0).map(|&l| l.ln()));
        let (cost, grad_full) = candidate
            .unified_lamlobjective_and_rhogradient(&beta, &state, &full_rho)
            .map_err(|err| format!("survival LAML evaluation failed: {err}"))?;
        // Project onto the smoothing coordinates. The active-block enumeration
        // lists the smoothing blocks first (they are constructed first and the
        // ridge is appended last), so the leading `num_smoothing` gradient
        // entries are exactly ∂LAML/∂ρ_smooth with the ridge held fixed.
        if grad_full.len() < num_smoothing || !cost.is_finite() {
            return Err("survival LAML returned an inconsistent gradient/cost".to_string());
        }
        let grad = grad_full.slice(s![..num_smoothing]).to_owned();
        if grad.iter().any(|g| !g.is_finite()) {
            return Err("survival LAML gradient is non-finite".to_string());
        }
        Ok((cost, grad))
    };

    let lower = seed_rho.mapv(|v| v - 12.0);
    let upper = seed_rho.mapv(|v| v + 12.0);
    let problem = OuterProblem::new(num_smoothing)
        .with_solver_class(SolverClass::Primary)
        .with_gradient(Derivative::Analytic)
        .with_hessian(crate::solver::outer_strategy::DeclaredHessianForm::Unavailable)
        .with_tolerance(1e-4)
        .with_max_iter(120)
        .with_bounds(lower, upper)
        .with_initial_rho(seed_rho.clone())
        .with_seed_config(crate::seeding::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let context =
        format!("survival transformation smoothing-parameter selection (dim={num_smoothing})");
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| {
            eval_at(rho)
                .map(|(c, _)| c)
                .map_err(crate::estimate::EstimationError::InvalidInput)
        },
        |_: &mut (), rho: &Array1<f64>| {
            let (cost, gradient) =
                eval_at(rho).map_err(crate::estimate::EstimationError::InvalidInput)?;
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<
            fn(
                &mut (),
                &Array1<f64>,
            )
                -> Result<crate::solver::outer_strategy::EfsEval, crate::estimate::EstimationError>,
        >,
    );
    let result = problem
        .run(&mut obj, &context)
        .map_err(|err| format!("{context} failed: {err}"))?;
    // The selector improves the fit; if the outer loop does not certify
    // convergence (rare flat-LAML plateau), fall back to the best ρ it reached
    // rather than failing the whole fit — the seed is already a valid model.
    let selected_rho = result.rho;
    let mut lambdas = seed_lambdas;
    for k in 0..num_smoothing.min(selected_rho.len()) {
        let lam = selected_rho[k].exp();
        if lam.is_finite() && lam > 0.0 {
            lambdas[k] = lam;
        }
    }
    Ok(Some(lambdas))
}

fn survival_unified_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    summary: &crate::pirls::WorkingModelPirlsResult,
    state: &crate::pirls::WorkingState,
    penalty_blocks: &[PenaltyBlock],
) -> Result<UnifiedFitResult, String> {
    let log_lambdas = lambdas.mapv(|v| v.max(LOG_LAMBDA_UNDERFLOW_FLOOR).ln());
    let reml_score = survival_working_reml_score(state);
    crate::estimate::validate_all_finite("survival fit beta", beta.iter().copied())?;
    crate::estimate::validate_all_finite("survival fit lambdas", lambdas.iter().copied())?;
    crate::estimate::ensure_finite_scalar("survival fit log_likelihood", state.log_likelihood)?;
    crate::estimate::ensure_finite_scalar("survival fit deviance", state.deviance)?;
    crate::estimate::ensure_finite_scalar("survival fit penalty", state.penalty_term)?;
    crate::estimate::ensure_finite_scalar("survival fit reml_score", reml_score)?;
    crate::estimate::ensure_finite_scalar("survival fit gradient_norm", summary.lastgradient_norm)?;
    crate::estimate::ensure_finite_scalar("survival fit max_abs_eta", summary.max_abs_eta)?;

    // Penalized effective degrees of freedom from the converged penalized
    // Hessian and penalty roots (issue #565). `lambdas` is built one entry per
    // penalty block, so `edf_by_block` aligns 1:1 with `lambdas` as the
    // `try_from_parts` invariant requires.
    let (edf_total, edf_by_block, penalized_hessian) =
        survival_transformation_edf(state, penalty_blocks)?;
    assert_eq!(edf_by_block.len(), lambdas.len());

    let inference = crate::estimate::FitInference {
        edf_by_block: edf_by_block.clone(),
        edf_total,
        smoothing_correction: None,
        penalized_hessian: penalized_hessian.into(),
        working_weights: Array1::zeros(0),
        working_response: Array1::zeros(0),
        reparam_qs: None,
        dispersion: crate::estimate::Dispersion::Known(1.0),
        beta_covariance: None,
        beta_standard_errors: None,
        beta_covariance_corrected: None,
        beta_standard_errors_corrected: None,
        beta_covariance_frequentist: None,
        coefficient_influence: None,
        weighted_gram: None,
        bias_correction_beta: None,
    };

    UnifiedFitResult::try_from_parts(crate::estimate::UnifiedFitResultParts {
        blocks: vec![crate::estimate::FittedBlock {
            beta: beta.clone(),
            role: crate::estimate::BlockRole::Mean,
            edf: edf_total,
            lambdas: lambdas.clone(),
        }],
        log_lambdas,
        lambdas,
        likelihood_family: Some(LikelihoodSpec::royston_parmar()),
        likelihood_scale: crate::types::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: crate::types::LogLikelihoodNormalization::UserProvided,
        log_likelihood: state.log_likelihood,
        deviance: state.deviance,
        reml_score,
        stable_penalty_term: state.penalty_term,
        penalized_objective: reml_score,
        outer_iterations: summary.iterations,
        outer_converged: true,
        outer_gradient_norm: Some(summary.lastgradient_norm),
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: Some(inference),
        fitted_link: FittedLinkState::Standard(None),
        geometry: None,
        block_states: Vec::new(),
        pirls_status: summary.status,
        max_abs_eta: summary.max_abs_eta,
        constraint_kkt: None,
        artifacts: crate::estimate::FitArtifacts {
            pirls: None,
            ..Default::default()
        },
        inner_cycles: 0,
    })
    .map_err(|err| err.to_string())
}

/// Replicate the single pooled-baseline coefficient seed (length `p`) across
/// every competing-risks cause.
///
/// `build_working_model` fits one shared single-hazard Royston-Parmar baseline
/// and returns a length-`p` coefficient seed (the Weibull scale/shape seed for
/// the parametric path). The cause-specific assembly in
/// `fit_cause_specific_survival_transformation_custom` stacks one coefficient
/// block per cause and slices `cause * p..(cause + 1) * p` out of its
/// `beta0_flat`, so it requires exactly `p * cause_count` initial coefficients.
/// Passing the un-replicated length-`p` seed straight through (the original
/// #378 fix did) aborts every `cause_count > 1` fit with a length-mismatch
/// `SchemaMismatch`. Seeding every cause from the same pooled baseline is the
/// correct start: each cause-specific block treats the competing causes as
/// censored, so they share the pooled baseline hazard until PIRLS specializes.
/// For `cause_count == 1` this is the identity.
fn replicate_pooled_baseline_seed_per_cause(
    pooled_seed: ArrayView1<'_, f64>,
    cause_count: usize,
) -> Array1<f64> {
    let p = pooled_seed.len();
    let mut beta0_flat = Array1::<f64>::zeros(p * cause_count);
    for cause in 0..cause_count {
        beta0_flat
            .slice_mut(s![cause * p..(cause + 1) * p])
            .assign(&pooled_seed);
    }
    beta0_flat
}

fn fit_cause_specific_survival_transformation_custom(
    spec: &SurvivalTransformationTermSpec,
    resolvedspec: TermCollectionSpec,
    baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    prepared: PreparedSurvivalTimeStack,
    dense_cov_design: &Array2<f64>,
    penalty_blocks: Vec<PenaltyBlock>,
    beta0_flat: Array1<f64>,
    derivative_floor: f64,
    penalty_block_gamma_priors: &[(String, f64, f64)],
) -> Result<SurvivalTransformationFitResult, String> {
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .into_workflow_result()?;
    if cause_count == 0 {
        return Err(WorkflowError::MissingDependency {
            reason: "cause-specific custom survival fit requires at least one cause".to_string(),
        }
        .into());
    }
    let n = spec.event_target.len();
    let p_time_total = prepared.time_design_exit.ncols();
    let p_cov = dense_cov_design.ncols();
    let p = p_time_total + p_cov;
    if beta0_flat.len() != p * cause_count {
        return Err(WorkflowError::SchemaMismatch {
            reason: format!(
                "cause-specific survival initial beta length mismatch: got {}, expected {}",
                beta0_flat.len(),
                p * cause_count
            ),
        }
        .into());
    }

    let dense_time_entry = prepared.time_design_entry.to_dense();
    let dense_time_exit = prepared.time_design_exit.to_dense();
    let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    if p_time_total > 0 {
        x_entry
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_entry);
        x_exit
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_exit);
        x_derivative
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_derivative);
    }
    if p_cov > 0 {
        x_entry
            .slice_mut(s![.., p_time_total..])
            .assign(dense_cov_design);
        x_exit
            .slice_mut(s![.., p_time_total..])
            .assign(dense_cov_design);
    }

    let mut family_blocks = Vec::with_capacity(cause_count);
    let mut block_specs = Vec::with_capacity(cause_count);
    for cause in 0..cause_count {
        let cause_code = (cause + 1) as u8;
        let event_target = spec
            .event_target
            .mapv(|observed| u8::from(observed == cause_code));
        family_blocks.push(crate::survival::CauseSpecificRoystonParmarBlock {
            age_entry: spec.age_entry.clone(),
            age_exit: spec.age_exit.clone(),
            event_target,
            sampleweight: spec.weights.clone(),
            x_entry: x_entry.clone(),
            x_exit: x_exit.clone(),
            x_derivative: x_derivative.clone(),
            offset_eta_entry: prepared.eta_offset_entry.clone() + &spec.covariate_offset,
            offset_eta_exit: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            offset_derivative_exit: prepared.derivative_offset_exit.clone(),
            derivative_floor,
        });

        let mut penalties = Vec::with_capacity(penalty_blocks.len());
        let mut nullspace_dims = Vec::with_capacity(penalty_blocks.len());
        let mut initial_log_lambdas = Array1::<f64>::zeros(penalty_blocks.len());
        for (penalty_idx, block) in penalty_blocks.iter().enumerate() {
            if block.range.end > p || block.range.start > block.range.end {
                return Err(WorkflowError::SchemaMismatch {
                    reason: "cause-specific survival penalty range is out of bounds".to_string(),
                }
                .into());
            }
            let block_dim = block.range.end - block.range.start;
            if block.matrix.nrows() != block_dim || block.matrix.ncols() != block_dim {
                return Err(WorkflowError::SchemaMismatch {
                    reason: format!(
                        "cause-specific survival penalty {penalty_idx} has shape {}x{} but range has width {block_dim}",
                        block.matrix.nrows(),
                        block.matrix.ncols()
                    ),
                }
                .into());
            }
            penalties.push(
                PenaltyMatrix::Blockwise {
                    local: block.matrix.clone(),
                    col_range: block.range.clone(),
                    total_dim: p,
                }
                .with_precision_label(format!("cause_specific_survival_penalty_{penalty_idx}")),
            );
            nullspace_dims.push(block.nullspace_dim);
            initial_log_lambdas[penalty_idx] = block.lambda.max(LOG_LAMBDA_UNDERFLOW_FLOOR).ln();
        }
        let beta_start = beta0_flat.slice(s![cause * p..(cause + 1) * p]).to_owned();
        // Cause-specific blocks share the same time-basis design `x_exit`
        // (the same I-spline evaluated at the same observed event times), so
        // the joint design carries K block-pairs of (near-)identical
        // columns. The model is identifiable because the cause-specific
        // likelihood routes each cause to disjoint risk sets and
        // event-indicator masks
        // (`CauseSpecificRoystonParmarFamily::likelihood_blocks_uncoupled =
        // true`), but the identifiability audit operates on the unweighted
        // joint design. With every cause carrying the same `gauge_priority`
        // and no Jacobian callback to declare channel ownership, the audit's
        // `hard_alias_pair` gate fires on the strongest cross-block pair and
        // refuses the full-rank fit even when `joint_rank == p_total`.
        //
        // Mirror the multinomial-class block convention: assign descending
        // priorities (cause 0 highest, cause K-1 lowest) so the audit's
        // `pa != pb` filter on cross-block alias pairs always succeeds, and
        // attach an `AdditiveBlockJacobian` with `own_output = cause` so the
        // channel-aware audit treats each cause's contribution as occupying
        // its own output-channel rows. The Jacobian callback also takes the
        // canonical-gauge orthogonalisation pass out of play (the
        // family-owned-geometry guard defers when any block exposes a
        // callback), so the shared near-aliased column is not residualised
        // into a degenerate near-zero column behind the family's back; the
        // penalty + line search at solve time still resolves any residual
        // near-collinearity.
        let cause_priority =
            100u8.saturating_add(u8::try_from(cause_count - cause).unwrap_or(u8::MAX));
        let cause_jacobian = std::sync::Arc::new(AdditiveBlockJacobian {
            design: x_exit.clone(),
            own_output: cause,
            n_family_outputs: cause_count,
        });
        block_specs.push(ParameterBlockSpec {
            name: format!("time_cause_{}", cause + 1),
            design: crate::matrix::DesignMatrix::from(x_exit.clone()),
            offset: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            penalties,
            nullspace_dims,
            initial_log_lambdas,
            initial_beta: Some(beta_start),
            gauge_priority: cause_priority,
            jacobian_callback: Some(cause_jacobian),
            stacked_design: None,
            stacked_offset: None,
        });
    }

    let family = crate::survival::CauseSpecificRoystonParmarFamily::new(family_blocks)?;
    let fit_options = BlockwiseFitOptions {
        compute_covariance: false,
        ..Default::default()
    };
    let rho_prior =
        cause_specific_survival_rho_prior(penalty_blocks.len(), penalty_block_gamma_priors)?;
    let mut fit = fit_custom_family_with_rho_prior(&family, &block_specs, &fit_options, rho_prior)
        .map_err(|err| format!("cause-specific survival custom-family fit failed: {err}"))?;
    fit.likelihood_family = Some(LikelihoodSpec::royston_parmar());
    let time_basis = crate::families::survival_construction::SavedSurvivalTimeBasis::from_build(
        &spec.time_build,
        spec.time_anchor,
    );
    // Recover the FITTED Weibull baseline from the converged linear-time
    // coefficients, mirroring the single-cause path (issues #689/#690). The
    // seed `baseline_cfg` carried only the pre-fit pooled scale/shape
    // (`shape = 1`, `scale = time-seed`), so any caller reading
    // `fit.baseline_cfg.scale/shape` to reconstruct `H = (t/scale)^shape`
    // would build the CIF from the uninitialized baseline and collapse it to
    // null. For Weibull-without-timewiggle the time basis is the 2-column
    // `[1, log t]` linear basis whose per-cause coefficients carry the full
    // log-cumulative-hazard. Because the basis is anchor-centered the constant
    // column (and thus `beta[0]`) is unidentified, so the fitted scale is the
    // identified anchor (`scale = anchor`, `shape = beta[1]`), not `beta[0]`
    // (issue #899). The shared `SurvivalBaselineConfig` holds a
    // single (scale, shape), so we report the first cause's fitted baseline as
    // the representative shared value — the same pooled-baseline convention the
    // seed used, but post-fit rather than uninitialized.
    let fitted_baseline_cfg = if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull
        && spec.timewiggle.is_none()
    {
        let first_block = fit.blocks.first().ok_or_else(|| {
            "cause-specific survival fit produced no coefficient blocks".to_string()
        })?;
        let time_beta = first_block
            .beta
            .slice(s![..spec.time_build.x_exit_time.ncols()])
            .to_owned();
        fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(|| {
            "failed to recover fitted Weibull scale/shape from the cause-specific linear time coefficients"
                .to_string()
        })?
    } else {
        baseline_cfg
    };
    Ok(SurvivalTransformationFitResult {
        fit,
        resolvedspec,
        baseline_cfg: fitted_baseline_cfg,
        likelihood_mode: spec.likelihood_mode,
        time_basis,
        time_base_ncols: spec.time_build.x_exit_time.ncols(),
        baseline_timewiggle: prepared.timewiggle_block,
    })
}

fn cause_specific_survival_rho_prior(
    penalty_count: usize,
    penalty_block_gamma_priors: &[(String, f64, f64)],
) -> Result<crate::types::RhoPrior, String> {
    if penalty_block_gamma_priors.is_empty() {
        return Ok(crate::types::RhoPrior::Flat);
    }
    let mut keyed = BTreeMap::<String, (f64, f64)>::new();
    for (label, shape, rate) in penalty_block_gamma_priors {
        if keyed.insert(label.clone(), (*shape, *rate)).is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "duplicate Gamma precision hyperprior for penalty block label '{label}'"
                ),
            }
            .into());
        }
        if !shape.is_finite() || *shape <= 0.0 {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "Gamma precision hyperprior for penalty block '{label}' requires shape > 0, got {shape}"
                ),
            }
            .into());
        }
        if !rate.is_finite() || *rate < 0.0 {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "Gamma precision hyperprior for penalty block '{label}' requires rate >= 0, got {rate}"
                ),
            }
            .into());
        }
    }
    let mut consumed = Vec::<String>::new();
    let mut priors = Vec::<crate::types::RhoPrior>::with_capacity(penalty_count);
    for penalty_idx in 0..penalty_count {
        let label = format!("cause_specific_survival_penalty_{penalty_idx}");
        if let Some((shape, rate)) = keyed.get(&label) {
            consumed.push(label);
            priors.push(crate::types::RhoPrior::GammaPrecision {
                shape: *shape,
                rate: *rate,
            });
        } else {
            priors.push(crate::types::RhoPrior::Flat);
        }
    }
    let unknown = keyed
        .keys()
        .filter(|label| !consumed.iter().any(|known| known == *label))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        let available = (0..penalty_count)
            .map(|idx| format!("cause_specific_survival_penalty_{idx}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "unknown Gamma precision hyperprior penalty block label(s): {}; available labels: {available}",
                unknown.join(", ")
            ),
        }
        .into());
    }
    Ok(crate::types::RhoPrior::Independent(priors))
}

fn hash_workflow_array_view(hasher: &mut crate::cache::Fingerprinter, array: ArrayView1<'_, f64>) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_f64(value);
    }
}

fn hash_workflow_u8_array(hasher: &mut crate::cache::Fingerprinter, array: ArrayView1<'_, u8>) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_usize(usize::from(value));
    }
}

fn hash_workflow_array2(hasher: &mut crate::cache::Fingerprinter, array: ArrayView2<'_, f64>) {
    hasher.write_usize(array.nrows());
    hasher.write_usize(array.ncols());
    for row in array.rows() {
        for &value in row {
            hasher.write_f64(value);
        }
    }
}

fn hash_workflow_design_matrix(
    hasher: &mut crate::cache::Fingerprinter,
    matrix: &crate::matrix::DesignMatrix,
) {
    let dense = matrix.to_dense();
    hash_workflow_array2(hasher, dense.view());
}

fn survival_transformation_log_lambdas(
    penalty_blocks: &[crate::survival::PenaltyBlock],
) -> Vec<f64> {
    penalty_blocks
        .iter()
        .map(|block| block.lambda.max(LOG_LAMBDA_UNDERFLOW_FLOOR).ln())
        .collect()
}

fn persistent_survival_transformation_key(
    spec: &SurvivalTransformationTermSpec,
    baseline_cfg: &crate::families::survival_construction::SurvivalBaselineConfig,
    dense_cov_design: ArrayView2<'_, f64>,
    prepared: &PreparedSurvivalTimeStack,
    penalty_blocks: &[crate::survival::PenaltyBlock],
    opts: &crate::pirls::WorkingModelPirlsOptions,
    n_cols: usize,
) -> String {
    let mut hasher = crate::cache::Fingerprinter::new();
    hasher.write_str("gamfit-persistent-survival-transformation-working-pirls");
    // Use the cache schema tag (NOT CARGO_PKG_VERSION) so routine
    // library version bumps don't invalidate users' on-disk warm-start
    // caches.
    hasher.write_str(&crate::solver::persistent_warm_start::cache_schema_tag());
    hasher.write_str(&format!("{:?}", spec.likelihood_mode));
    hasher.write_f64(spec.time_anchor);
    hasher.write_f64(spec.ridge_lambda);
    hasher.write_str(&format!("{:?}", baseline_cfg.target));
    for value in [
        baseline_cfg.scale,
        baseline_cfg.shape,
        baseline_cfg.rate,
        baseline_cfg.makeham,
    ] {
        hasher.write_bool(value.is_some());
        if let Some(value) = value {
            hasher.write_f64(value);
        }
    }
    hasher.write_str(&spec.time_build.basisname);
    hasher.write_usize(spec.time_build.x_entry_time.nrows());
    hasher.write_usize(spec.time_build.x_entry_time.ncols());
    hasher.write_usize(spec.time_build.x_exit_time.nrows());
    hasher.write_usize(spec.time_build.x_exit_time.ncols());
    hasher.write_usize(spec.time_build.x_derivative_time.nrows());
    hasher.write_usize(spec.time_build.x_derivative_time.ncols());
    hasher.write_bool(spec.time_build.degree.is_some());
    if let Some(degree) = spec.time_build.degree {
        hasher.write_usize(degree);
    }
    match spec.time_build.knots.as_ref() {
        Some(knots) => {
            hasher.write_bool(true);
            hasher.write_usize(knots.len());
            for &knot in knots {
                hasher.write_f64(knot);
            }
        }
        None => hasher.write_bool(false),
    }
    match spec.time_build.keep_cols.as_ref() {
        Some(cols) => {
            hasher.write_bool(true);
            hasher.write_usize(cols.len());
            for &col in cols {
                hasher.write_usize(col);
            }
        }
        None => hasher.write_bool(false),
    }
    hasher.write_bool(spec.time_build.smooth_lambda.is_some());
    if let Some(lambda) = spec.time_build.smooth_lambda {
        hasher.write_f64(lambda);
    }
    hasher.write_usize(n_cols);
    hash_workflow_array_view(&mut hasher, spec.age_entry.view());
    hash_workflow_array_view(&mut hasher, spec.age_exit.view());
    hash_workflow_u8_array(&mut hasher, spec.event_target.view());
    hash_workflow_array_view(&mut hasher, spec.weights.view());
    hash_workflow_array_view(&mut hasher, spec.covariate_offset.view());
    hash_workflow_array2(&mut hasher, dense_cov_design);
    hash_workflow_array_view(&mut hasher, prepared.eta_offset_entry.view());
    hash_workflow_array_view(&mut hasher, prepared.eta_offset_exit.view());
    hash_workflow_array_view(&mut hasher, prepared.derivative_offset_exit.view());
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_entry);
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_exit);
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_derivative_exit);
    hasher.write_usize(penalty_blocks.len());
    for block in penalty_blocks {
        hasher.write_f64(block.lambda);
        hasher.write_usize(block.range.start);
        hasher.write_usize(block.range.end);
        hasher.write_usize(block.nullspace_dim);
        hash_workflow_array2(&mut hasher, block.matrix.view());
    }
    hasher.write_usize(opts.max_iterations);
    hasher.write_f64(opts.convergence_tolerance);
    hasher.write_usize(opts.max_step_halving);
    hasher.write_f64(opts.min_step_size);
    hasher.write_bool(opts.firth_bias_reduction);
    hasher.write_bool(opts.coefficient_lower_bounds.is_some());
    if let Some(bounds) = opts.coefficient_lower_bounds.as_ref() {
        hash_workflow_array_view(&mut hasher, bounds.view());
    }
    hasher.write_bool(opts.linear_constraints.is_some());
    format!("surv-transform-{}", hasher.finish_hex())
}

fn load_survival_transformation_persistent_warm_start(
    key: &str,
    spec: &SurvivalTransformationTermSpec,
    n_cols: usize,
    rho: &[f64],
) -> Option<(Array1<f64>, Option<f64>)> {
    let record = crate::solver::persistent_warm_start::load_record(key)?;
    if !record.is_compatible(key, spec.age_entry.len(), n_cols)
        || record.rho.len() != rho.len()
        || !record
            .rho
            .iter()
            .zip(rho.iter())
            .all(|(cached, expected)| (*cached - *expected).abs() <= 1e-10)
    {
        return None;
    }
    log::info!("[warm-start-cache] restored survival transformation warm start key={key}");
    let lm_lambda = record
        .last_pirls_lm_lambda
        .filter(|value| value.is_finite() && *value > 0.0);
    Some((Array1::from_vec(record.beta), lm_lambda))
}

fn store_survival_transformation_persistent_warm_start(
    key: &str,
    spec: &SurvivalTransformationTermSpec,
    n_cols: usize,
    rho: Vec<f64>,
    beta: &Array1<f64>,
    summary: &crate::pirls::WorkingModelPirlsResult,
) {
    if beta.len() != n_cols
        || beta.iter().any(|value| !value.is_finite())
        || rho.iter().any(|value| !value.is_finite())
    {
        return;
    }
    let mut record = crate::solver::persistent_warm_start::PersistentWarmStartRecord::new(
        key.to_string(),
        spec.age_entry.len(),
        n_cols,
    );
    record.rho = rho;
    record.beta = beta.to_vec();
    record.last_inner_iters = summary.iterations;
    record.last_inner_converged = matches!(
        summary.status,
        crate::pirls::PirlsStatus::Converged | crate::pirls::PirlsStatus::StalledAtValidMinimum
    );
    record.last_pirls_lm_lambda = (summary.final_lm_lambda.is_finite()
        && summary.final_lm_lambda > 0.0)
        .then_some(summary.final_lm_lambda);
    record.last_pirls_accept_rho = summary
        .final_accept_rho
        .filter(|value| value.is_finite() && *value >= 0.0);
    if let Err(err) = crate::solver::persistent_warm_start::store_record(&record) {
        log::warn!(
            "[warm-start-cache] failed to persist survival transformation warm start: {err}"
        );
    }
}

fn fit_survival_transformation_model(
    request: SurvivalTransformationFitRequest<'_>,
) -> Result<SurvivalTransformationFitResult, String> {
    use crate::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};

    let SurvivalTransformationFitRequest {
        data,
        spec,
        cache_session: _cache_session,
    } = request;
    let mut baseline_cfg = spec.baseline_cfg.clone();
    let covariate_design =
        build_term_collection_design(data, &spec.covariate_spec).map_err(|err| err.to_string())?;
    let resolvedspec =
        crate::smooth::freeze_term_collection_from_design(&spec.covariate_spec, &covariate_design)
            .map_err(|err| err.to_string())?;
    let dense_cov_design = covariate_design.design.to_dense();
    let p_cov = dense_cov_design.ncols();
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .into_workflow_result()?;
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(spec.likelihood_mode);

    let build_working_model =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let prepared = prepare_survival_time_stack(
                &spec.age_entry,
                &spec.age_exit,
                candidate,
                spec.likelihood_mode,
                None,
                spec.time_anchor,
                exact_derivative_guard,
                &spec.time_build,
                spec.timewiggle.as_ref(),
                None,
            )?;
            let mut eta_offset_entry = prepared.eta_offset_entry.clone();
            let mut eta_offset_exit = prepared.eta_offset_exit.clone();
            eta_offset_entry += &spec.covariate_offset;
            eta_offset_exit += &spec.covariate_offset;
            let p_time_total = prepared.time_design_exit.ncols();
            let p = p_time_total + p_cov;
            let mut penalty_blocks = Vec::<PenaltyBlock>::new();
            for (idx, penalty) in prepared.time_penalties.iter().enumerate() {
                if penalty.nrows() == p_time_total && penalty.ncols() == p_time_total {
                    penalty_blocks.push(PenaltyBlock {
                        matrix: penalty.clone(),
                        lambda: spec.time_build.smooth_lambda.unwrap_or(1e-2),
                        range: 0..p_time_total,
                        nullspace_dim: prepared.time_nullspace_dims.get(idx).copied().unwrap_or(0),
                    });
                }
            }
            // Covariate-smooth penalties (e.g. `s(x)`, `s(group, bs="re")`
            // frailty) live in the covariate term-collection design; the survival
            // transformation fit stacks the covariate columns at
            // `p_time_total..p`, so each covariate penalty's local block maps to
            // the joint range `p_time_total + col_range`. Penalizing them here
            // (rather than leaving them to the tiny stabilization ridge) is what
            // lets the frailty / covariate smooths shrink — and they are added
            // BEFORE the ridge so they too become REML-selected smoothing blocks
            // (issues #563/#565). Only zero-prior-mean blocks are admissible as a
            // plain quadratic `λ βᵀSβ`; a non-zero centering would need an offset
            // the survival PenaltyBlock does not model, so such blocks are left to
            // the ridge rather than mis-applied.
            for cov_penalty in &covariate_design.penalties {
                let cr = &cov_penalty.col_range;
                let block_dim = cr.end - cr.start;
                let matches_dims = cov_penalty.local.nrows() == block_dim
                    && cov_penalty.local.ncols() == block_dim;
                let zero_prior = matches!(
                    cov_penalty.prior_mean,
                    crate::estimate::CoefficientPriorMean::Zero
                );
                if block_dim > 0 && matches_dims && zero_prior && cr.end <= p_cov {
                    penalty_blocks.push(PenaltyBlock {
                        matrix: cov_penalty.local.clone(),
                        lambda: 1e-2,
                        range: (p_time_total + cr.start)..(p_time_total + cr.end),
                        nullspace_dim: 0,
                    });
                }
            }
            // The smoothing blocks are exactly those pushed above (time +
            // covariate penalties); any ridge appended below is a FIXED
            // stabilization, not a REML-selected smoothing parameter, so the
            // count of smoothing blocks is recorded before the ridge is added
            // (issue #563).
            let num_smoothing_blocks = penalty_blocks.len();
            let ridge_range_start = if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull
                && spec.time_build.basisname == "linear"
                && spec.timewiggle.is_none()
            {
                1
            } else {
                0
            };
            if spec.ridge_lambda > 0.0 && p > ridge_range_start {
                let dim = p - ridge_range_start;
                let mut ridge = Array2::<f64>::zeros((dim, dim));
                for d in 0..dim {
                    ridge[[d, d]] = 1.0;
                }
                penalty_blocks.push(PenaltyBlock {
                    matrix: ridge,
                    lambda: spec.ridge_lambda,
                    range: ridge_range_start..p,
                    nullspace_dim: 0,
                });
            }
            let dense_time_entry = prepared.time_design_entry.to_dense();
            let dense_time_exit = prepared.time_design_exit.to_dense();
            let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
            let event_competing = Array1::<u8>::zeros(spec.event_target.len());
            // `spec.event_target` carries *cause labels* (0 = censored, k = cause k).
            // The shared baseline working model is a single-hazard Royston-Parmar
            // model whose binary `event_target` contract is {0, 1}. For the pooled
            // baseline that seeds scale/shape across all causes, every observed event
            // (any cause) informs the shared baseline hazard, so collapse cause labels
            // to a {0, 1} any-event indicator. The per-cause specialization (event for
            // cause k vs. competing-cause-as-censored) happens later when the
            // cause-specific blocks are built.
            let baseline_event_indicator = spec.event_target.mapv(|label| u8::from(label > 0));
            let mut model =
                crate::families::royston_parmar::working_model_from_time_covariateshared(
                    PenaltyBlocks::new(penalty_blocks.clone()),
                    MonotonicityPenalty { tolerance: 0.0 },
                    SurvivalSpec::Net,
                    crate::families::royston_parmar::RoystonParmarSharedTimeCovariateInputs {
                        age_entry: spec.age_entry.view(),
                        age_exit: spec.age_exit.view(),
                        event_target: baseline_event_indicator.view(),
                        event_competing: event_competing.view(),
                        weights: spec.weights.view(),
                        time_entry: dense_time_entry.view(),
                        time_exit: dense_time_exit.view(),
                        time_derivative: dense_time_derivative.view(),
                        covariates: dense_cov_design.view(),
                        monotonicity_constraint_rows: None,
                        monotonicity_constraint_offsets: None,
                        eta_offset_entry: Some(eta_offset_entry.view()),
                        eta_offset_exit: Some(eta_offset_exit.view()),
                        derivative_offset_exit: Some(prepared.derivative_offset_exit.view()),
                    },
                )
                .map_err(|err| format!("failed to construct survival model: {err}"))?;
            if spec.likelihood_mode != SurvivalLikelihoodMode::Weibull {
                model
                    .set_structural_monotonicity(true, p_time_total)
                    .map_err(|err| format!("failed to enable structural monotonicity: {err}"))?;
            }
            let mut beta0 = Array1::<f64>::zeros(p);
            if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none()
            {
                let (scale, shape) = spec
                    .weibull_seed
                    .ok_or_else(|| "weibull survival fit missing scale/shape seed".to_string())?;
                if p_time_total < 2 {
                    return Err(format!(
                        "weibull built-in time basis has {p_time_total} columns but needs 2 to seed scale/shape"
                    ));
                }
                beta0[0] = -shape * scale.ln();
                beta0[1] = shape;
            }
            let structural_lower_bounds =
                if spec.likelihood_mode != SurvivalLikelihoodMode::Weibull && p_time_total > 0 {
                    let mut lb = Array1::from_elem(p, f64::NEG_INFINITY);
                    for j in 0..p_time_total {
                        lb[j] = 0.0;
                        beta0[j] = 1e-4;
                    }
                    Some(lb)
                } else {
                    None
                };
            Ok::<_, String>((
                prepared,
                penalty_blocks,
                beta0,
                structural_lower_bounds,
                model,
                num_smoothing_blocks,
            ))
        };

    if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        baseline_cfg = optimize_survival_baseline_config(
            &baseline_cfg,
            "workflow survival transformation baseline",
            |candidate| {
                let (_, _, beta0, structural_lower_bounds, mut model, _) =
                    build_working_model(candidate)?;
                let opts = crate::pirls::WorkingModelPirlsOptions {
                    max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
                    convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
                    adaptive_kkt_tolerance: None,
                    max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
                    min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
                    firth_bias_reduction: false,
                    coefficient_lower_bounds: structural_lower_bounds,
                    linear_constraints: None,
                    initial_lm_lambda: None,
                    geodesic_acceleration: false,
                    arrow_schur: None,
                };
                let summary = crate::pirls::runworking_model_pirls(
                    &mut model,
                    crate::types::Coefficients::new(beta0),
                    &opts,
                    |_| {},
                )
                .map_err(|err| format!("survival PIRLS failed: {err}"))?;
                let beta = summary.beta.as_ref().to_owned();
                let state = model.update_state(&beta).map_err(|err| {
                    format!("failed to evaluate survival baseline candidate: {err}")
                })?;
                Ok(survival_working_reml_score(&state))
            },
        )?;
    }

    let (
        prepared,
        mut penalty_blocks,
        beta0,
        structural_lower_bounds,
        mut model,
        num_smoothing_blocks,
    ) = build_working_model(&baseline_cfg)?;
    if cause_count > 1 || !spec.penalty_block_gamma_priors.is_empty() {
        let beta0_flat = replicate_pooled_baseline_seed_per_cause(beta0.view(), cause_count);
        return fit_cause_specific_survival_transformation_custom(
            &spec,
            resolvedspec,
            baseline_cfg,
            prepared,
            &dense_cov_design,
            penalty_blocks,
            beta0_flat,
            exact_derivative_guard,
            &spec.penalty_block_gamma_priors,
        );
    }
    // REML/LAML-select the time-smoothing λ (issue #563). With λ pinned at its
    // seed the monotone I-spline baseline oversmooths toward an affine
    // log-cumulative-hazard; selecting λ from the survival LAML lets it recover
    // real curvature. The inner solve keeps the structural γ ≥ 0 box at every
    // candidate, so the constrained optimum stays valid; the fixed stabilization
    // ridge is held at its seed. The selected λ is written back into both the
    // working model and `penalty_blocks` so the final fit, edf, and warm-start
    // cache all use the data-adaptive value.
    if let Some(selected_lambdas) = optimize_survival_transformation_smoothing(
        &model,
        &penalty_blocks,
        num_smoothing_blocks,
        &beta0,
        structural_lower_bounds.as_ref(),
    )? {
        model
            .set_penalty_lambdas(&selected_lambdas)
            .map_err(|e| e.to_string())?;
        for (block, &lam) in penalty_blocks.iter_mut().zip(selected_lambdas.iter()) {
            block.lambda = lam;
        }
    }
    let opts = crate::pirls::WorkingModelPirlsOptions {
        max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
        convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
        adaptive_kkt_tolerance: None,
        max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
        min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
        firth_bias_reduction: false,
        coefficient_lower_bounds: structural_lower_bounds,
        linear_constraints: None,
        initial_lm_lambda: None,
        geodesic_acceleration: false,
        arrow_schur: None,
    };
    let rho_for_cache = survival_transformation_log_lambdas(&penalty_blocks);
    let persistent_warm_start_key = persistent_survival_transformation_key(
        &spec,
        &baseline_cfg,
        dense_cov_design.view(),
        &prepared,
        &penalty_blocks,
        &opts,
        beta0.len(),
    );
    let mut opts = opts;
    let beta_start = match load_survival_transformation_persistent_warm_start(
        &persistent_warm_start_key,
        &spec,
        beta0.len(),
        &rho_for_cache,
    ) {
        Some((beta, lm_lambda)) => {
            opts.initial_lm_lambda = lm_lambda;
            beta
        }
        None => beta0,
    };
    let summary = crate::pirls::runworking_model_pirls(
        &mut model,
        crate::types::Coefficients::new(beta_start),
        &opts,
        |_| {},
    )
    .map_err(|err| format!("survival PIRLS failed: {err}"))?;
    match summary.status {
        crate::pirls::PirlsStatus::Converged | crate::pirls::PirlsStatus::StalledAtValidMinimum => {
        }
        ref other => {
            return Err(WorkflowError::IntegrationFailed {
                reason: format!(
                    "survival PIRLS did not converge: status={other:?}, grad_norm={:.3e}, iterations={}, deviance={:.6e}",
                    summary.lastgradient_norm, summary.iterations, summary.state.deviance
                ),
            }
            .into());
        }
    }
    let beta = summary.beta.as_ref().to_owned();
    store_survival_transformation_persistent_warm_start(
        &persistent_warm_start_key,
        &spec,
        beta.len(),
        rho_for_cache,
        &beta,
        &summary,
    );
    let state = model
        .update_state(&beta)
        .map_err(|err| format!("failed to evaluate survival optimum: {err}"))?;
    let lambdas = Array1::from_iter(penalty_blocks.iter().map(|block| block.lambda));
    let fitted_baseline_cfg =
        if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none() {
            let time_beta = beta
                .slice(s![..spec.time_build.x_exit_time.ncols()])
                .to_owned();
            fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(
                || {
                    "failed to recover fitted Weibull scale/shape from the linear time coefficients"
                        .to_string()
                },
            )?
        } else {
            baseline_cfg
        };
    let fit = survival_unified_fit_result(beta, lambdas, &summary, &state, &penalty_blocks)?;

    let time_base_ncols = spec.time_build.x_exit_time.ncols();
    let time_basis = crate::families::survival_construction::SavedSurvivalTimeBasis::from_build(
        &spec.time_build,
        spec.time_anchor,
    );
    Ok(SurvivalTransformationFitResult {
        fit,
        resolvedspec,
        baseline_cfg: fitted_baseline_cfg,
        likelihood_mode: spec.likelihood_mode,
        time_basis,
        time_base_ncols,
        baseline_timewiggle: prepared.timewiggle_block,
    })
}

fn fit_survival_location_scale_model(
    request: SurvivalLocationScaleFitRequest<'_>,
) -> Result<SurvivalLocationScaleFitResult, String> {
    // Profile one coherent survival subproblem at a fixed inverse-link state:
    // select/apply the link-wiggle basis for that state, then solve the full
    // penalized location-scale fit on the resulting model.
    fn profile_survival_location_scale(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut wiggle_knots = None;
        let mut wiggle_degree = None;
        let inverse_link = spec.inverse_link.clone();

        let fit = if let Some(wiggle) = wiggle {
            require_inverse_link_supports_joint_wiggle(&inverse_link, "survival link wiggle")?;
            let mut pilot_spec = spec.clone();
            pilot_spec.linkwiggle_block = None;
            let pilot = fit_survival_location_scale_terms(data, pilot_spec, kappa_options)?;
            let selected_wiggle_basis = select_survival_link_wiggle_basis_from_pilot(
                &pilot,
                &WiggleBlockConfig {
                    degree: wiggle.degree,
                    num_internal_knots: wiggle.num_internal_knots,
                    penalty_order: 2,
                    double_penalty: wiggle.double_penalty,
                },
                &wiggle.penalty_orders,
            )?;
            wiggle_knots = Some(selected_wiggle_basis.knots.clone());
            wiggle_degree = Some(selected_wiggle_basis.degree);
            fit_survival_location_scale_terms_with_selected_wiggle(
                data,
                spec,
                selected_wiggle_basis,
                kappa_options,
            )?
        } else {
            fit_survival_location_scale_terms(data, spec, kappa_options)?
        };

        Ok(SurvivalLocationScaleProfile {
            fit,
            inverse_link,
            wiggle_knots,
            wiggle_degree,
        })
    }

    fn profile_survival_location_scale_with_inverse_link(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        inverse_link: InverseLink,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut spec_at_link = spec.clone();
        spec_at_link.inverse_link = inverse_link;
        profile_survival_location_scale(data, spec_at_link, wiggle, kappa_options)
    }

    fn optimize_survival_inverse_link_profile(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        fn optimize_link_parameters<F, R>(
            data: ArrayView2<'_, f64>,
            spec: &SurvivalLocationScaleTermSpec,
            kappa_options: &SpatialLengthScaleOptimizationOptions,
            init: Array1<f64>,
            name: &str,
            final_wiggle: Option<LinkWiggleConfig>,
            objective: F,
            recover: R,
        ) -> Result<SurvivalLocationScaleProfile, String>
        where
            F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
            R: Fn(&Array1<f64>) -> Option<InverseLink>,
        {
            use crate::solver::outer_strategy::{OuterProblem, SolverClass};
            let dim = init.len();
            // Inverse-link parameters (SAS epsilon/log_delta, BetaLogistic shape,
            // Mixture rho) have no analytic ∂LAML/∂θ_link; route through the
            // gated gradient-free CompassSearch variant rather than BFGS. Box
            // bounds keep line-search probes inside a physically admissible
            // region (|epsilon|, |log_delta| ≤ 6 gives the SAS link a finite
            // range on both tails).
            let lower = init.mapv(|v| v - 6.0);
            let upper = init.mapv(|v| v + 6.0);
            let problem = OuterProblem::new(dim)
                .with_solver_class(SolverClass::AuxiliaryGradientFree)
                .with_tolerance(1e-4)
                .with_max_iter(240)
                .with_bounds(lower, upper)
                .with_initial_rho(init.clone())
                .with_seed_config(crate::seeding::SeedConfig {
                    max_seeds: 1,
                    seed_budget: 1,
                    num_auxiliary_trailing: dim,
                    ..Default::default()
                });
            let context = format!("survival inverse-link optimization ({name}, dim={dim})");
            let mut obj = problem.build_objective(
                objective,
                |f: &mut F, rho: &ndarray::Array1<f64>| f(rho),
                |_: &mut F, _: &ndarray::Array1<f64>| {
                    Err(EstimationError::InvalidInput(
                        "inverse-link aux optimizer: CompassSearch dispatch only \
                         calls eval_cost; eval(gradient) is unreachable by \
                         construction"
                            .to_string(),
                    ))
                },
                None::<fn(&mut F)>,
                None::<
                    fn(
                        &mut F,
                        &ndarray::Array1<f64>,
                    )
                        -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
                >,
            );
            let result = problem
                .run(&mut obj, &context)
                .map_err(|err| format!("{context} failed: {err}"))?;
            let link = recover_converged_survival_inverse_link(result, &context, recover)?;
            profile_survival_location_scale_with_inverse_link(
                data,
                spec,
                link,
                final_wiggle,
                kappa_options,
            )
            .map_err(|err| format!("{context} final profiling failed: {err}"))
        }

        match spec.inverse_link.clone() {
            InverseLink::Sas(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    init,
                    "SAS",
                    wiggle.clone(),
                    |theta: &Array1<f64>| {
                        let state = state_from_sasspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::Sas(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    },
                    |rho| {
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::Sas)
                    },
                )
            }
            InverseLink::BetaLogistic(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    init,
                    "BetaLogistic",
                    wiggle.clone(),
                    |theta: &Array1<f64>| {
                        let state = state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::BetaLogistic(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    },
                    |rho| {
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::BetaLogistic)
                    },
                )
            }
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                let components_recover = components.clone();
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    state0.rho.clone(),
                    "mixture",
                    wiggle.clone(),
                    move |rho: &Array1<f64>| {
                        let state = state_fromspec(&MixtureLinkSpec {
                            components: components.clone(),
                            initial_rho: rho.clone(),
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::Mixture(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    },
                    move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components_recover.clone(),
                            initial_rho: rho.to_owned(),
                        })
                        .ok()
                        .map(InverseLink::Mixture)
                    },
                )
            }
            _ => profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options),
        }
    }

    let profile = if request.optimize_inverse_link {
        optimize_survival_inverse_link_profile(
            request.data,
            &request.spec,
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    } else {
        profile_survival_location_scale(
            request.data,
            request.spec.clone(),
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    };

    Ok(profile.into_result())
}

fn fit_bernoulli_marginal_slope_model(
    request: BernoulliMarginalSlopeFitRequest<'_>,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    fit_bernoulli_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
        &request.policy,
    )
}

fn fit_survival_marginal_slope_model(
    request: SurvivalMarginalSlopeFitRequest<'_>,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    fit_survival_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

fn fit_latent_survival_model(
    request: LatentSurvivalFitRequest<'_>,
) -> Result<LatentSurvivalTermFitResult, String> {
    fit_latent_survival_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

fn fit_latent_binary_model(
    request: LatentBinaryFitRequest<'_>,
) -> Result<LatentBinaryTermFitResult, String> {
    fit_latent_binary_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

fn fit_transformation_normal_model(
    request: TransformationNormalFitRequest<'_>,
) -> Result<TransformationNormalFitResult, String> {
    fit_transformation_normal(
        &request.response,
        &request.weights,
        &request.offset,
        request.data,
        &request.covariate_spec,
        &request.config,
        &request.options,
        &request.kappa_options,
        request.warm_start.as_ref(),
    )
}

// ---------------------------------------------------------------------------
// Cross-fitted score calibration (Neyman-orthogonal marginal slope, #461)
//
// A marginal-slope Stage-2 model consumes the CTN Stage-1 latent score `z` as a
// generated regressor. `z` depends on θ̂₁, so the β estimating equation leaks
// Stage-1 sampling error (design `marginal_slope_orthogonal_design.md` §1-§4).
// The two DML ingredients are (i) the realized leakage directions
// `J = ∂z/∂θ₁` (an n × p₁ influence Jacobian, computed by the core
// `marginal_slope_orthogonal::score_influence_jacobian`) and (ii) cross-fitting:
// `z` and `J` are evaluated out-of-fold so own-row overfitting of θ̂₁ does not
// bias the absorbed projection. This module owns ingredient (ii) and the
// auto-enable detection: it partitions the rows into K folds, refits the CTN on
// each complement with a basis frozen on the full data, and concatenates the
// per-fold held-out `z` and `J` back into full-n order.
// ---------------------------------------------------------------------------

/// Out-of-fold Stage-1 latent score and its score-influence Jacobian for a
/// CTN → marginal-slope chain. `z_oof` (length n) replaces the in-sample `z`
/// the Stage-2 model consumes; `jac_oof` (n × p₁) is fed to the Stage-2 spec's
/// `score_influence_jacobian` so the joint solve absorbs the realized leakage
/// directions `Z_infl = diag(s_f·β̂₀)·J`.
pub struct CrossFitScoreCalibration {
    pub z_oof: Array1<f64>,
    pub jac_oof: Array2<f64>,
}

/// Internal recipe describing the CTN Stage-1 fit that produced a Stage-2 `z`
/// column. This is in-process plumbing — never a CLI flag, env var, or feature
/// gate. The orchestration layer populates [`FitConfig::ctn_stage1`] when (and
/// only when) the marginal-slope `z` was generated by a transformation-normal
/// Stage-1 fit; its presence is the sole auto-enable signal for cross-fitted
/// orthogonalization (design §5). When absent, Stage-2 falls back to the free
/// 1-D `score_warp` spline (which spans only the x-free leakage column).
#[derive(Clone, Debug)]
pub struct CtnStage1Recipe {
    /// Stage-1 response column name (the `y` the CTN transforms).
    pub response_column: String,
    /// Stage-1 covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"`),
    /// with no `~` and no response symbol. [`crossfit_score_calibration`] parses
    /// it and builds the CTN covariate basis exactly as
    /// `materialize_transformation_normal` does, then FREEZES that basis once on
    /// the full data and reuses the frozen spec for every fold's refit — so the
    /// rebuilt covariate design has an identical column geometry across folds,
    /// keeping `J`'s `p₁ = p_resp · p_cov` columns aligned (design §3).
    ///
    /// The recipe carries the formula RHS (a primitive string) rather than a
    /// resolved [`TermCollectionSpec`] because this struct is populated both via
    /// [`CtnStage1Recipe::new`] (set on [`FitConfig::ctn_stage1`], then
    /// [`fit_from_formula`]) and by the gamfit FFI marshaller
    /// (`gamfit/_calibrated_slope.py`), which can only serialize primitives over
    /// the JSON boundary — a `TermCollectionSpec` is not serializable. Freezing on
    /// the full Stage-2 data is equivalent to
    /// freezing on the Stage-1 data whenever the two stages share a frame (the
    /// calibrated-chain contract), so the column geometry still matches Stage-1.
    pub covariate_formula_rhs: String,
    /// Stage-1 CTN config (response basis degree / knot count / penalties).
    /// Its `response_num_internal_knots` is the FIXED response-basis size; the
    /// cross-fit pins it across folds so `p_resp` (and hence `p₁`) is
    /// fold-invariant (design §3).
    pub config: TransformationNormalConfig,
    /// Optional Stage-1 weight column name.
    pub weight_column: Option<String>,
    /// Optional Stage-1 offset column name.
    pub offset_column: Option<String>,
}

impl CtnStage1Recipe {
    /// Build a Stage-1 CTN recipe from the Stage-1 description. This is the public
    /// way to populate [`FitConfig::ctn_stage1`] — set it on a marginal-slope
    /// config and run [`fit_from_formula`] (the entry IS `fit_from_formula` with
    /// `ctn_stage1` set; there is no separate combined entry function). The
    /// materializer then cross-fits the CTN and installs the leakage-projection
    /// block; supplying the recipe *is* the request for orthogonalization.
    ///
    /// `response` is the Stage-1 CTN response column; `covariates` is the
    /// covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"` — no `~`,
    /// no response symbol). Validates both are non-empty and that `covariates`
    /// is an RHS only.
    pub fn new(
        response: &str,
        covariates: &str,
        config: TransformationNormalConfig,
        weight_column: Option<&str>,
        offset_column: Option<&str>,
    ) -> Result<Self, String> {
        let response_column = response.trim().to_string();
        if response_column.is_empty() {
            return Err("CtnStage1Recipe requires a non-empty Stage-1 response column".to_string());
        }
        let covariate_formula_rhs = covariates.trim().to_string();
        if covariate_formula_rhs.is_empty() {
            return Err(
                "CtnStage1Recipe requires a non-empty Stage-1 covariate formula RHS".to_string(),
            );
        }
        if covariate_formula_rhs.contains('~') {
            return Err(
                "CtnStage1Recipe covariates is a right-hand side only; pass 's(pc1) + s(pc2)', \
                 not 'score ~ s(pc1) + s(pc2)'"
                    .to_string(),
            );
        }
        Ok(Self {
            response_column,
            covariate_formula_rhs,
            config,
            weight_column: weight_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
            offset_column: offset_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
        })
    }
}

/// Number of cross-fit folds for a problem of `n` rows.
///
/// Cross-fitting refits the CTN once per fold, so the cost is `K` full Stage-1
/// fits. The standard DML default is `K = 5` for moderate `n`. At large scale
/// each CTN refit is expensive while the out-of-fold bias from a single split is
/// already negligible (θ̂₁ is precisely estimated on a complement of ≈ `n·(K−1)/K`
/// rows), so `K` is reduced toward 2 as `n` grows — keeping the refit budget
/// bounded without sacrificing OOF de-biasing. Small `n` keeps more folds (larger
/// per-fold training complements ⇒ less OOF estimation noise), dropping below 5
/// only when there are too few rows to populate 5 folds with a usable held-out
/// block.
///
/// The schedule (no flag, no env — derived purely from `n`):
///   - `n < 250`               : `K = min(n, 3)` (tiny data; keep ≥ 2 folds)
///   - `250 ≤ n < 200_000`      : `K = 5` (DML moderate-n default)
///   - `200_000 ≤ n < 2_000_000` : `K = 3` (large-scale: bound refit cost, ≈ ⅔ train)
///   - `n ≥ 2_000_000`          : `K = 2` (mega-large-scale: ½ train still ample)
fn crossfit_fold_count(n: usize) -> usize {
    if n < 250 {
        n.min(3).max(2)
    } else if n < 200_000 {
        5
    } else if n < 2_000_000 {
        3
    } else {
        2
    }
}

/// Partition `n` rows into `k` folds of balanced, contiguous blocks.
///
/// Entry `f` of the returned vector holds the ascending row indices held out in
/// fold `f`; the union over folds is exactly `0..n`. Sizes are `n / k` or
/// `n / k + 1`, so every fold's complement size differs by at most one row,
/// which keeps the response-basis sample cap — and therefore `p₁` — uniform
/// across folds (design §3). Contiguous blocks let the persistent warm-start
/// prefix cache seed each fold's CTN refit from a structurally identical prior
/// fold.
fn crossfit_partition(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = Vec::with_capacity(k);
    let base = n / k;
    let remainder = n % k;
    let mut start = 0usize;
    for f in 0..k {
        // The first `remainder` folds carry one extra row so the union is exactly n.
        let len = base + usize::from(f < remainder);
        let end = start + len;
        folds.push((start..end).collect());
        start = end;
    }
    folds
}

/// Gather `source[idx]` for each `idx` into a fresh contiguous `Array1`.
fn crossfit_select_rows_1d(source: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_iter(indices.iter().map(|&i| source[i]))
}

/// Cross-fitted out-of-fold `z` and score-influence Jacobian `J` for a CTN →
/// marginal-slope chain (design §4-§5).
///
/// Returns `None` when no CTN Stage-1 recipe is present (`recipe` is `None`):
/// the caller then leaves the Stage-2 spec's `score_influence_jacobian` field
/// `None` and Stage-2 uses the supplied raw `z` with the free-warp fallback.
///
/// When a recipe is present, the covariate basis is built from the recipe's
/// formula RHS and FROZEN once on the full data; every fold then refits the CTN
/// against that frozen spec, and the response-basis knot count is pre-resolved at
/// the *smallest* fold complement size, so every fold refits at an identical
/// `(p_resp, p_cov)` and therefore an identical `p₁ = p_resp · p_cov` column
/// layout — the per-fold `J` blocks concatenate into a coherent `n × p₁` matrix
/// (design §3). For each fold `f` the CTN is refit on the complement rows, then
/// `marginal_slope_orthogonal::score_influence_jacobian` evaluates the held-out
/// `z` and `J` on fold `f`'s rows; results scatter back into full-n order.
fn crossfit_score_calibration(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    recipe: Option<&CtnStage1Recipe>,
    policy: &crate::resource::ResourcePolicy,
) -> Result<Option<CrossFitScoreCalibration>, String> {
    let Some(recipe) = recipe else {
        return Ok(None);
    };

    let n = data.values.nrows();
    if n == 0 {
        return Err("cross-fit score calibration requires a non-empty dataset".to_string());
    }

    // Stage-1 response / weights / offset, resolved against the full dataset.
    let y_col = resolve_role_col(col_map, &recipe.response_column, "response")
        .map_err(|e| e.to_string())?;
    let response_full = data.values.column(y_col).to_owned();
    let weights_full = resolve_weight_column(data, col_map, recipe.weight_column.as_deref())
        .map_err(|e| e.to_string())?;
    let offset_full = resolve_offset_column(data, col_map, recipe.offset_column.as_deref())
        .map_err(|e| e.to_string())?;

    // Build the CTN covariate basis from the recipe's formula RHS and FREEZE it
    // ONCE on full data, so every fold refit reuses identical spatial centers /
    // knots ⇒ identical p_cov across folds (design §3). The freeze is what makes
    // the per-fold covariate designs column-aligned.
    let parsed_cov = parse_formula(&format!(
        "{} ~ {}",
        recipe.response_column, recipe.covariate_formula_rhs
    ))
    .map_err(|e| e.to_string())?;
    let mut frozen_notes = Vec::new();
    let covariate_spec_raw = build_termspec_with_geometry_and_overrides(
        &parsed_cov.terms,
        data,
        col_map,
        &mut frozen_notes,
        false,
        policy,
        None,
    )
    .map_err(|e| e.to_string())?;
    let full_cov_design = build_term_collection_design(data.values.view(), &covariate_spec_raw)
        .map_err(|e| e.to_string())?;
    let frozen_cov_spec =
        crate::smooth::freeze_term_collection_from_design(&covariate_spec_raw, &full_cov_design)
            .map_err(|e| e.to_string())?;
    let p_cov = full_cov_design.design.ncols();

    let k = crossfit_fold_count(n);
    let folds = crossfit_partition(n, k);

    // Pre-resolve the response-basis internal-knot count at the *smallest* fold
    // complement size. The CTN sample cap on this count is monotone in the
    // complement size, so pinning the per-fold config to the value resolved at
    // the smallest complement makes every fold resolve to the same count —
    // hence a fold-invariant p_resp and an aligned p₁ across folds (design §3).
    let min_complement = folds.iter().map(|held| n - held.len()).min().unwrap_or(n);
    let mut fold_config = recipe.config.clone();
    fold_config.response_num_internal_knots =
        crate::families::transformation_normal::effective_response_num_internal_knots(
            &recipe.config,
            min_complement,
            p_cov,
            response_full.view(),
        );
    // Pin the resolved knot count: each fold's CTN refit must use exactly this
    // value, not re-derive it from its own response subsample. Without this the
    // data-driven complexity cap rounds to different counts per fold, so p_resp
    // (and p₁ = p_resp · p_cov) drifts across folds and the OOF Jacobian
    // assembly fails the fold-alignment check below ("cross-fit fold p₁ mismatch").
    fold_config.response_num_internal_knots_pinned = true;

    let mut z_oof = Array1::<f64>::zeros(n);
    let mut jac_oof: Option<Array2<f64>> = None;

    for held in &folds {
        if held.is_empty() {
            continue;
        }
        let held_set: std::collections::HashSet<usize> = held.iter().copied().collect();
        let complement: Vec<usize> = (0..n).filter(|i| !held_set.contains(i)).collect();
        if complement.is_empty() {
            return Err(
                "cross-fit fold left an empty training complement; too few rows for K folds"
                    .to_string(),
            );
        }

        // Refit the CTN on the complement (training) rows. The covariate design
        // it uses comes from the frozen spec, so its column geometry matches the
        // held-out Jacobian evaluation below and every other fold.
        let train_cov = data.values.select(Axis(0), &complement);
        let train_resp = crossfit_select_rows_1d(&response_full, &complement);
        let train_weights = crossfit_select_rows_1d(&weights_full, &complement);
        let train_offset = crossfit_select_rows_1d(&offset_full, &complement);

        let fold_fit = fit_transformation_normal(
            &train_resp,
            &train_weights,
            &train_offset,
            train_cov.view(),
            &frozen_cov_spec,
            &fold_config,
            &BlockwiseFitOptions::default(),
            &SpatialLengthScaleOptimizationOptions::default(),
            None,
        )?;

        // Evaluate the OOF score z and the OOF influence Jacobian J on fold f's
        // held-out rows from this fold's fitted CTN. The held-out offset rows
        // enter the PIT operating point identically to the Stage-1 row build, so
        // the emitted z matches the fitted model when the recipe carries an
        // offset column (zeros otherwise ⇒ no-op).
        let held_cov = data.values.select(Axis(0), held);
        let held_resp = crossfit_select_rows_1d(&response_full, held);
        let held_offset = crossfit_select_rows_1d(&offset_full, held);

        let jac = crate::families::marginal_slope_orthogonal::score_influence_jacobian(
            &fold_fit,
            &held_resp,
            held_cov.view(),
            &held_offset,
        )?;

        if jac.columns.nrows() != held.len() {
            return Err(format!(
                "cross-fit fold Jacobian row count {} != held-out fold size {}",
                jac.columns.nrows(),
                held.len()
            ));
        }
        if jac.z.len() != held.len() {
            return Err(format!(
                "cross-fit fold OOF z length {} != held-out fold size {}",
                jac.z.len(),
                held.len()
            ));
        }

        let p1 = jac.columns.ncols();
        let jac_full = jac_oof.get_or_insert_with(|| Array2::<f64>::zeros((n, p1)));
        if jac_full.ncols() != p1 {
            return Err(format!(
                "cross-fit fold p₁ mismatch: this fold has {p1} columns but a prior fold had {}; \
                 the frozen response/covariate basis failed to align across folds",
                jac_full.ncols()
            ));
        }

        for (local, &global) in held.iter().enumerate() {
            z_oof[global] = jac.z[local];
            for c in 0..p1 {
                jac_full[[global, c]] = jac.columns[[local, c]];
            }
        }
    }

    let jac_oof = jac_oof.ok_or_else(|| {
        "cross-fit produced no folds with held-out rows; cannot assemble OOF Jacobian".to_string()
    })?;

    Ok(Some(CrossFitScoreCalibration { z_oof, jac_oof }))
}

pub fn fit_model(request: FitRequest<'_>) -> Result<FitResult, WorkflowError> {
    // Single warm-start chokepoint: open a persistent cache session
    // keyed on the FitRequest's exact family-shape fingerprint, and
    // attach it to the variant's BlockwiseFitOptions (or top-level
    // slot). Family-specific fit functions then consult
    // `options.cache_session` to seed θ from the last accepted iterate
    // and checkpoint accepted iterates. This is what makes warm-start
    // uniform across every model class — adding a new family does NOT
    // require remembering to wire the cache.
    //
    // Hierarchical near-match: if the exact key has no entry, try a
    // data-independent seed prefix. Two fits that share the same family,
    // link, baseline, and column structure but differ in their row sets
    // (cross-validation folds, hyperparameter sweeps, anything refitting
    // structurally identical models on related data) get their first
    // fit's ρ as a seed instead of cold-starting. Save always goes to
    // the exact key so future identical refits get exact hits.
    let mut request = request;
    let exact_key = request.cache_key();
    let seed_key = request.cache_seed_key();
    if let Some(session) = crate::solver::persistent_warm_start::open_outer_session(&exact_key) {
        let exact_present = session.peek_load().is_some();
        if !exact_present
            && let Some(seed) =
                crate::solver::persistent_warm_start::lookup_outer_iterate_payload(&seed_key)
        {
            let prior_obj = seed.objective.unwrap_or(f64::NAN);
            log::info!(
                "[CACHE] seed key={}.. via prefix family={} prior_obj={:.6e}",
                &exact_key[..8.min(exact_key.len())],
                request.family_tag(),
                prior_obj,
            );
            session.preload(seed);
        }
        request.attach_cache_session(session);
    }
    // Mirror checkpoints and finalize to the seed-prefix keyspace so the
    // *next* fit with related-but-not-identical structure can pick this run's
    // ρ up via the prefix lookup above, even if the current process is killed
    // before convergence.
    let mirror_session = crate::solver::persistent_warm_start::open_outer_session(&seed_key);
    if let Some(mirror) = mirror_session.as_ref() {
        request.attach_cache_mirror(Arc::clone(mirror));
    }
    // Each `fit_*_model` helper still returns `Result<_, String>` internally;
    // the boundary conversion happens here so the public API returns
    // `WorkflowError::IntegrationFailed` carrying the underlying solver text.
    let wrap_solver_err =
        |reason: String| -> WorkflowError { WorkflowError::IntegrationFailed { reason } };
    match request {
        FitRequest::Standard(request) => fit_standard_model(request)
            .map(FitResult::Standard)
            .map_err(wrap_solver_err),
        FitRequest::GaussianLocationScale(request) => fit_gaussian_location_scale_model(request)
            .map(FitResult::GaussianLocationScale)
            .map_err(wrap_solver_err),
        FitRequest::BinomialLocationScale(request) => fit_binomial_location_scale_model(request)
            .map(FitResult::BinomialLocationScale)
            .map_err(wrap_solver_err),
        FitRequest::DispersionLocationScale(request) => {
            fit_dispersion_location_scale_model(request)
                .map(FitResult::DispersionLocationScale)
                .map_err(wrap_solver_err)
        }
        FitRequest::SurvivalLocationScale(request) => {
            // Outermost defensive catch: any path that surfaces empty
            // `block_states` to a family method (multiple possible —
            // `validate_joint_states`, `blockwise_fit_from_parts`, etc.)
            // produces a cryptic "expects N blocks, got 0" Python
            // exception. The v0.3.31 projected-kernel REML fix should make
            // the outer ρ-gradient cancel at large λ, preventing the
            // ARC stall that triggers the downstream empty-states path —
            // but as a last resort we convert any remaining error of that
            // shape into a user-actionable message so the Python caller
            // sees a clear cause instead of a Rust assertion artifact.
            match fit_survival_location_scale_model(request).map(FitResult::SurvivalLocationScale) {
                Ok(fit) => Ok(fit),
                Err(e)
                    if e.contains("expects 3 blocks, got 0")
                        || e.contains("expects 4 blocks, got 0")
                        || (e.contains("block_states") && e.contains("got 0"))
                        || e.contains("blockwise fit requires at least one block state") =>
                {
                    Err(WorkflowError::IntegrationFailed {
                        reason: format!(
                            "survival location-scale fit failed: the smoothing-parameter optimizer \
                             landed at a degenerate iterate where the inner solver's block state \
                             was empty. This is the symptom of an under-identified smooth driven \
                             to a numerically pathological λ (e.g. exp(20+)) on a small-data \
                             subsample. Try: (1) reducing covariate count, (2) increasing n_train, \
                             (3) `baseline_target=\"linear\"` to drop the parametric baseline, or \
                             (4) `noise_formula=\"1\"` to drop the noise GAM. Underlying error: {e}"
                        ),
                    })
                }
                Err(reason) => Err(wrap_solver_err(reason)),
            }
        }
        FitRequest::SurvivalTransformation(request) => fit_survival_transformation_model(request)
            .map(FitResult::SurvivalTransformation)
            .map_err(wrap_solver_err),
        FitRequest::BernoulliMarginalSlope(request) => fit_bernoulli_marginal_slope_model(request)
            .map(FitResult::BernoulliMarginalSlope)
            .map_err(wrap_solver_err),
        FitRequest::SurvivalMarginalSlope(request) => fit_survival_marginal_slope_model(request)
            .map(FitResult::SurvivalMarginalSlope)
            .map_err(wrap_solver_err),
        FitRequest::LatentSurvival(request) => fit_latent_survival_model(request)
            .map(FitResult::LatentSurvival)
            .map_err(wrap_solver_err),
        FitRequest::LatentBinary(request) => fit_latent_binary_model(request)
            .map(FitResult::LatentBinary)
            .map_err(wrap_solver_err),
        FitRequest::TransformationNormal(request) => fit_transformation_normal_model(request)
            .map(FitResult::TransformationNormal)
            .map_err(wrap_solver_err),
    }
}

// ---------------------------------------------------------------------------
// High-level formula-to-fit API
// ---------------------------------------------------------------------------

use crate::families::survival_construction::{
    SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeBasisConfig,
    add_survival_time_derivative_guard_offset, append_zero_tail_columns,
    baseline_chain_rule_gradient, build_latent_survival_baseline_offsets,
    build_survival_time_basis, build_survival_time_offsets_for_likelihood,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    initial_survival_baseline_config_for_fit, location_scale_uses_probit_survival_baseline,
    marginal_slope_baseline_chain_rule_gradient, marginal_slope_baseline_chain_rule_hessian,
    normalize_survival_time_pair, optimize_survival_baseline_config,
    optimize_survival_baseline_config_with_gradient,
    optimize_survival_baseline_config_with_gradient_only, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config, positive_survival_time_seed,
    require_structural_survival_time_basis, resolve_survival_marginal_slope_time_anchor_value,
    resolve_survival_time_anchor_value, resolved_survival_time_basis_config_from_build,
    survival_derivative_guard_for_likelihood,
};
use crate::families::survival_location_scale::{
    SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER, SurvivalCovariateTermBlockTemplate,
    TimeBlockInput, TimeWiggleBlockInput, residual_distribution_inverse_link,
};
use crate::inference::data::EncodedDataset as Dataset;
use crate::inference::formula_dsl::{
    LinkChoice, LinkWiggleFormulaSpec, ParsedFormula, ParsedTerm, effectivelinkwiggle_formulaspec,
    marginal_slope_logslope_surfaces, parse_formula, parse_link_choice,
    parse_matching_auxiliary_formula, parse_surv_response,
    require_inverse_link_supports_joint_wiggle, validate_marginal_slope_z_column_exclusion,
};
use crate::term_builder::{
    SECONDARY_CENTER_CAP_OPTION, build_termspec, column_map_with_alias, enable_scale_dimensions,
    has_explicit_countwith_basis_alias, resolve_role_col, resolve_smooth_type_name,
    smooth_type_uses_spatial_center_heuristic,
};

/// Non-formula configuration for model fitting. All fields have sensible defaults.
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Family: "gaussian", "binomial", "poisson", "negative-binomial",
    /// "gamma", "tweedie" (alias "tw"; variance power fixed at p = 1.5), or
    /// None for auto-detect.
    pub family: Option<String>,
    /// Fixed size/overdispersion parameter for `family="negative-binomial"`.
    pub negative_binomial_theta: Option<f64>,
    /// Link: "identity", "logit", "probit", "cloglog", "sas", "beta-logistic", or None.
    pub link: Option<String>,
    /// Whether to use flexible (wiggle-augmented) link.
    pub flexible_link: bool,
    /// Optional additive offset column for the primary linear predictor.
    pub offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    pub noise_offset_column: Option<String>,
    /// Optional family-level frailty modifier.
    pub frailty: Option<FrailtySpec>,

    // Survival-specific
    /// Baseline target: "linear", "weibull", "gompertz", "gompertz-makeham".
    pub baseline_target: String,
    pub baseline_scale: Option<f64>,
    pub baseline_shape: Option<f64>,
    pub baseline_rate: Option<f64>,
    pub baseline_makeham: Option<f64>,
    /// Time basis: "ispline" or "none".
    pub time_basis: String,
    pub time_degree: usize,
    pub time_num_internal_knots: usize,
    pub time_smooth_lambda: f64,
    /// Survival likelihood mode: "location-scale", "transformation", "weibull",
    /// "marginal-slope", "latent", or "latent-binary".
    pub survival_likelihood: String,
    /// Residual distribution: "gaussian", "logistic", "gumbel".
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,

    // Location-scale (GAMLSS)
    /// If set, fit a location-scale model with this formula for the noise parameter.
    pub noise_formula: Option<String>,

    // Marginal-slope
    /// Formula for the log-slope model (survival marginal-slope or Bernoulli marginal-slope).
    pub logslope_formula: Option<String>,
    /// Column name for the z (exposure/dose) variable in marginal-slope models.
    pub z_column: Option<String>,
    /// Optional non-negative per-row training weights column.
    pub weight_column: Option<String>,
    /// Internal CTN Stage-1 provenance for the marginal-slope `z` column.
    ///
    /// When the marginal-slope `z` was generated by a transformation-normal
    /// Stage-1 fit, the orchestration layer fills this with the Stage-1 recipe.
    /// Its presence is the sole auto-enable signal for cross-fitted, Neyman-
    /// orthogonal score calibration (#461): the materializer cross-fits the CTN
    /// to produce out-of-fold `z` and the score-influence Jacobian `J`, replaces
    /// the raw `z` with `z_oof`, and absorbs `J` as a leakage-projection block in
    /// Stage-2. This is in-process plumbing only — there is no CLI flag, env var,
    /// or feature gate. `None` ⇒ raw `z` with the free-warp `score_warp`
    /// fallback. See [`CtnStage1Recipe`].
    pub ctn_stage1: Option<CtnStage1Recipe>,

    // Fitting options
    pub scale_dimensions: bool,
    /// Enable exact spatial adaptive regularization for standard formula fits.
    /// `None` uses the quality-first automatic policy. The current automatic
    /// policy leaves LAREG off unless explicitly requested because the
    /// optimizer's REML-selected local weights can over-regularize small
    /// high-yield spatial signals.
    pub adaptive_regularization: Option<bool>,
    pub ridge_lambda: f64,

    /// Route the fit through the transformation-normal family.  When set, the
    /// formula terms are treated as the covariate side of the transformation
    /// model and the response basis is built internally.  Incompatible with
    /// `noise_formula` and with `Surv(...)` responses.
    pub transformation_normal: bool,

    /// Enable Firth bias reduction for standard single-parameter families.
    pub firth: bool,
    /// Optional cap on the REML/LAML outer smoothing-parameter iterations for
    /// standard formula fits. `None` uses the production default.
    pub outer_max_iter: Option<usize>,

    /// GPU backend selection policy. `Auto` uses supported device kernels for
    /// large workloads, `Off` pins execution to CPU kernels, and `Force` fails
    /// loudly when a requested GPU kernel has no compiled backend.
    pub gpu_policy: crate::gpu::GpuPolicy,
    /// Optional override of the [`crate::resource::ResourcePolicy`] used when
    /// planning spatial bases (TPS / Matern / Duchon) during term construction.
    /// When `None`, the default-library policy is used.
    pub resource_policy: Option<crate::resource::ResourcePolicy>,

    /// Optional per-group metadata supplied by the caller. Fitting ignores this
    /// field; saved-model builders pass it through so deployment consumers can
    /// recover group provenance.
    pub group_metadata: Option<BTreeMap<String, JsonValue>>,

    /// Optional user-defined coefficient groups with separate precision
    /// parameters. Group-local priors, including catalog-metadata-informed
    /// Gamma precision hyperpriors, are resolved during design setup.
    pub coefficient_groups: Vec<CoefficientGroupSpec>,

    /// Optional per-existing-penalty-block Gamma(shape, rate) precision
    /// hyperpriors keyed by penalty-block label. This is the
    /// catalog-metadata-informed-prior hook for models that do not need a new
    /// user-defined coefficient group.
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,

    /// Python `gamfit.fit(..., latents={...})` configuration. This reaches
    /// the standard formula workflow as an owned latent-coordinate block:
    /// the named smooth's synthetic covariates are rebuilt from `t`, and
    /// joint REML optimizes `[rho, vec(t)]` through latent design hyper-dirs.
    pub latents: Option<JsonValue>,
    /// Python `gamfit.fit(..., penalties=[...])` analytic-penalty descriptors,
    /// validated against the declared latent-coordinate blocks before a
    /// standard latent fit starts.
    pub analytic_penalties: Option<JsonValue>,
    /// Formula-path latent topology selector descriptor. The selector itself
    /// fits candidates through the ordinary workflow; this slot lets callers
    /// request and validate that path from the same config registry.
    pub topology_auto_selector: Option<crate::solver::topology_selector::TopologyAutoSelector>,
    /// `gamfit.fit(..., smooths={...})` Python kwarg routed through the FFI
    /// bridge. JSON object keyed by formula symbol (single column name or
    /// comma-joined tuple) → smooth descriptor (`{"kind": "duchon",
    /// "centers": [[...], ...], ...}`). Applied as a post-processing step on
    /// the [`TermCollectionSpec`] produced by the formula DSL: each smooth
    /// term whose `feature_cols` match a registry key has its kind-specific
    /// tunables (centers, knots, kernel hyperparameters) overridden with the
    /// user-supplied values. The single canonical lowering path guarantees
    /// `smooths={"x": Duchon(centers=K)}` (integer) produces a bit-identical
    /// block spec to writing `duchon(x, centers=K)` in the formula; only
    /// explicit array-valued `centers=` differs, routing through
    /// `CenterStrategy::UserProvided` instead of `FarthestPoint`/`EqualMass`.
    pub smooth_overrides: Option<JsonValue>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            family: None,
            negative_binomial_theta: None,
            link: None,
            flexible_link: false,
            offset_column: None,
            noise_offset_column: None,
            frailty: None,
            baseline_target: "linear".into(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".into(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            survival_likelihood: "location-scale".into(),
            survival_distribution: "gaussian".into(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            noise_formula: None,
            logslope_formula: None,
            z_column: None,
            weight_column: None,
            ctn_stage1: None,
            scale_dimensions: false,
            adaptive_regularization: None,
            ridge_lambda: 1e-6,
            transformation_normal: false,
            firth: false,
            outer_max_iter: None,
            gpu_policy: crate::gpu::GpuPolicy::Auto,
            resource_policy: None,
            group_metadata: None,
            coefficient_groups: Vec::new(),
            penalty_block_gamma_priors: Vec::new(),
            latents: None,
            analytic_penalties: None,
            topology_auto_selector: None,
            smooth_overrides: None,
        }
    }
}

/// Resolve the [`crate::resource::ResourcePolicy`] backing term construction
/// for a given [`FitConfig`] + dataset.
///
/// If the caller hasn't supplied an explicit policy override, derive one from
/// the shape of the problem via
/// [`crate::resource::ResourcePolicy::for_problem`]. At large scale (n_rows
/// >= 100k or the marginal-slope large-scale path active) this returns
/// > `analytic_operator_required` so that any silent dense materialization in
/// > the term-construction layer fails fast rather than allocating tens of GiB;
/// > at small scale it falls through to the permissive default-library policy
/// > so that non-operator bases still build cleanly.
///
/// `p_estimate = 0` because the per-block coefficient count isn't known until
/// the spec has been built; the n_rows and hints triggers are sufficient to
/// flip strict mode for every shape that needs it.
pub(crate) fn resolved_resource_policy(
    config: &FitConfig,
    data: &Dataset,
    hints: crate::resource::ProblemHints,
) -> crate::resource::ResourcePolicy {
    if let Some(p) = config.resource_policy.clone() {
        return p;
    }
    crate::resource::ResourcePolicy::for_problem(data.values.nrows(), 0, hints)
}

fn marginal_slope_hints(config: &FitConfig) -> crate::resource::ProblemHints {
    crate::resource::ProblemHints {
        marginal_slope_large_scale_active: config.logslope_formula.is_some()
            || config.z_column.is_some(),
    }
}

/// The result of materializing a formula + config against a dataset.
pub struct MaterializedModel<'a> {
    pub request: FitRequest<'a>,
    pub inference_notes: Vec<String>,
}

/// Parse, materialize, and fit a model in one call.
pub fn fit_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FitResult, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    // `fit_model` already returns `WorkflowError` end-to-end; propagate it
    // directly instead of stringifying then re-wrapping.
    fit_model(mat.request)
}

/// Parse a formula, resolve it against a dataset, and produce a ready-to-fit `FitRequest`.
pub fn materialize<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    crate::gpu::configure_global_policy(config.gpu_policy);
    let parsed = parse_formula(formula)?;
    let col_map = data.column_map();

    if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        if config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with a Surv(...) response"
                    .to_string(),
            });
        }
        // `materialize_*` now return `WorkflowError` directly so the typed
        // `ColumnNotFound` payload (and any future variant-typed leaf
        // errors) survive the dispatcher hop instead of being flattened
        // into `IntegrationFailed { reason: String }`.
        materialize_survival(
            &parsed,
            data,
            &col_map,
            config,
            entry_col.as_deref(),
            &exit_col,
            &event_col,
        )
    } else {
        // Non-survival response: `timewiggle(...)` and `survmodel(...)` are
        // structurally meaningless (there is no baseline hazard / time axis to
        // wiggle and no survival likelihood to configure). They are parsed into
        // `ParsedFormula` but consumed *only* by `materialize_survival`; without
        // this guard every non-survival materializer below would silently drop
        // them, fitting an ordinary GAM while the user believes they requested a
        // time-varying / survival model (#371). Reject here — the single
        // chokepoint for all non-survival paths — mirroring the symmetric
        // auxiliary-formula rejection in `validate_auxiliary_formula_controls`.
        reject_survival_only_terms_for_nonsurvival(&parsed)?;
        if config.transformation_normal {
            // Issue #789A: a Bernoulli marginal-slope request with
            // `transformation_normal=true` used to dispatch as a CTN fit while
            // retaining marginal-slope controls, leaving the transformation path
            // in a non-advancing loop. CTN score calibration now uses the
            // explicit `ctn_stage1` recipe instead, so the legacy boolean is a
            // hard configuration error for marginal-slope requests.
            reject_marginal_slope_controls_for_transformation_normal(config)?;
            if config.noise_formula.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "transformation_normal cannot be combined with noise_formula"
                        .to_string(),
                });
            }
            materialize_transformation_normal(&parsed, data, &col_map, config)
        } else if config.logslope_formula.is_some() || config.z_column.is_some() {
            materialize_bernoulli_marginal_slope(&parsed, data, &col_map, config)
        } else if config.noise_formula.is_some() {
            materialize_location_scale(&parsed, data, &col_map, config)
        } else {
            materialize_standard(&parsed, data, &col_map, config)
        }
    }
}

fn reject_marginal_slope_controls_for_transformation_normal(
    config: &FitConfig,
) -> Result<(), WorkflowError> {
    let family_requests_marginal_slope = config.family.as_deref().is_some_and(|family| {
        let canonical = family.to_ascii_lowercase().replace('_', "-");
        canonical == "bernoulli-marginal-slope" || canonical == "binary-marginal-slope"
    });
    if family_requests_marginal_slope
        || config.logslope_formula.is_some()
        || config.z_column.is_some()
        || config.ctn_stage1.is_some()
    {
        return Err(WorkflowError::InvalidConfig {
            reason: "transformation_normal cannot be combined with marginal-slope family controls"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject `timewiggle(...)` / `survmodel(...)` in a formula whose response is
/// not `Surv(...)`.
///
/// These two DSL controls only have meaning under the survival likelihood: a
/// `timewiggle(...)` term parameterizes the time-varying baseline-hazard /
/// log-cumulative-hazard surface, and `survmodel(...)` selects the survival
/// likelihood mode. Both are read exclusively by `materialize_survival`. When
/// the main formula has no `Surv(...)` response, leaving them unguarded means
/// the term is parsed and option-validated and then dropped on the floor —
/// the contract violation reported in #371. We error instead, with the same
/// "only supported in the main survival formula" phrasing the auxiliary-formula
/// path already uses.
fn reject_survival_only_terms_for_nonsurvival(parsed: &ParsedFormula) -> Result<(), WorkflowError> {
    if parsed.timewiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "timewiggle(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    if parsed.survivalspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survmodel(...) is only supported in the main survival formula \
                     (a formula with a Surv(...) response); it is meaningless for a \
                     non-survival response and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

/// Reject an *explicitly requested* `linkwiggle(...)` term when the resolved
/// response family is not binomial.
///
/// `linkwiggle(...)` adds a spline-flexible correction to the *link* function
/// (logit / probit / cloglog), which only carries meaning for a binomial mean
/// model — the standard and location-scale materializers wire `wiggle` into the
/// fit only inside their `family.is_binomial()` arm. For a Gaussian / Gamma /
/// Poisson / etc. response the term is built and then dropped on the floor,
/// the same silent-no-op contract violation as #371. We error here.
///
/// This guards only the *explicit* formula term (`parsed.linkwiggle`), not the
/// implicit wiggle auto-derived from a `Flexible` link choice: a flexible link
/// requested against a non-binomial family is a separate, already-handled link
/// concern, and silently declining to add a binomial-only correction there is
/// the intended behavior rather than a dropped user-authored term.
fn reject_explicit_linkwiggle_for_nonbinomial(
    parsed: &ParsedFormula,
    family: &LikelihoodSpec,
) -> Result<(), WorkflowError> {
    if parsed.linkwiggle.is_some() && !family.is_binomial() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) corrects the link function of a binomial mean model \
                     and is only supported for a binomial response; it is meaningless for \
                     the resolved non-binomial family and would otherwise be silently ignored"
                .to_string(),
        });
    }
    Ok(())
}

/// Detect whether a response column is binary (0/1 only).
pub fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Verify that the dataset has at least as many rows as the smooth terms in
/// `spec` need for their bases to be well-posed.
///
/// Each [`SmoothBasisSpec`] owns its own `min_sample_rows` lower bound — the
/// B-spline knot count, the *penalized* tensor-product floor (the sum of the
/// per-marginal column counts, not their Kronecker product, because a `te()`
/// is regularized and its effective dof is a small fraction of the column
/// count), the PCA matrix width — so this helper is a thin sum-and-compare:
/// the workflow has no per-basis-kind knowledge. Adding a new smooth kind
/// extends the basis `match` in `min_sample_rows`, not this gate.
///
/// Catches the README-quickstart failure mode (#309) where `n=4` against
/// `y ~ s(x)` would otherwise surface as an opaque `cached inner beta has
/// length 8` message from the inner-state seeding hook.
fn check_smooth_capacity(
    spec: &crate::terms::smooth::TermCollectionSpec,
    n_rows: usize,
    response_name: &str,
) -> Result<(), WorkflowError> {
    // Intercept + 1 dof for the smoothing-parameter optimizer.
    let mut required: usize = 2;
    let mut per_term: Vec<(String, usize)> = Vec::new();
    for term in &spec.smooth_terms {
        let need = term.basis.min_sample_rows();
        required = required.saturating_add(need);
        per_term.push((term.name.clone(), need));
    }
    if per_term.is_empty() || n_rows >= required {
        return Ok(());
    }
    let breakdown = per_term
        .iter()
        .map(|(name, k)| format!("{name}≥{k}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "not enough observations to fit the requested formula: dataset has n={n_rows} \
             rows but the smooth terms on response '{response_name}' need at least \
             {required} rows total ({breakdown}, plus intercept + smoothing-parameter dof) \
             before REML estimation is well-posed. \
             Fix: add more training rows, replace `s(x)` with a linear term, or pass a \
             smaller basis via `s(x, k=3)`."
        ),
    })
}

/// Project an ingest-layer [`ColumnKindTag`] (plus the column's level table)
/// onto the [`ResponseColumnKind`] consumed by the family layer.
///
/// `Categorical` carries the source-string levels through so the
/// auto-inference refusal can echo them; `Binary` short-circuits the
/// numeric scan inside [`ResponseFamily::infer_from_response`]; `Continuous`
/// maps to `Numeric` and the family layer scans `y` itself to decide
/// Gaussian vs. Binomial.
pub(crate) fn response_column_kind(data: &Dataset, y_col: usize) -> ResponseColumnKind {
    match data.column_kinds.get(y_col) {
        Some(ColumnKindTag::Categorical) => ResponseColumnKind::Categorical {
            levels: data
                .schema
                .columns
                .get(y_col)
                .map(|sc| sc.levels.clone())
                .unwrap_or_default(),
        },
        Some(ColumnKindTag::Binary) => ResponseColumnKind::Binary,
        Some(ColumnKindTag::Continuous) | None => ResponseColumnKind::Numeric,
    }
}

/// Legality of a `(response family, link)` pairing.
///
/// This is the single source of truth for which links a given response family
/// accepts. It is consulted only when the caller supplied an *explicit* family
/// together with a link (`family=..., link(type=...)`): the link must be
/// validated against that family rather than the family re-inferred from the
/// link. The legal pairings are:
///
/// * `Gaussian` + `Identity`
/// * `{Poisson, Gamma, Tweedie, NegativeBinomial}` + `Log`
/// * `Beta` + `Logit`
/// * `Binomial` + `{Logit, Probit, CLogLog, Sas, BetaLogistic}` (and the
///   Logit-shaped `Mixture`, handled by the caller via `mixture_components`)
///
/// `RoystonParmar` is a flexible-parametric survival family whose link is fixed
/// at construction and is never reached through the scalar link-choice path, so
/// it accepts no link override here.
fn link_legal_for_family(response: &ResponseFamily, link: LinkFunction) -> bool {
    match response {
        ResponseFamily::Gaussian => matches!(link, LinkFunction::Identity),
        ResponseFamily::Poisson
        | ResponseFamily::Gamma
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. } => matches!(link, LinkFunction::Log),
        ResponseFamily::Beta { .. } => matches!(link, LinkFunction::Logit),
        ResponseFamily::Binomial => matches!(
            link,
            LinkFunction::Logit
                | LinkFunction::Probit
                | LinkFunction::CLogLog
                | LinkFunction::Sas
                | LinkFunction::BetaLogistic
        ),
        ResponseFamily::RoystonParmar => false,
    }
}

/// Resolve a family from an optional name, optional link choice, and response data.
///
/// `y_kind` describes the *source* representation of the response column
/// (string-valued `Categorical`, numeric `Binary` short-circuit, or generic
/// `Numeric`). It is consulted only on the auto-detect path — explicit
/// `family=...` always wins — but is required there because the same numeric
/// `y = [0.0, 1.0, ...]` payload may come from a real binary outcome or from
/// a categorical column whose two levels happened to encode to those indices.
/// Routing the kind through [`ResponseFamily::infer_from_response`] is what
/// stops the auto-detector from silently inferring Binomial off encoded
/// strings (see `tests/issues/issue_304`).
pub fn resolve_family(
    family: Option<&str>,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    let nb_theta = negative_binomial_theta.unwrap_or(1.0);
    if !nb_theta.is_finite() || nb_theta <= 0.0 {
        return Err(format!(
            "negative-binomial theta must be finite and > 0; got {nb_theta}"
        ));
    }
    // `link_pinned = true` means the family name carried a specific link suffix
    // (e.g. "binomial-probit"); `false` means the user only declared the response
    // family (e.g. "binomial") and any link_choice may legally refine the link
    // without being treated as a contradiction.
    let explicit: Option<(LikelihoodSpec, bool)> = match family {
        Some(name) => {
            // Accept both '-' and '_' as separators so e.g. "binomial_logit" and
            // "negative-binomial" resolve identically. Canonicalize to '-'.
            let canonical = name.to_ascii_lowercase().replace('_', "-");
            let resolved = match canonical.as_str() {
                "gaussian" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    false,
                ),
                "binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "binomial-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "binomial-probit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::Probit),
                    ),
                    true,
                ),
                "binomial-cloglog" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::Standard(StandardLink::CLogLog),
                    ),
                    true,
                ),
                "latent-cloglog-binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Binomial,
                        InverseLink::LatentCLogLog(
                            LatentCLogLogState::new(1.0)
                                .map_err(|err| format!("latent cloglog default state: {err}"))?,
                        ),
                    ),
                    true,
                ),
                "poisson" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Poisson,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // #983: a user-supplied `--negative-binomial-theta` holds θ
                // fixed at exactly that value (`theta_fixed = true` →
                // `FixedNegBinTheta` scale → the PIRLS refresh gate
                // `negbin_theta_is_estimated()` stays closed). With no flag,
                // θ is the running ML estimate (the #802 default seed 1.0).
                "nb" | "negbin" | "negative-binomial" => (
                    LikelihoodSpec::new(
                        ResponseFamily::NegativeBinomial {
                            theta: nb_theta,
                            theta_fixed: negative_binomial_theta.is_some(),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                "negative-binomial-log" => (
                    LikelihoodSpec::new(
                        ResponseFamily::NegativeBinomial {
                            theta: nb_theta,
                            theta_fixed: negative_binomial_theta.is_some(),
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    true,
                ),
                "beta" | "beta-regression" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    false,
                ),
                "beta-logit" | "beta-regression-logit" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 1.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    true,
                ),
                "gamma" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gamma,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                // Royston-Parmar flexible-parametric survival and the
                // transformation-normal response model are CLI/formula families
                // whose materialization is dispatched before the scalar GLM
                // family resolver runs (survival via `Surv(...)`, transformation
                // via the dedicated transformation-normal path). They are listed
                // here so this resolver is the single total source of truth for
                // every family name the surface accepts: `royston-parmar` maps to
                // the canonical flexible-parametric likelihood, and
                // `transformation-normal` shares Gaussian-identity scalar
                // semantics (the transformation is learned outside this spec).
                "royston-parmar" => (LikelihoodSpec::royston_parmar(), true),
                "transformation-normal" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Gaussian,
                        InverseLink::Standard(StandardLink::Identity),
                    ),
                    true,
                ),
                // Tweedie compound-Poisson-Gamma family. The variance power
                // p must lie strictly in (1, 2); we default to mgcv's
                // canonical p = 1.5. The link is fixed to log (the only link
                // wired through the Tweedie working-response and dispersion
                // machinery). "tw" matches mgcv's family alias.
                "tweedie" | "tw" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie { p: 1.5 },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    false,
                ),
                "tweedie-log" => (
                    LikelihoodSpec::new(
                        ResponseFamily::Tweedie { p: 1.5 },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    true,
                ),
                "multinomial" | "multinomial-logit" | "categorical" | "categorical-logit"
                | "softmax" => {
                    // Multinomial-logit is a vector-response family with K-1
                    // active linear predictors and a per-row dense Fisher
                    // block — it cannot be represented by the scalar
                    // `LikelihoodSpec` (one `ResponseFamily` × one
                    // `InverseLink`) that this entry point produces.
                    //
                    // The principled coefficient-space solver lives in
                    // `crate::families::multinomial::fit_penalized_multinomial`,
                    // which routes the canonical
                    // `MultinomialLogitLikelihood: VectorLikelihood` through
                    // `crate::pirls::dense_block_xtwx` in output-major
                    // coefficient ordering. The forthcoming
                    // `gamfit.fit_multinomial(...)` Python entry exposes that
                    // path with formula → design wiring; until that wrapper
                    // lands, callers reach the driver directly through the
                    // FFI surface.
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "family '{name}' is a vector-response family; use \
                             the dedicated multinomial entry point \
                             (`crate::families::multinomial::fit_penalized_multinomial` \
                             in Rust, or `gamfit.fit_multinomial(...)` in Python) \
                             rather than the scalar `fit(family=...)` path"
                        ),
                    }
                    .into());
                }
                _ => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!("unknown family '{name}'"),
                    }
                    .into());
                }
            };
            Some(resolved)
        }
        None => {
            if negative_binomial_theta.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason: "negative_binomial_theta requires family='negative-binomial'"
                        .to_string(),
                }
                .into());
            }
            None
        }
    };

    if let Some(choice) = link_choice {
        let from_link: LikelihoodSpec = if let Some(components) = choice.mixture_components.as_ref()
        {
            let n = components.len();
            let free = n.saturating_sub(1);
            let mix_spec = MixtureLinkSpec {
                components: components.clone(),
                initial_rho: Array1::<f64>::zeros(free),
            };
            let state = state_fromspec(&mix_spec)
                .map_err(|err| format!("mixture link initial state: {err}"))?;
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state))
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                ),
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodSpec::new(
                            ResponseFamily::Poisson,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    } else {
                        LikelihoodSpec::new(
                            ResponseFamily::Gamma,
                            InverseLink::Standard(StandardLink::Log),
                        )
                    }
                }
                LinkFunction::Logit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                ),
                LinkFunction::Probit => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                ),
                LinkFunction::CLogLog => LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                ),
                LinkFunction::Sas => {
                    // The SAS initial state (epsilon, log_delta) is carried into
                    // the fit through `FitOptions.sas_link`, not the family spec:
                    // the standard path's `effective_sas_link_for_family` rebuilds
                    // the inverse link from that option, overriding whatever the
                    // family embeds here. The canonical zero seed is therefore the
                    // correct, link-only placeholder for family resolution.
                    let state = state_from_sasspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("SAS link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(state))
                }
                LinkFunction::BetaLogistic => {
                    let state = state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .map_err(|err| format!("Beta-Logistic link initial state: {err}"))?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(state))
                }
            }
        };
        if let Some((explicit_spec, link_pinned)) = explicit.as_ref() {
            // An explicit response family was supplied: never re-infer the
            // family from the link. Validate that the requested link is legal
            // for *this* family, then apply the link (carrying any embedded
            // Sas/BetaLogistic/Mixture state, which `from_link.link` already
            // holds) to the explicit family's response variant (preserving e.g.
            // NB theta, Tweedie p, Beta phi).
            let mixture_requested = choice.mixture_components.is_some();
            let legal = if mixture_requested {
                // The mixture link is a Binomial latent construct; it has no
                // legal pairing with any other response family.
                matches!(explicit_spec.response, ResponseFamily::Binomial)
            } else {
                link_legal_for_family(&explicit_spec.response, choice.link)
            };
            if !legal {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "link '{}' is not supported for family '{}'",
                        choice.link.name(),
                        explicit_spec.response.name()
                    ),
                }
                .into());
            }
            // A family name that pinned its own link (e.g. "binomial-probit")
            // may not be re-pointed at a different link by `link(type=...)`.
            if *link_pinned && explicit_spec.link.link_function() != from_link.link.link_function()
            {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "family '{}' pins link '{}', which conflicts with requested link '{}'",
                        explicit_spec.name(),
                        explicit_spec.link.link_function().name(),
                        choice.link.name(),
                    ),
                }
                .into());
            }
            return Ok(LikelihoodSpec::new(
                explicit_spec.response.clone(),
                from_link.link,
            ));
        }
        return Ok(from_link);
    }

    if let Some((spec, _)) = explicit {
        return Ok(spec);
    }

    // Auto-detect: delegate to `ResponseFamily::infer_from_response` so the
    // refusal policy for non-numeric response columns lives in one place
    // (the family layer), not duplicated across every entry point. The link
    // is derived from the inferred response: Binomial → Logit, Gaussian →
    // Identity. The link_choice branch above already covered the case where
    // the user pinned a link without a family.
    let response = ResponseFamily::infer_from_response(y, y_kind).map_err(|refusal| {
        let err: String = WorkflowError::InvalidConfig {
            reason: refusal.message_for(response_name),
        }
        .into();
        err
    })?;
    let link = match response {
        ResponseFamily::Binomial => InverseLink::Standard(StandardLink::Logit),
        _ => InverseLink::Standard(StandardLink::Identity),
    };
    Ok(LikelihoodSpec::new(response, link))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Canonical termspec lowering path: formula DSL builds the initial
/// `SmoothBasisSpec`, then any `gamfit.fit(..., smooths={...})` Python
/// override registry entry whose `feature_cols` match the term's column set
/// replaces the spec's kind-specific tunables in place (explicit center
/// coordinate matrices, knot vectors, kernel hyperparameters). When all
/// descriptor fields default to the same values the DSL would auto-pick, the
/// override is a no-op and the spec is bit-identical to the formula-only
/// path. Callers that don't have overrides simply pass `smooth_overrides =
/// None`.
pub(crate) fn build_termspec_with_geometry_and_overrides(
    terms: &[ParsedTerm],
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
    scale_dimensions: bool,
    policy: &crate::resource::ResourcePolicy,
    smooth_overrides: Option<&JsonValue>,
) -> Result<TermCollectionSpec, WorkflowError> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes, policy)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    if let Some(overrides) = smooth_overrides {
        crate::terms::smooth_overrides::apply_smooth_overrides(
            &mut spec,
            overrides,
            data,
            inference_notes,
        )
        .map_err(|reason| WorkflowError::InvalidConfig { reason })?;
    }
    Ok(spec)
}

fn linear_term_training_column(
    data: &Dataset,
    term: &LinearTermSpec,
) -> Result<Array1<f64>, WorkflowError> {
    let cols = term.effective_feature_cols();
    if cols.is_empty() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linear term '{}' has no feature columns; cannot build its training column",
                term.name
            ),
        });
    }
    let n = data.values.nrows();
    let mut out = Array1::<f64>::ones(n);
    for &col in &cols {
        if col >= data.values.ncols() {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "linear term '{}' feature column {} out of bounds for {} columns",
                    term.name,
                    col,
                    data.values.ncols()
                ),
            }
            .into());
        }
        for row in 0..n {
            out[row] *= data.values[[row, col]];
        }
    }
    Ok(out)
}

fn residualize_against_orthonormal_basis(
    column: &Array1<f64>,
    basis: &[Array1<f64>],
) -> Array1<f64> {
    let mut residual = column.clone();
    for q in basis {
        let coeff = residual.dot(q);
        residual.scaled_add(-coeff, q);
    }
    residual
}

fn l2_norm(column: &Array1<f64>) -> f64 {
    column.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn prune_unidentified_linear_terms_for_marginal_slope(
    spec: &mut TermCollectionSpec,
    data: &Dataset,
    label: &str,
    inference_notes: &mut Vec<String>,
) -> Result<(), WorkflowError> {
    if spec.linear_terms.is_empty() {
        return Ok(());
    }

    let n = data.values.nrows();
    if n == 0 {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: cannot rank-check scalar terms on zero rows"),
        });
    }

    let mut basis = Vec::<Array1<f64>>::new();
    let intercept = Array1::<f64>::ones(n);
    let intercept_norm = l2_norm(&intercept);
    if intercept_norm == 0.0 || !intercept_norm.is_finite() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!("{label}: implicit intercept has invalid norm {intercept_norm}"),
        });
    }
    basis.push(intercept.mapv(|v| v / intercept_norm));

    let rank_alpha = crate::linalg::faer_ndarray::default_rrqr_rank_alpha();
    let mut scale = intercept_norm.max(1.0);
    let mut kept = Vec::<LinearTermSpec>::with_capacity(spec.linear_terms.len());
    let mut dropped = Vec::<String>::new();

    for term in &spec.linear_terms {
        let column = linear_term_training_column(data, term)?;
        let norm = l2_norm(&column);
        if !norm.is_finite() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!("{label}: linear term '{}' has non-finite norm", term.name),
            });
        }
        scale = scale.max(norm.max(1.0));
        let residual = residualize_against_orthonormal_basis(&column, &basis);
        let residual_norm = l2_norm(&residual);
        let tol = rank_alpha * f64::EPSILON * ((n + basis.len() + 1).max(1) as f64) * scale;
        let is_data_redundant = residual_norm <= tol;
        let has_constraints = term.coefficient_min.is_some() || term.coefficient_max.is_some();
        if is_data_redundant {
            if has_constraints {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: constrained linear term '{}' is redundant with the implicit \
                         intercept or earlier scalar terms; remove the constraint or the \
                         redundant term",
                        term.name
                    ),
                });
            }
            if term.double_penalty {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "{label}: explicitly penalized linear term '{}' is redundant with the \
                         implicit intercept or earlier scalar terms; remove the redundant term \
                         instead of relying on a ridge to identify a duplicate data direction",
                        term.name
                    ),
                });
            }
            dropped.push(format!(
                "{} (residual_norm={:.3e}, tol={:.3e})",
                term.name, residual_norm, tol
            ));
            continue;
        }
        if residual_norm > tol {
            basis.push(residual.mapv(|v| v / residual_norm));
        }
        kept.push(term.clone());
    }

    if !dropped.is_empty() {
        inference_notes.push(format!(
            "{label}: removed {} scalar term(s) that add no identifiable \
             direction beyond the implicit intercept and earlier scalar terms: {}",
            dropped.len(),
            dropped.join(", ")
        ));
        spec.linear_terms = kept;
    }
    Ok(())
}

fn standard_adaptive_regularization_options(
    config: &FitConfig,
) -> Option<AdaptiveRegularizationOptions> {
    let enabled = config.adaptive_regularization.unwrap_or(false);
    enabled.then(|| AdaptiveRegularizationOptions {
        enabled: true,
        ..AdaptiveRegularizationOptions::default()
    })
}

fn resolve_survival_marginal_slope_base_link(
    linkspec: Option<&crate::inference::formula_dsl::LinkFormulaSpec>,
) -> Result<InverseLink, String> {
    let Some(linkspec) = linkspec else {
        return Ok(InverseLink::Standard(StandardLink::Probit));
    };
    let choice = parse_link_choice(Some(&linkspec.link), false)?
        .ok_or_else(|| "invalid survival marginal-slope link".to_string())?;
    if choice.mixture_components.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "survival marginal-slope currently supports only link(type=probit)".to_string(),
        }
        .into());
    }
    match choice.link {
        LinkFunction::Probit => Ok(InverseLink::Standard(StandardLink::Probit)),
        other => Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival marginal-slope currently supports only link(type=probit), got {other:?}"
            ),
        }
        .into()),
    }
}

/// Canonical baseline-time stack shared by the workflow materializer and the
/// CLI survival path (`crate::bin::main`-side `run_survival`). Both entry points
/// build the survival time block identically — baseline offsets, derivative
/// guard, optional baseline time-wiggle augmentation — so the assembly lives
/// here once and the CLI consumes it through a thin re-export rather than
/// reconstructing the same decision tree.
pub struct PreparedSurvivalTimeStack {
    pub eta_offset_entry: Array1<f64>,
    pub eta_offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub time_design_entry: crate::matrix::DesignMatrix,
    pub time_design_exit: crate::matrix::DesignMatrix,
    pub time_design_derivative_exit: crate::matrix::DesignMatrix,
    pub time_penalties: Vec<Array2<f64>>,
    pub time_nullspace_dims: Vec<usize>,
    pub timewiggle_build: Option<crate::families::survival_construction::SurvivalTimeWiggleBuild>,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
}

pub fn prepare_survival_time_stack(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &crate::families::survival_construction::SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
    inverse_link: Option<&InverseLink>,
    time_anchor: f64,
    derivative_guard: f64,
    time_build: &crate::families::survival_construction::SurvivalTimeBuildOutput,
    effective_timewiggle: Option<&LinkWiggleFormulaSpec>,
    latent_loading: Option<crate::families::lognormal_kernel::HazardLoading>,
) -> Result<PreparedSurvivalTimeStack, String> {
    let (
        mut eta_offset_entry,
        mut eta_offset_exit,
        mut derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    ) = if let Some(loading) = latent_loading {
        let offsets =
            build_latent_survival_baseline_offsets(age_entry, age_exit, baseline_cfg, loading)?;
        (
            offsets.loaded_eta_entry,
            offsets.loaded_eta_exit,
            offsets.loaded_derivative_exit,
            offsets.unloaded_mass_entry,
            offsets.unloaded_mass_exit,
            offsets.unloaded_hazard_exit,
        )
    } else {
        // Baseline-hazard barrier conditioning for the marginal-slope likelihood
        // (gam#797). That likelihood carries `-d·log(qd1)`, a log-barrier on the
        // baseline-hazard time derivative `qd1 = X_d·β_time + derivative_offset`.
        // The default `baseline-target=linear` is DEGENERATE for this barrier:
        // `evaluate_survival_baseline` returns `(0, 0)` for Linear, so the offset
        // collapses to `derivative_guard` (1e-6) and the I-spline time seed starts
        // at `qd1 ≈ 1e-6` — exactly ON the barrier boundary, where the
        // self-concordant Newton step is `∝ qd1` (intrinsically ~1e-4), the
        // barrier gradient/Hessian are ~1e6 / ~1e12, and the inner joint-Newton
        // crawls and never reaches the data-scale baseline within the cycle
        // budget — every outer seed is rejected and the fit hard-fails.
        //
        // Condition the COLD START by building the baseline OFFSET from a fixed,
        // data-seeded Weibull (scale = mean positive exit time, shape = 1) instead
        // of the zero-derivative Linear baseline, but ONLY for the offset: the
        // outer `baseline_cfg.target` stays `Linear`, so the
        // `baseline_cfg.target != Linear` optimize gate
        // (`optimize_survival_baseline_config*`) never fires and no baseline-shape
        // search is introduced. With shape = 1 the Weibull baseline-hazard
        // derivative is `1/age_exit` (the natural data hazard scale), so the seed
        // starts with `qd1` at O(1/T) interior — barrier gradient O(10-10²),
        // comparable to the marginal/logslope blocks — and `β_time ≈ 0`. This
        // changes only the STARTING point / offset split: the I-spline still learns
        // the data-driven deviation from this parametric baseline (the converged
        // fitted hazard is the same flexible family), so the fix is a pure
        // preconditioning of the cold start. Gated to MarginalSlope with a Linear
        // target so every other Linear-baseline survival path is byte-unchanged.
        let conditioning_cfg;
        let offset_cfg = if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope
            && baseline_cfg.target == SurvivalBaselineTarget::Linear
        {
            let scale =
                crate::families::survival_construction::positive_survival_time_seed(age_exit);
            conditioning_cfg = crate::families::survival_construction::SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(scale),
                shape: Some(1.0),
                rate: None,
                makeham: None,
            };
            &conditioning_cfg
        } else {
            baseline_cfg
        };
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                age_entry,
                age_exit,
                offset_cfg,
                likelihood_mode,
                inverse_link,
            )?;
        let n = age_entry.len();
        (
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::zeros(n),
        )
    };
    add_survival_time_derivative_guard_offset(
        age_entry,
        age_exit,
        time_anchor,
        derivative_guard,
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative_exit = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(wiggle) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative_exit,
            wiggle.ncols,
        );
        for (idx, penalty) in wiggle.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + wiggle.ncols, p_base + wiggle.ncols));
            embedded
                .slice_mut(s![
                    p_base..p_base + wiggle.ncols,
                    p_base..p_base + wiggle.ncols
                ])
                .assign(penalty);
            time_penalties.push(embedded);
            time_nullspace_dims.push(wiggle.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: wiggle.knots.clone(),
            degree: wiggle.degree,
            ncols: wiggle.ncols,
        });
    }
    Ok(PreparedSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative_exit,
        time_penalties,
        time_nullspace_dims,
        timewiggle_build,
        timewiggle_block,
    })
}

fn resolve_continuous_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: &str,
    role: &str,
) -> Result<Array1<f64>, WorkflowError> {
    let col_idx = resolve_role_col(col_map, column_name, role)?;
    let values = data.values.column(col_idx).to_owned();
    for (row_idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "{role} column '{column_name}' contains non-finite value at row {row_idx}: {value}"
                ),
            });
        }
    }
    Ok(values)
}

pub fn resolve_offset_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::zeros(data.values.nrows()));
    };
    resolve_continuous_column(data, col_map, column_name, "offset")
}

pub fn resolve_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::ones(data.values.nrows()));
    };
    let values = resolve_continuous_column(data, col_map, column_name, "weights")?;
    for (row_idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "weights column '{column_name}' must be non-negative; found {value} at row {row_idx}"
                ),
            });
        }
    }
    Ok(values)
}

const MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR: f64 = 1e-12;

fn validate_bernoulli_marginal_slope_z_column_variance(
    z_column: &str,
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<(), WorkflowError> {
    if z.len() != weights.len() {
        return Err(WorkflowError::SchemaMismatch {
            reason: format!(
                "z_column '{z_column}' length mismatch for bernoulli-marginal-slope: z={}, weights={}",
                z.len(),
                weights.len()
            ),
        });
    }
    let n = z.len();
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "z_column '{z_column}' cannot be weighted for bernoulli-marginal-slope because the fit data have non-positive or non-finite total weight"
            ),
        });
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let weighted_sd = var.sqrt();
    if weighted_sd.is_finite() && weighted_sd > MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR {
        return Ok(());
    }

    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() <= MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR);
    let unique_count = sorted.len();
    let value_summary = match sorted.as_slice() {
        [] => "no observed finite values".to_string(),
        [only] => format!("all {n} values ~= {only:.6}"),
        [first, second] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}")
        }
        [first, second, ..] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}, ...")
        }
    };
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "z_column '{z_column}' has zero weighted variance on the fit data ({value_summary}; weighted_sd={weighted_sd:.6e}, n={n}); bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score. Check the score column and fit population."
        ),
    })
}

#[derive(Clone)]
enum LatentInitSpec {
    Pca,
    Random,
    Explicit(Array2<f64>),
}

#[derive(Clone)]
struct LatentAuxPriorSpec {
    u: Array2<f64>,
    family: AuxPriorFamily,
    strength: AuxPriorStrength,
}

#[derive(Clone)]
struct LatentDimSelectionSpec {
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentAuxOutcomeSpec {
    family: crate::terms::behavioral_head::AuxOutcomeFamily,
    /// Behavioral labels, length `n`. Binomial: 0/1; Multinomial: class index.
    y: Array1<f64>,
    /// Optional per-row head weight (semi-supervised); `None` ⇒ all rows
    /// labeled with unit weight. `0.0` on a row excludes it from the head
    /// channel — the missing-label seam.
    row_weights: Option<Array1<f64>>,
    /// ARD log-precision seed composed with the head (length `d`).
    init_log_precision: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LatentManifoldSpec {
    manifold: LatentManifold,
    auto: bool,
}

#[derive(Clone)]
struct LatentSpec {
    target: String,
    n: usize,
    d: usize,
    init: LatentInitSpec,
    manifold: LatentManifoldSpec,
    retraction_registry: LatentRetractionRegistry,
    aux_prior: Option<LatentAuxPriorSpec>,
    dim_selection: Option<LatentDimSelectionSpec>,
    aux_outcome: Option<LatentAuxOutcomeSpec>,
    explicit_none_mode: bool,
}

fn json_array2(value: &JsonValue, context: &str) -> Result<Array2<f64>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a two-dimensional numeric array"))?;
    let n = rows.len();
    let first = rows
        .first()
        .and_then(|row| row.as_array())
        .ok_or_else(|| format!("{context} must contain array rows"))?;
    let d = first.len();
    let mut out = Array2::<f64>::zeros((n, d));
    for (i, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_array()
            .ok_or_else(|| format!("{context} row {i} must be an array"))?;
        if row.len() != d {
            return Err(format!(
                "{context} row {i} has length {}, expected {d}",
                row.len()
            ));
        }
        for (j, cell) in row.iter().enumerate() {
            let value = cell
                .as_f64()
                .ok_or_else(|| format!("{context}[{i}][{j}] must be a finite number"))?;
            if !value.is_finite() {
                return Err(format!("{context}[{i}][{j}] must be finite"));
            }
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn json_array1(value: &JsonValue, context: &str) -> Result<Array1<f64>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context} must be a numeric array"))?;
    let mut out = Array1::<f64>::zeros(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}[{idx}] must be finite"));
        }
        out[idx] = value;
    }
    Ok(out)
}

fn parse_latent_manifold(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentManifoldSpec, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    };
    if value
        .as_str()
        .is_some_and(|s| s.eq_ignore_ascii_case("auto"))
    {
        return Ok(LatentManifoldSpec {
            manifold: LatentManifold::Euclidean,
            auto: true,
        });
    }
    let parse_named = |name: &str| -> Result<LatentManifold, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(LatentManifold::Euclidean),
            "circle" | "s1" | "periodic" => {
                let radians = LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                };
                if d == 1 {
                    Ok(radians)
                } else {
                    Ok(LatentManifold::Product(
                        (0..d).map(|_| radians.clone()).collect(),
                    ))
                }
            }
            "sphere" | "sn" => Ok(LatentManifold::Sphere { dim: d }),
            "torus" => Ok(LatentManifold::Product(
                (0..d)
                    .map(|_| LatentManifold::Circle {
                        period: std::f64::consts::TAU,
                    })
                    .collect(),
            )),
            "cylinder" => {
                if d < 2 {
                    return Err(format!("{context}='cylinder' requires d >= 2"));
                }
                let mut parts = Vec::with_capacity(d);
                parts.push(LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                });
                for _ in 1..d {
                    parts.push(LatentManifold::Euclidean);
                }
                Ok(LatentManifold::Product(parts))
            }
            other => Err(format!(
                "{context} must be 'auto', 'euclidean', 'circle', 'sphere', 'torus', or 'cylinder'; got '{other}'"
            )),
        }
    };
    let manifold = if let Some(name) = value.as_str() {
        parse_named(name)?
    } else if let Some(obj) = value.as_object() {
        let kind = obj
            .get("type")
            .or_else(|| obj.get("kind"))
            .and_then(JsonValue::as_str)
            .unwrap_or("euclidean");
        match kind.to_ascii_lowercase().as_str() {
            "auto" => {
                return Ok(LatentManifoldSpec {
                    manifold: LatentManifold::Euclidean,
                    auto: true,
                });
            }
            "interval" => {
                let lo = obj
                    .get("lo")
                    .or_else(|| obj.get("min"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.lo is required for interval"))?;
                let hi = obj
                    .get("hi")
                    .or_else(|| obj.get("max"))
                    .and_then(JsonValue::as_f64)
                    .ok_or_else(|| format!("{context}.hi is required for interval"))?;
                if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                    return Err(format!("{context} interval requires finite lo < hi"));
                }
                LatentManifold::Interval { lo, hi }
            }
            other => parse_named(other)?,
        }
    } else if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts
                .push(parse_latent_manifold(Some(item), 1, &format!("{context}[{idx}]"))?.manifold);
        }
        LatentManifold::Product(parts)
    } else {
        return Err(format!(
            "{context} must be a string, object, or product array"
        ));
    };
    if manifold.ambient_dim(d) != d {
        return Err(format!(
            "{context} ambient dimension {} does not match latent d={d}",
            manifold.ambient_dim(d)
        ));
    }
    Ok(LatentManifoldSpec {
        manifold,
        auto: false,
    })
}

fn parse_retraction_kind(
    value: &JsonValue,
    fallback_dim: usize,
    context: &str,
) -> Result<RetractionKind, String> {
    let parse_named = |name: &str| -> Result<RetractionKind, String> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "r" | "real" => Ok(RetractionKind::euclidean(fallback_dim)),
            "circle" | "s1" | "periodic" => {
                if fallback_dim == 1 {
                    Ok(RetractionKind::Circle)
                } else {
                    Ok(RetractionKind::Product(ProductRetraction {
                        parts: (0..fallback_dim).map(|_| RetractionKind::Circle).collect(),
                    }))
                }
            }
            "sphere" | "sn" => Ok(RetractionKind::Sphere { dim: fallback_dim }),
            other => Err(format!(
                "{context} must be 'euclidean', 'circle', 'sphere', or a product; got '{other}'"
            )),
        }
    };
    if let Some(name) = value.as_str() {
        return parse_named(name);
    }
    if let Some(items) = value.as_array() {
        let mut parts = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            parts.push(parse_retraction_kind(
                item,
                1,
                &format!("{context}[{idx}]"),
            )?);
        }
        return Ok(RetractionKind::Product(ProductRetraction { parts }));
    }
    let obj = value
        .as_object()
        .ok_or_else(|| format!("{context} must be a string, object, or product array"))?;
    let kind = obj
        .get("type")
        .or_else(|| obj.get("kind"))
        .and_then(JsonValue::as_str)
        .unwrap_or("euclidean");
    match kind.to_ascii_lowercase().as_str() {
        "euclidean" | "r" | "real" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::euclidean(dim))
        }
        "circle" | "s1" | "periodic" => Ok(RetractionKind::Circle),
        "sphere" | "sn" => {
            let dim = obj
                .get("dim")
                .or_else(|| obj.get("d"))
                .and_then(JsonValue::as_u64)
                .map_or(fallback_dim, |value| value as usize);
            if dim == 0 {
                return Err(format!("{context}.dim must be positive"));
            }
            Ok(RetractionKind::Sphere { dim })
        }
        "product" => {
            let items = obj
                .get("parts")
                .or_else(|| obj.get("components"))
                .and_then(JsonValue::as_array)
                .ok_or_else(|| format!("{context}.parts is required for product retraction"))?;
            let mut parts = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                parts.push(parse_retraction_kind(
                    item,
                    1,
                    &format!("{context}.parts[{idx}]"),
                )?);
            }
            Ok(RetractionKind::Product(ProductRetraction { parts }))
        }
        other => parse_named(other),
    }
}

fn parse_latent_retraction(
    value: Option<&JsonValue>,
    d: usize,
    context: &str,
) -> Result<LatentRetractionRegistry, String> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(LatentRetractionRegistry::all_euclidean());
    };
    let kind = parse_retraction_kind(value, d, context)?;
    let registry = LatentRetractionRegistry::new(kind);
    registry.validate_dim(d, context)?;
    Ok(registry)
}

fn parse_latent_specs(payload: Option<&JsonValue>) -> Result<Vec<LatentSpec>, String> {
    let Some(payload) = payload.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = payload
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut specs = Vec::with_capacity(map.len());
    for (key, raw) in map {
        let obj = raw
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let target = obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or(key)
            .to_string();
        let n = obj
            .get("n")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].n is required"))? as usize;
        let d = obj
            .get("d")
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("latents['{key}'].d is required"))? as usize;
        if n == 0 || d == 0 {
            return Err(format!("latents['{key}'] requires positive n and d"));
        }
        let manifold = parse_latent_manifold(
            obj.get("manifold"),
            d,
            &format!("latents['{key}'].manifold"),
        )?;
        let retraction_registry = parse_latent_retraction(
            obj.get("retraction"),
            d,
            &format!("latents['{key}'].retraction"),
        )?;
        let init = match obj.get("init") {
            None => LatentInitSpec::Pca,
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("pca")) =>
            {
                LatentInitSpec::Pca
            }
            Some(value)
                if value
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("random")) =>
            {
                LatentInitSpec::Random
            }
            Some(value) => {
                LatentInitSpec::Explicit(json_array2(value, &format!("latents['{key}'].init"))?)
            }
        };
        let aux_prior = match obj.get("aux_prior").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                let aux = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_prior must be an object"))?;
                let u = json_array2(
                    aux.get("u")
                        .ok_or_else(|| format!("latents['{key}'].aux_prior.u is required"))?,
                    &format!("latents['{key}'].aux_prior.u"),
                )?;
                let family = match aux
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("ridge")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "ridge" => AuxPriorFamily::Ridge,
                    "linear" => AuxPriorFamily::Linear,
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_prior.family must be 'ridge' or 'linear', got '{other}'"
                        ));
                    }
                };
                let strength = match aux.get("strength") {
                    None => AuxPriorStrength::Fixed(1.0),
                    Some(value)
                        if value
                            .as_str()
                            .is_some_and(|s| s.eq_ignore_ascii_case("auto")) =>
                    {
                        AuxPriorStrength::Auto
                    }
                    Some(value) => {
                        let mu = value.as_f64().ok_or_else(|| {
                            format!(
                                "latents['{key}'].aux_prior.strength must be positive or 'auto'"
                            )
                        })?;
                        if !mu.is_finite() || mu <= 0.0 {
                            return Err(format!(
                                "latents['{key}'].aux_prior.strength must be positive"
                            ));
                        }
                        AuxPriorStrength::Fixed(mu)
                    }
                };
                Some(LatentAuxPriorSpec {
                    u,
                    family,
                    strength,
                })
            }
        };
        let dim_selection = match obj.get("dim_selection") {
            None | Some(JsonValue::Bool(false)) => None,
            Some(JsonValue::Bool(true)) => Some(LatentDimSelectionSpec {
                init_log_precision: None,
            }),
            Some(value) => {
                let dim = value.as_object().ok_or_else(|| {
                    format!("latents['{key}'].dim_selection must be a bool or object")
                })?;
                let init_log_precision = dim
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].dim_selection.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentDimSelectionSpec { init_log_precision })
            }
        };
        let aux_outcome = match obj.get("aux_outcome").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                use crate::terms::behavioral_head::AuxOutcomeFamily;
                let ao = value
                    .as_object()
                    .ok_or_else(|| format!("latents['{key}'].aux_outcome must be an object"))?;
                let family = match ao
                    .get("family")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("binomial")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "binomial" => AuxOutcomeFamily::Binomial,
                    "multinomial" => {
                        let n_classes = ao
                            .get("n_classes")
                            .and_then(JsonValue::as_u64)
                            .ok_or_else(|| {
                                format!(
                                    "latents['{key}'].aux_outcome.n_classes is required for multinomial"
                                )
                            })? as usize;
                        AuxOutcomeFamily::Multinomial { n_classes }
                    }
                    other => {
                        return Err(format!(
                            "latents['{key}'].aux_outcome.family must be 'binomial' or 'multinomial', got '{other}'"
                        ));
                    }
                };
                let y = json_array1(
                    ao.get("y")
                        .ok_or_else(|| format!("latents['{key}'].aux_outcome.y is required"))?,
                    &format!("latents['{key}'].aux_outcome.y"),
                )?;
                if y.len() != n {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.y has length {}, expected n = {n}",
                        y.len()
                    ));
                }
                let row_weights = ao
                    .get("row_weights")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        json_array1(value, &format!("latents['{key}'].aux_outcome.row_weights"))
                    })
                    .transpose()?;
                if let Some(w) = row_weights.as_ref()
                    && w.len() != n
                {
                    return Err(format!(
                        "latents['{key}'].aux_outcome.row_weights has length {}, expected n = {n}",
                        w.len()
                    ));
                }
                let init_log_precision = ao
                    .get("init_log_precision")
                    .map(|value| {
                        json_array1(
                            value,
                            &format!("latents['{key}'].aux_outcome.init_log_precision"),
                        )
                    })
                    .transpose()?;
                Some(LatentAuxOutcomeSpec {
                    family,
                    y,
                    row_weights,
                    init_log_precision,
                })
            }
        };
        if dim_selection.is_some() && aux_prior.is_none() && aux_outcome.is_none() {
            return Err(format!(
                "latents['{key}'] uses dim_selection without aux_prior or aux_outcome; ARD alone is not an identifiable latent-coordinate gauge"
            ));
        }
        if aux_outcome.is_some() && aux_prior.is_some() {
            return Err(format!(
                "latents['{key}'] specifies both aux_prior and aux_outcome; the auxiliary signal is either a prior (gauge-pin covariate) or a modeled outcome (behavioral head), not both"
            ));
        }
        let explicit_none_mode = obj
            .get("id_mode")
            .or_else(|| obj.get("mode"))
            .and_then(JsonValue::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("none"));
        if aux_prior.is_none()
            && dim_selection.is_none()
            && aux_outcome.is_none()
            && !explicit_none_mode
        {
            return Err(format!(
                "latents['{key}'] requires aux_prior or aux_outcome for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied"
            ));
        }
        specs.push(LatentSpec {
            target,
            n,
            d,
            init,
            manifold,
            retraction_registry,
            aux_prior,
            dim_selection,
            aux_outcome,
            explicit_none_mode,
        });
    }
    Ok(specs)
}

fn deterministic_unit(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn initial_latent_matrix(spec: &LatentSpec, y: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    match &spec.init {
        LatentInitSpec::Explicit(matrix) => {
            if matrix.nrows() != spec.n || matrix.ncols() != spec.d {
                return Err(format!(
                    "latent '{}' explicit init has shape {}x{}, expected {}x{}",
                    spec.target,
                    matrix.nrows(),
                    matrix.ncols(),
                    spec.n,
                    spec.d
                ));
            }
            Ok(matrix.clone())
        }
        LatentInitSpec::Random => {
            let mut seed = 0x9E3779B97F4A7C15_u64 ^ ((spec.n as u64) << 32) ^ spec.d as u64;
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            for value in out.iter_mut() {
                *value = deterministic_unit(&mut seed);
            }
            Ok(out)
        }
        LatentInitSpec::Pca => {
            let mut out = Array2::<f64>::zeros((spec.n, spec.d));
            let mean = y.iter().sum::<f64>() / y.len().max(1) as f64;
            let var = y
                .iter()
                .map(|v| {
                    let centered = *v - mean;
                    centered * centered
                })
                .sum::<f64>()
                / y.len().max(1) as f64;
            let sd = var.sqrt().max(1e-12);
            for n in 0..spec.n {
                out[[n, 0]] = (y[n] - mean) / sd;
            }
            if spec.d > 1 {
                let mut seed = 0xD1B54A32D192ED03_u64 ^ ((spec.n as u64) << 16) ^ spec.d as u64;
                for n in 0..spec.n {
                    for axis in 1..spec.d {
                        out[[n, axis]] = deterministic_unit(&mut seed) - 0.5;
                    }
                }
            }
            Ok(out)
        }
    }
}

fn latent_id_mode(spec: &LatentSpec) -> Result<LatentIdMode, String> {
    if let Some(ao) = spec.aux_outcome.as_ref() {
        use crate::terms::behavioral_head::BehavioralHead;
        if let Some(init) = ao.init_log_precision.as_ref()
            && init.len() != spec.d
        {
            return Err(format!(
                "latent '{}' aux_outcome.init_log_precision has length {}, expected {}",
                spec.target,
                init.len(),
                spec.d
            ));
        }
        let head = match ao.row_weights.as_ref() {
            Some(w) => BehavioralHead::new(ao.family, ao.y.clone(), w.clone()),
            None => BehavioralHead::fully_supervised(ao.family, ao.y.clone()),
        }
        .map_err(|e| format!("latent '{}' aux_outcome head: {e}", spec.target))?;
        return Ok(LatentIdMode::AuxOutcome {
            head,
            init_log_precision: ao.init_log_precision.clone(),
        });
    }
    match (&spec.aux_prior, &spec.dim_selection) {
        (Some(aux), Some(dim)) => {
            if let Some(init) = dim.init_log_precision.as_ref()
                && init.len() != spec.d
            {
                return Err(format!(
                    "latent '{}' dim_selection.init_log_precision has length {}, expected {}",
                    spec.target,
                    init.len(),
                    spec.d
                ));
            }
            Ok(LatentIdMode::AuxPriorDimSelection {
                u: aux.u.clone(),
                family: aux.family,
                strength: aux.strength,
                init_log_precision: dim.init_log_precision.clone(),
            })
        }
        (Some(aux), None) => Ok(LatentIdMode::AuxPrior {
            u: aux.u.clone(),
            family: aux.family,
            strength: aux.strength,
        }),
        (None, None) if spec.explicit_none_mode => Ok(LatentIdMode::None),
        (None, None) => Err(format!(
            "latent '{}' requires aux_prior for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied",
            spec.target
        )),
        (None, Some(_)) => Err(format!(
            "latent '{}' dim_selection requires aux_prior for identifiability",
            spec.target
        )),
    }
}

fn prepare_standard_latent_coord(
    parsed: &ParsedFormula,
    data: &Dataset,
    y: ArrayView1<'_, f64>,
    config: &FitConfig,
) -> Result<Option<(Dataset, ParsedFormula, StandardLatentCoordConfig)>, String> {
    let specs = parse_latent_specs(config.latents.as_ref())?;
    let analytic_penalties = descriptors::build_analytic_penalty_registry_from_descriptors(
        config.latents.as_ref(),
        config.analytic_penalties.as_ref(),
    )?;
    if config.topology_auto_selector.is_some() && specs.is_empty() {
        return Err(
            "TopologyAutoSelector requires a Smooth with latent coords; pass latents={...}"
                .to_string(),
        );
    }
    if specs.is_empty() {
        return Ok(None);
    }
    if specs.len() != 1 {
        return Err(
            "standard latent-coordinate REML currently accepts exactly one latent smooth term"
                .to_string(),
        );
    }
    let spec = specs.into_iter().next().unwrap();
    if let Some(selector) = config.topology_auto_selector.as_ref()
        && let Some(requested) = selector.latent.as_ref()
        && requested != &spec.target
    {
        return Err(format!(
            "TopologyAutoSelector requested latent {requested:?}, but the formula path materialized latent {:?}",
            spec.target
        ));
    }
    if spec.n != data.values.nrows() || spec.n != y.len() {
        return Err(format!(
            "latent '{}' row count {} does not match data rows {}",
            spec.target,
            spec.n,
            data.values.nrows()
        ));
    }
    if let Some(aux) = spec.aux_prior.as_ref()
        && aux.u.nrows() != spec.n
    {
        return Err(format!(
            "latent '{}' aux_prior.u has {} rows, expected {}",
            spec.target,
            aux.u.nrows(),
            spec.n
        ));
    }

    let matrix = initial_latent_matrix(&spec, y)?;
    let id_mode = latent_id_mode(&spec)?;
    let latent_values = Arc::new(LatentCoordValues::from_matrix_with_manifold_and_retraction(
        matrix.view(),
        id_mode,
        spec.manifold.manifold.clone(),
        spec.retraction_registry.clone(),
    ));

    let base_cols = data.values.ncols();
    let mut values = Array2::<f64>::zeros((data.values.nrows(), base_cols + spec.d));
    values.slice_mut(s![.., ..base_cols]).assign(&data.values);
    let mut headers = data.headers.clone();
    let mut columns = data.schema.columns.clone();
    let mut column_kinds = data.column_kinds.clone();
    let mut synthetic_vars = Vec::with_capacity(spec.d);
    let mut feature_cols = Vec::with_capacity(spec.d);
    for axis in 0..spec.d {
        let name = format!("{}__latent{}", spec.target, axis);
        let col = base_cols + axis;
        values.column_mut(col).assign(&matrix.column(axis));
        headers.push(name.clone());
        columns.push(SchemaColumn {
            name: name.clone(),
            kind: ColumnKindTag::Continuous,
            levels: Vec::new(),
        });
        column_kinds.push(ColumnKindTag::Continuous);
        synthetic_vars.push(name);
        feature_cols.push(col);
    }
    let augmented = Dataset {
        headers,
        values,
        schema: DataSchema { columns },
        column_kinds,
    };

    let mut rewritten = parsed.clone();
    let mut matched = false;
    for term in &mut rewritten.terms {
        if let ParsedTerm::Smooth { vars, .. } = term
            && vars.len() == 1
            && vars[0] == spec.target
        {
            *vars = synthetic_vars.clone();
            matched = true;
        }
    }
    if !matched {
        return Err(format!(
            "latents provided '{}' but no formula smooth term s({}, ...) was found",
            spec.target, spec.target
        ));
    }

    Ok(Some((
        augmented,
        rewritten,
        StandardLatentCoordConfig {
            values: latent_values,
            term_index: crate::types::SmoothTermIdx::placeholder(),
            feature_cols,
            manifold: spec.manifold.manifold,
            manifold_auto: spec.manifold.auto,
            retraction_registry: spec.retraction_registry,
            analytic_penalties: (!analytic_penalties.penalties.is_empty())
                .then(|| Arc::new(analytic_penalties)),
        },
    )))
}

fn smooth_basis_feature_cols_for_latent(
    basis: &crate::smooth::SmoothBasisSpec,
) -> Option<Vec<usize>> {
    match basis {
        crate::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(vec![*feature_col]),
        crate::smooth::SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Sphere { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Matern { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Duchon { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Pca { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            Some(feature_cols.clone())
        }
        crate::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_basis_feature_cols_for_latent(smooth)
        }
        crate::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | crate::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_feature_cols_for_latent(inner)
        }
        crate::smooth::SmoothBasisSpec::FactorSmooth { .. } => None,
    }
}

fn natural_latent_manifold_for_basis(
    basis: &crate::smooth::SmoothBasisSpec,
    d: usize,
) -> LatentManifold {
    match basis {
        crate::smooth::SmoothBasisSpec::BSpline1D { spec, .. } => {
            if let crate::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                &spec.knotspec
            {
                LatentManifold::Circle {
                    period: data_range.1 - data_range.0,
                }
            } else {
                LatentManifold::Euclidean
            }
        }
        crate::smooth::SmoothBasisSpec::Sphere { .. } => LatentManifold::Sphere { dim: d },
        crate::smooth::SmoothBasisSpec::Duchon { spec, .. }
            if spec.periodic.is_some() && d == 1 =>
        {
            let period = spec
                .periodic
                .as_ref()
                .and_then(|v| v.first().copied().flatten())
                .unwrap_or(std::f64::consts::TAU);
            LatentManifold::Circle { period }
        }
        crate::smooth::SmoothBasisSpec::TensorBSpline { spec, .. } => {
            let parts: Vec<LatentManifold> = spec
                .marginalspecs
                .iter()
                .map(|margin| {
                    if let crate::basis::BSplineKnotSpec::PeriodicUniform { data_range, .. } =
                        &margin.knotspec
                    {
                        LatentManifold::Circle {
                            period: data_range.1 - data_range.0,
                        }
                    } else {
                        LatentManifold::Euclidean
                    }
                })
                .collect();
            if parts.iter().all(|part| part.is_euclidean()) {
                LatentManifold::Euclidean
            } else {
                LatentManifold::Product(parts)
            }
        }
        crate::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            natural_latent_manifold_for_basis(smooth, d)
        }
        crate::smooth::SmoothBasisSpec::ByVariable { inner, .. }
        | crate::smooth::SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            natural_latent_manifold_for_basis(inner, d)
        }
        crate::smooth::SmoothBasisSpec::ThinPlate { .. }
        // ConstantCurvature: the chart coordinates are Euclidean-valued (any
        // finite point for κ ≥ 0; the latent optimizer's chart-validity is the
        // term's own concern), so the latent retraction stays Euclidean. A
        // κ-aware latent seed/retraction is part of the later ψ-channel stage.
        | crate::smooth::SmoothBasisSpec::ConstantCurvature { .. }
        | crate::smooth::SmoothBasisSpec::Matern { .. }
        | crate::smooth::SmoothBasisSpec::Duchon { .. }
        | crate::smooth::SmoothBasisSpec::Pca { .. }
        | crate::smooth::SmoothBasisSpec::FactorSmooth { .. } => LatentManifold::Euclidean,
    }
}

fn materialize_standard<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if config.noise_offset_column.is_some() {
        return Err(
            "noise_offset_column requires a location-scale model with noise_formula"
                .to_string()
                .into(),
        );
    }
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation (#335 Gamma requires y > 0;
    // #337 Poisson/NegativeBinomial require y ≥ 0; mirrors the Beta
    // (0,1)-support check in the external-design GLM path). The family
    // itself owns the check — see `ResponseFamily::validate_response_support`
    // — so adding a new family that constrains its support is a single edit
    // on the type, not a coordinated update across every materializer.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331 all-0/all-1 Bernoulli,
    // #332 near-constant Gaussian). Symmetric to validate_response_support —
    // each `ResponseFamily` variant owns its own degeneracy classifier, the
    // workflow only forwards the column name.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let latent_prepared = prepare_standard_latent_coord(parsed, data, y.view(), config)?;
    let (latent_dataset, latent_parsed, mut latent_coord) = match latent_prepared {
        Some((dataset, parsed, coord)) => (Some(dataset), Some(parsed), Some(coord)),
        None => (None, None, None),
    };
    let term_data = latent_dataset.as_ref().unwrap_or(data);
    let term_parsed = latent_parsed.as_ref().unwrap_or(parsed);
    let term_col_map = term_data.column_map();

    let policy =
        resolved_resource_policy(config, term_data, crate::resource::ProblemHints::default());
    let spec = build_termspec_with_geometry_and_overrides(
        &term_parsed.terms,
        term_data,
        &term_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;

    // Sample size vs basis-rank gate (#309). Each smooth basis answers
    // `min_sample_rows()` for itself; this helper just sums and compares.
    // Runs *after* `build_termspec_with_geometry_and_overrides` so the lower bound is
    // computed on the fully resolved basis spec (e.g. tensor-product columns,
    // knot counts inferred at materialization time).
    check_smooth_capacity(&spec, y.len(), &parsed.response)?;
    if let Some(coord) = latent_coord.as_mut() {
        let resolved_idx = spec
            .smooth_terms
            .iter()
            .position(|term| {
                smooth_basis_feature_cols_for_latent(&term.basis)
                    .is_some_and(|cols| cols == coord.feature_cols)
            })
            .ok_or_else(|| {
                "latent-coordinate smooth term disappeared during formula materialization"
                    .to_string()
            })?;
        coord.term_index = crate::types::SmoothTermIdx::new(resolved_idx);
        if coord.manifold_auto {
            let inferred = natural_latent_manifold_for_basis(
                &spec.smooth_terms[coord.term_index.get()].basis,
                coord.feature_cols.len(),
            );
            coord.manifold = inferred.clone();
            coord.values = Arc::new(coord.values.with_manifold(inferred));
        }
    }

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let latent_cloglog = if family.is_latent_cloglog() {
        let sigma = match config.frailty.clone().unwrap_or(FrailtySpec::None) {
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: crate::families::lognormal_kernel::HazardLoading::Full,
            } => sigma,
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(_),
                loading,
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason: format!(
                        "latent-cloglog-binomial requires HazardLoading::Full, got {loading:?}"
                    ),
                }
                .into());
            }
            FrailtySpec::HazardMultiplier {
                sigma_fixed: None, ..
            } => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial currently requires a fixed hazard-multiplier sigma"
                            .to_string(),
                }
                .into());
            }
            FrailtySpec::GaussianShift { .. } => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "latent-cloglog-binomial does not support GaussianShift frailty"
                        .to_string(),
                }
                .into());
            }
            FrailtySpec::None => {
                return Err(WorkflowError::MissingDependency {
                    reason:
                        "latent-cloglog-binomial requires config.frailty=HazardMultiplier with a fixed sigma"
                            .to_string(),
                }
                .into());
            }
        };
        Some(
            LatentCLogLogState::new(sigma)
                .map_err(|e| format!("invalid latent_cloglog state: {e}"))?,
        )
    } else {
        if config.frailty.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "config.frailty is not supported for standard family {:?}; use a frailty-aware family instead",
                    family
                ),
            }
            .into());
        }
        None
    };
    let options = FitOptions {
        latent_cloglog,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: config.outer_max_iter.unwrap_or(200),
        // Outer REML/LAML smoothing-selection tolerance. The outer convergence
        // test (`outer_gradient_tolerance`) uses a `rel_cost` criterion whose
        // effective projected-gradient threshold is ≈ `tol · (1 + |V(ρ)|)`. The
        // LAML cost grows like O(n), so at the old `1e-7` the effective gradient
        // threshold was ≈ `1e-7 · |V|` ≈ 1e-4 for a typical fit — far too coarse
        // to *resolve* the smoothing parameter: the descent halted while λ̂ could
        // still move several percent. That under-resolution is benign for a
        // single fit but breaks an exact invariance — for a fixed-dispersion
        // family a uniform prior weight `w = c` is exact `c`-fold replication, so
        // the two encodings share a byte-identical LAML surface and an identical
        // optimum, yet their (replication-equal) surfaces carry O(1e-7)
        // floating-point differences AWAY from the optimum. With the coarse
        // threshold the descent stopped at those encoding-dependent points,
        // systematically over-smoothing the weighted encoding (gam#893; up to a
        // ~22× λ ratio across seeds). Tightening to `1e-10` (effective gradient
        // threshold ≈ 1e-7, ~100× below the FP-noise floor) drives both
        // encodings to the shared optimum, restoring `w=c ⇔ c-fold replication`
        // in smoothing selection to optimiser precision. Max-iter is handled
        // best-effort (the optimiser returns its best ρ on budget exhaustion, it
        // does not hard-fail), so a harder problem that cannot reach 1e-10 in
        // `outer_max_iter` is no worse off than before — just better-resolved
        // when it can.
        tol: 1e-10,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: config.firth,
        adaptive_regularization: standard_adaptive_regularization_options(config),
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    let wiggle = effective_linkwiggle.as_ref().and_then(|cfg| {
        if !family.is_binomial() {
            return None;
        }
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                // linkwiggle is gated by `linkname_supports_joint_wiggle` which
                // rejects Sas / BetaLogistic upstream, so reaching this arm
                // means the gate was bypassed.
                Err(_) => return None,
            },
            None => {
                if let Some(state) = latent_cloglog {
                    InverseLink::LatentCLogLog(state)
                } else {
                    InverseLink::Standard(StandardLink::Logit)
                }
            }
        };
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders.clone(),
                double_penalty: cfg.double_penalty,
            },
            // The second-stage refit options live inside the wiggle config so
            // the pilot can't be configured without them (see
            // `StandardBinomialWiggleConfig` doc + #320). Magic-by-default:
            // no caller-supplied options are required for the Python /
            // formula-DSL path.
            refit_options: BlockwiseFitOptions::default(),
        })
    });

    Ok(MaterializedModel {
        request: FitRequest::Standard(StandardFitRequest {
            data: term_data.values.clone(),
            y,
            weights,
            offset,
            spec,
            family,
            options,
            kappa_options,
            wiggle,
            coefficient_groups: config.coefficient_groups.clone(),
            penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
            latent_coord,
            _marker: std::marker::PhantomData,
        }),
        inference_notes,
    })
}

fn materialize_bernoulli_marginal_slope<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();

    if !is_binary_response(y.view()) {
        return Err(WorkflowError::SchemaMismatch {
            reason: "Bernoulli marginal-slope requires a binary {0,1} response".to_string(),
        }
        .into());
    }
    if config.noise_formula.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope cannot also use noise_formula".to_string(),
        }
        .into());
    }

    let logslope_formula = config
        .logslope_formula
        .as_deref()
        .ok_or_else(|| "Bernoulli marginal-slope requires logslope_formula".to_string())?;
    // `z_column` is OPTIONAL when a CTN Stage-1 recipe is present: the calibrated
    // chain produces `z` out-of-fold from the cross-fitted CTN, so there is no
    // raw dose column to read (and no throwaway pre-fit column — that round-trip
    // is what the no-slop cutover removes, #461). Without a recipe, the primitive
    // standalone marginal-slope still requires a raw `z_column` dose.
    let z_column = config.z_column.as_deref();
    if z_column.is_none() && config.ctn_stage1.is_none() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope requires z_column (or a CTN Stage-1 recipe via \
                     ctn_stage1, which produces z by cross-fitting)"
                .to_string(),
        });
    }

    let (_, parsed_logslope) =
        parse_matching_auxiliary_formula(logslope_formula, &parsed.response, "logslope_formula")?;
    if parsed_logslope.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported inside logslope_formula".to_string(),
        }
        .into());
    }
    if let Some(z_column) = z_column {
        validate_marginal_slope_z_column_exclusion(
            parsed,
            &parsed_logslope,
            z_column,
            "Bernoulli marginal-slope",
            "logslope_formula",
        )?;
    }

    let mut inference_notes = Vec::new();
    // Bernoulli marginal-slope: structurally operator-only at large scale, so
    // flip the hint regardless of n to keep dense fallbacks blocked.
    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            marginal_slope_large_scale_active: true,
        },
    );
    // Alias `z` to the dose column only when a raw z_column is supplied; with a
    // CTN Stage-1 chain there is no dose column and the formulas reference only
    // the x covariates.
    let aliased_col_map = match z_column {
        Some(z_column) => column_map_with_alias(col_map, "z", z_column),
        None => col_map.clone(),
    };
    let mut marginalspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut marginalspec,
        data,
        "bernoulli marginal-slope marginal formula",
        &mut inference_notes,
    )?;
    let mut logslopespec = build_termspec_with_geometry_and_overrides(
        &parsed_logslope.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut logslopespec,
        data,
        "bernoulli marginal-slope logslope_formula",
        &mut inference_notes,
    )?;
    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let marginal_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let logslope_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let routing = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
    )?;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when a CTN
    // Stage-1 recipe is present (design §5). Cross-fitting yields out-of-fold `z`
    // (the calibrated dose, with no raw column read) and the score-influence
    // Jacobian `J`, absorbed by Stage-2 as the realized leakage-projection block.
    // With no CTN Stage-1 recipe, `z` is the raw dose column and the free-warp
    // `score_warp` is the fallback basis.
    let (z, score_influence_jacobian) =
        match crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
        {
            Some(calibration) => (calibration.z_oof, Some(calibration.jac_oof)),
            None => {
                // No recipe ⇒ a raw z_column is required (guarded above) and read here.
                let z_column = z_column.expect("z_column presence checked when ctn_stage1 is None");
                let z_idx = resolve_role_col(col_map, z_column, "z")?;
                let z = data.values.column(z_idx).to_owned();
                validate_bernoulli_marginal_slope_z_column_variance(
                    z_column,
                    z.view(),
                    weights.view(),
                )?;
                (z, None)
            }
        };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
        score_warp: routing.score_warp,
        link_dev: routing.link_dev,
        latent_z_policy: Default::default(),
        score_influence_jacobian,
    };

    Ok(MaterializedModel {
        request: FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
            data: data.values.view(),
            spec,
            options: BlockwiseFitOptions {
                compute_covariance: true,
                // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                // default for bernoulli marginal-slope — no flag to thread.
                ..Default::default()
            },
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            policy,
        }),
        inference_notes,
    })
}

fn materialize_survival<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
    entry_col: Option<&str>,
    exit_col: &str,
    event_col: &str,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let mut inference_notes = Vec::new();

    // Extract columns. `entry_col == None` is the right-censored shorthand
    // `Surv(time, event)`: every subject enters at time zero, so we
    // synthesize a constant-zero entry vector instead of resolving a column.
    let entry_idx = entry_col
        .map(|name| resolve_role_col(col_map, name, "entry"))
        .transpose()?;
    let exit_idx = resolve_role_col(col_map, exit_col, "exit")?;
    let event_idx = resolve_role_col(col_map, event_col, "event")?;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = data.values.nrows();
    let event = data.values.column(event_idx).to_owned();
    let event_codes = Array1::from_iter(
        event
            .iter()
            .copied()
            .enumerate()
            .map(|(i, value)| crate::survival::survival_event_code_from_value(value, i))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let pairs: Result<Vec<(f64, f64)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let entry_val = entry_idx.map_or(0.0, |idx| data.values[[i, idx]]);
            normalize_survival_time_pair(entry_val, data.values[[i, exit_idx]], i)
        })
        .collect();
    let pairs = pairs?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for (i, (e, x)) in pairs.into_iter().enumerate() {
        age_entry[i] = e;
        age_exit[i] = x;
    }

    let survival_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    // Fail fast on all-censored (zero-event) survival data for every survival
    // likelihood (#789B / construction-time fittability split). With no row
    // marking a target event, the survival likelihood has no event score: the
    // hazard direction is unidentified and the inner/outer solve either spins
    // on a flat landscape (marginal-slope) or returns a numerically degenerate
    // fit (other modes). This is the single chokepoint every survival fit
    // dispatcher routes through (Surv(...) responses + all FitConfig survival
    // modes), so catching it here keeps every downstream constructor —
    // `WorkingModelSurvival`, the Royston-Parmar wrapper, the marginal-slope
    // builders — free to materialize models on censored fixtures (which the
    // engine's structural unit tests rely on) without losing the user-facing
    // safety on real fits.
    if !event_codes.iter().any(|&code| code > 0) {
        let mode_label = match survival_mode {
            SurvivalLikelihoodMode::MarginalSlope => "survival marginal-slope",
            _ => "survival fit",
        };
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "{mode_label} requires at least one target event; all rows are censored, so the likelihood has no event score and cannot identify the hazard"
            ),
        });
    }
    let cause_count =
        crate::survival::cause_count_from_event_codes(event_codes.view()).into_workflow_result()?;
    if cause_count > 1
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "cause-specific competing risks with {cause_count} causes are currently supported for survival_likelihood='transformation' and 'weibull'; got '{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkwiggle.is_some()
        && !matches!(
            survival_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "linkwiggle(...) is not defined for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    if parsed.linkspec.is_some()
        && matches!(
            survival_mode,
            SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::Latent
                | SurvivalLikelihoodMode::LatentBinary
        )
    {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "link(...) is not implemented for survival_likelihood='{}'",
                config.survival_likelihood
            ),
        }
        .into());
    }
    // Hoist the survival marginal-slope z-column exclusion check above the
    // time-basis / termspec construction below.  Those downstream steps fail
    // fast on small or tightly-spaced time data (e.g. an I-spline of degree 3
    // cannot be supported by a 2-row fixture), which would otherwise swallow
    // the z-column misuse error and surface a knot-count error instead.
    // Checking here keeps the user-visible error tied to the actual config
    // problem the caller can fix (rename `z` or remove the alias) rather than
    // to an unrelated basis-shape failure further downstream.
    if matches!(survival_mode, SurvivalLikelihoodMode::MarginalSlope)
        && let Some(z_column) = config.z_column.as_deref()
    {
        let logslope_parsed_for_check = match config.logslope_formula.as_deref() {
            Some(ls_formula) => Some(
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?
                    .1,
            ),
            None => None,
        };
        let logslope_ref = logslope_parsed_for_check.as_ref().unwrap_or(parsed);
        validate_marginal_slope_z_column_exclusion(
            parsed,
            logslope_ref,
            z_column,
            "survival marginal-slope",
            "logslope_formula",
        )?;
    }
    let effective_timewiggle = parsed.timewiggle.clone();
    let baseline_target_raw = match survival_mode {
        SurvivalLikelihoodMode::Weibull if effective_timewiggle.is_some() => "weibull",
        SurvivalLikelihoodMode::Weibull => "linear",
        _ => &config.baseline_target,
    };
    let baseline_cfg = initial_survival_baseline_config_for_fit(
        baseline_target_raw,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent hazard-window families require a non-linear scalar baseline target; use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }
    let time_cfg = if effective_timewiggle.is_some() {
        // Match the CLI path: the parametric baseline plus timewiggle supplies
        // the time structure, so the base time basis is disabled.
        SurvivalTimeBasisConfig::None
    } else if survival_mode == SurvivalLikelihoodMode::Weibull {
        SurvivalTimeBasisConfig::Linear
    } else {
        parse_survival_time_basis_config(
            &config.time_basis,
            config.time_degree,
            config.time_num_internal_knots,
            config.time_smooth_lambda,
        )?
    };
    // Marginal-slope centers the baseline-hazard I-spline at a robust interior
    // exit-scale time (median exit) rather than the earliest entry age: under
    // left truncation the earliest entry is a positive left-tail point and
    // centering there inflates the unpenalized linear-trend column, blowing up
    // the time-block seed score so REML rejects every seed (issue #751).
    // Location-scale keeps the earliest-entry anchor.
    let time_anchor = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, None)?
    } else {
        resolve_survival_time_anchor_value(&age_entry, None)?
    };
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(survival_mode);

    // Build time basis
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg.clone(),
        Some((config.time_num_internal_knots, config.time_smooth_lambda)),
    )?;
    if survival_mode != SurvivalLikelihoodMode::Weibull && effective_timewiggle.is_none() {
        require_structural_survival_time_basis(&time_build.basisname, "workflow survival fitting")?;
    }
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string()
                .into(),
        );
    }

    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            // Survival marginal-slope shares the operator-only invariant with
            // the Bernoulli path; flag it as such so strict mode is selected
            // even at small n.
            marginal_slope_large_scale_active: survival_mode
                == SurvivalLikelihoodMode::MarginalSlope,
        },
    );
    // Alias `z` to the dose column for the marginal termspec only when a raw
    // z_column is supplied. With a CTN Stage-1 recipe there is no dose column
    // (z is produced out-of-fold by cross-fitting) and the marginal formula
    // references only the x covariates, so no alias is needed.
    let marginal_slope_aliased_col_map = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(z_column) => Some(column_map_with_alias(col_map, "z", z_column)),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let termspec_col_map = marginal_slope_aliased_col_map.as_ref().unwrap_or(col_map);
    let mut termspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        termspec_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        prune_unidentified_linear_terms_for_marginal_slope(
            &mut termspec,
            data,
            "survival marginal-slope marginal formula",
            &mut inference_notes,
        )?;
    }

    let residual_dist = parse_survival_distribution(&config.survival_distribution)?;
    let survival_inverse_link = residual_distribution_inverse_link(residual_dist);
    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());
    let effective_linkwiggle_cfg = effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let threshold_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let threshold_template = if let Some(k) = config.threshold_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.threshold_time_degree,
            "threshold",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigma_template = if let Some(k) = config.sigma_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.sigma_time_degree,
            "sigma",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigmaspec = if let Some(noise) = config.noise_formula.as_deref() {
        let mut noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());
        // Use the same aliased col_map as the main termspec — survival
        // marginal-slope reserves `z` as a placeholder for `--z-column`,
        // and the logslope/noise formula may reference it too.
        build_termspec_with_geometry_and_overrides(
            &noise_parsed.terms,
            data,
            termspec_col_map,
            &mut inference_notes,
            config.scale_dimensions,
            &policy,
            config.smooth_overrides.as_ref(),
        )?
    } else {
        // No `noise_formula` ⇒ default to an empty log-σ spec for every
        // survival likelihood (constant log-σ baseline owned by the family
        // adapter). The previous `LocationScale`-only branch cloned the
        // mean `termspec` here, which duplicated every threshold term onto
        // the log-σ block. For a smooth `s(x)` on the mean that was
        // structurally fatal: the canonical-gauge identifiability audit
        // saw the log-σ block as exact-aliased to threshold and (per the
        // descending priorities time=200 > threshold=150 > log_sigma=120,
        // issue #366) attributed/dropped every log-σ column, leaving the
        // solver's `ParameterBlockSpec` design at width 0 while the
        // family kept the un-audited `x_log_sigma` at the smooth's width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then errored "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        // The empty default routes through the same
        // `infer_non_intercept_start_design`/`design_column_tail`
        // contract every other mode uses (yielding a 0-column
        // `x_log_sigma` that matches the spec), so the family and spec
        // agree by construction.
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    // `z_column` is OPTIONAL for the survival marginal-slope when a CTN Stage-1
    // recipe is present: the calibrated chain produces the single `z` surface
    // out-of-fold from the cross-fitted CTN, so there is no raw dose column to
    // read (no throwaway pre-fit column — the no-slop cutover, #461). Without a
    // recipe, the primitive standalone survival marginal-slope still requires a
    // raw `z_column` dose.
    let marginal_z_column_name = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        match config.z_column.as_deref() {
            Some(name) => Some(name),
            None if config.ctn_stage1.is_some() => None,
            None => {
                return Err(WorkflowError::InvalidConfig {
                    reason: "marginal-slope survival requires z_column in FitConfig (or a CTN \
                             Stage-1 recipe via ctn_stage1, which produces z by cross-fitting)"
                        .to_string(),
                });
            }
        }
    } else {
        None
    };
    let (
        marginal_z,
        marginal_logslopespec,
        marginal_logslopespecs,
        marginal_slope_deviation_routing,
        marginal_slope_base_link,
    ) = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        let base_link = resolve_survival_marginal_slope_base_link(parsed.linkspec.as_ref())?;
        if marginal_z_column_name.is_none() {
            // Calibrated chain: the CTN Stage-1 recipe produces a SINGLE z surface
            // out-of-fold, so no dose column is read. Stand in an n×1 placeholder
            // surface (the cross-fit below overrides column 0) and build the
            // logslope surface from the formula (or the marginal termspec). The
            // single-surface invariant matches the cross-fit guard further down.
            let placeholder_z = Array2::<f64>::zeros((data.values.nrows(), 1));
            let (logslopespec, routing) = if let Some(ls_formula) =
                config.logslope_formula.as_deref()
            {
                let (_, ls_parsed) = parse_matching_auxiliary_formula(
                    ls_formula,
                    &parsed.response,
                    "logslope_formula",
                )?;
                if ls_parsed.linkspec.is_some() {
                    return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.timewiggle.is_some() {
                    return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                if ls_parsed.survivalspec.is_some() {
                    return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
                }
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &ls_parsed.terms,
                    data,
                    col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                let routing = route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?;
                (spec, routing)
            } else {
                (
                    termspec.clone(),
                    route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                )
            };
            (
                Some(placeholder_z),
                Some(logslopespec.clone()),
                Some(vec![logslopespec]),
                routing,
                Some(base_link),
            )
        } else if let Some(ls_formula) = config.logslope_formula.as_deref() {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string()
                            .into(),
                    );
            }
            validate_marginal_slope_z_column_exclusion(
                parsed,
                &ls_parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let surfaces = marginal_slope_logslope_surfaces(&ls_parsed, default_z_column)?;
            let mut z = Array2::<f64>::zeros((data.values.nrows(), surfaces.len()));
            let mut specs = Vec::with_capacity(surfaces.len());
            for (surface_idx, surface) in surfaces.iter().enumerate() {
                let z_idx = resolve_role_col(col_map, &surface.z_column, "z")?;
                z.column_mut(surface_idx).assign(&data.values.column(z_idx));
                let aliased_col_map = column_map_with_alias(col_map, "z", &surface.z_column);
                let mut spec = build_termspec_with_geometry_and_overrides(
                    &surface.terms,
                    data,
                    &aliased_col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                    config.smooth_overrides.as_ref(),
                )?;
                prune_unidentified_linear_terms_for_marginal_slope(
                    &mut spec,
                    data,
                    "survival marginal-slope logslope_formula",
                    &mut inference_notes,
                )?;
                specs.push(spec);
            }
            (
                Some(z),
                specs.first().cloned(),
                Some(specs),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?,
                Some(base_link),
            )
        } else {
            let default_z_column = marginal_z_column_name.expect("z column present when no recipe");
            validate_marginal_slope_z_column_exclusion(
                parsed,
                parsed,
                default_z_column,
                "survival marginal-slope",
                "logslope_formula",
            )?;
            let z_idx = resolve_role_col(col_map, default_z_column, "z")?;
            let z = data.values.column(z_idx).to_owned().insert_axis(Axis(1));
            (
                Some(z),
                Some(termspec.clone()),
                Some(vec![termspec.clone()]),
                route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)?,
                Some(base_link),
            )
        }
    } else {
        (
            None,
            None,
            None,
            MarginalSlopeDeviationRouting {
                score_warp: None,
                link_dev: None,
            },
            None,
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when the
    // survival marginal-slope `z` was generated by a CTN Stage-1 fit (design
    // §5). Computed once (it refits the CTN K times) — outside the per-baseline
    // request closure below. When active it replaces the (single) CTN-generated
    // z surface with its out-of-fold value and captures the score-influence
    // Jacobian `J` for Stage-2's leakage-projection block. With no CTN Stage-1
    // recipe, the raw z surfaces stand and `score_warp` is the fallback basis.
    let crossfit_calibration = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
    } else {
        None
    };
    let (marginal_z, marginal_slope_jac_oof) = match (marginal_z, crossfit_calibration) {
        (Some(mut z_surfaces), Some(calibration)) => {
            // A CTN Stage-1 chain produces exactly one latent score surface; the
            // OOF projection is defined against that single column.
            if z_surfaces.ncols() != 1 {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "cross-fitted score calibration applies to a single CTN-generated z \
                         surface, but the survival marginal-slope model has {} z surfaces; \
                         multi-surface logslope is incompatible with the CTN Stage-1 chain",
                        z_surfaces.ncols()
                    ),
                });
            }
            z_surfaces.column_mut(0).assign(&calibration.z_oof);
            (Some(z_surfaces), Some(calibration.jac_oof))
        }
        (z, _) => (z, None),
    };

    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes formula-level linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_score_warp.is_some() {
            inference_notes.push(
                "survival marginal-slope routes logslope_formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_link_dev.is_none() && marginal_slope_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }
    }
    let marginal_slope_frailty = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        Some(fixed_gaussian_shift_frailty_from_spec(
            config.frailty.as_ref().unwrap_or(&FrailtySpec::None),
            "survival marginal-slope",
        )?)
    } else {
        None
    };
    match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
            if config.frailty.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "frailty is not supported for transformation/weibull survival models"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::LocationScale if config.frailty.is_some() => {
            return Err(WorkflowError::InvalidConfig {
                reason: "config.frailty is not implemented for survival-likelihood=location-scale"
                    .to_string(),
            }
            .into());
        }
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
            if effective_timewiggle.is_some() =>
        {
            return Err(WorkflowError::InvalidConfig {
                reason: "timewiggle is not implemented for latent survival/binary likelihoods"
                    .to_string(),
            }
            .into());
        }
        _ => {}
    }
    let latent_loading = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let frailty = config.frailty.as_ref().unwrap_or(&FrailtySpec::None);
        Some(latent_hazard_loading(
            frailty,
            "workflow latent survival/binary",
        )?)
    } else {
        None
    };

    let build_time_block =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                (survival_mode == SurvivalLikelihoodMode::LocationScale)
                    .then_some(&survival_inverse_link),
                time_anchor,
                exact_derivative_guard,
                &time_build,
                effective_timewiggle.as_ref(),
                None,
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let initial_beta = if survival_mode == SurvivalLikelihoodMode::LocationScale {
                None
            } else {
                Some(Array1::from_elem(time_p, 1e-4))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta,
            };
            Ok::<_, String>((prepared, time_block))
        };

    // Warm-start cache for the outer CompassSearch over the baseline config: each probe
    // runs a complete inner BFGS over ρ (log-smoothing) starting from zeros if cold; by
    // capturing the previous probe's converged ρ (threshold + log_sigma blocks) and
    // injecting it here, the next inner BFGS typically converges in 1-3 iterations
    // instead of ~10, cutting per-probe cost roughly 5-10× across up to 60 probes per fit.
    let location_scale_smoothing_warm_start: RefCell<Option<(Array1<f64>, Array1<f64>)>> =
        RefCell::new(None);
    let build_location_scale_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig,
         allow_inverse_link_optimization: bool| {
            let (prepared, time_block) = build_time_block(candidate)?;
            let (initial_threshold_log_lambdas, initial_log_sigma_log_lambdas) =
                match location_scale_smoothing_warm_start.borrow().as_ref() {
                    Some((thr, lsg)) => (Some(thr.clone()), Some(lsg.clone())),
                    None => (None, None),
                };
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                inverse_link: survival_inverse_link.clone(),
                derivative_guard: exact_derivative_guard,
                max_iter: 200,
                tol: 1e-7,
                time_block,
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block,
                linkwiggle_block: None,
                initial_threshold_log_lambdas,
                initial_log_sigma_log_lambdas,
                cache_session: None,
                cache_mirror_sessions: Vec::new(),
            };
            // During baseline-θ BFGS probes we hold the inverse-link state
            // fixed: otherwise every probe would trigger a nested
            // CompassSearch over the SAS / BetaLogistic / Mixture link
            // parameters, defeating the BFGS speedup entirely. The final
            // fit (after baseline has converged) flips this back on, so
            // joint baseline + link optimization still happens — just
            // alternating instead of nested.
            let optimize_inverse_link = allow_inverse_link_optimization
                && survival_inverse_link_has_free_parameters(&spec.inverse_link);
            Ok::<_, String>(SurvivalLocationScaleFitRequest {
                data: data.values.view(),
                spec,
                wiggle: effective_linkwiggle_cfg.clone(),
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
                optimize_inverse_link,
                cache_session: None,
            })
        };

    let build_marginal_slope_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let (prepared, mut time_block) = build_time_block(candidate)?;
            time_block.time_monotonicity =
                crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByRowConstraint;
            Ok::<_, String>(SurvivalMarginalSlopeFitRequest {
                data: data.values.view(),
                spec: SurvivalMarginalSlopeTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.clone(),
                    weights: weights.clone(),
                    z: marginal_z.clone().ok_or_else(|| {
                        "marginal-slope survival requires z_column in FitConfig".to_string()
                    })?,
                    base_link: marginal_slope_base_link.clone().ok_or_else(|| {
                        "internal error: marginal-slope base link validation missing".to_string()
                    })?,
                    marginalspec: termspec.clone(),
                    marginal_offset: threshold_offset.clone(),
                    frailty: marginal_slope_frailty.clone().ok_or_else(|| {
                        "internal error: marginal-slope frailty validation missing".to_string()
                    })?,
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    timewiggle_block: prepared.timewiggle_block,
                    logslopespec: marginal_logslopespec.clone().ok_or_else(|| {
                        "marginal-slope survival is missing logslope spec".to_string()
                    })?,
                    logslopespecs: marginal_logslopespecs.clone(),
                    logslope_offset: log_sigma_offset.clone(),
                    score_warp: marginal_slope_score_warp.clone(),
                    link_dev: marginal_slope_link_dev.clone(),
                    latent_z_policy: Default::default(),
                    score_influence_jacobian: marginal_slope_jac_oof.clone(),
                },
                options: BlockwiseFitOptions {
                    compute_covariance: false,
                    // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                    // default for survival marginal-slope — no flag to thread.
                    ..Default::default()
                },
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            })
        };

    let build_latent_survival_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent survival loading missing after frailty validation"
                    .to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentSurvivalFitRequest {
                data: data.values.view(),
                spec: LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let build_latent_binary_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent binary loading missing after frailty validation".to_string()
            })?;
            let prepared = prepare_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                survival_mode,
                None,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                time_monotonicity: crate::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentBinaryFitRequest {
                data: data.values.view(),
                spec: LatentBinaryTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: exact_derivative_guard,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let baseline_cfg = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        baseline_cfg
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::MarginalSlope
    {
        optimize_survival_baseline_config_with_gradient(
            &baseline_cfg,
            "workflow survival marginal-slope baseline",
            |candidate| {
                let fit =
                    fit_survival_marginal_slope_model(build_marginal_slope_request(candidate)?)
                        .map_err(|e| format!("survival marginal-slope fit failed: {e}"))?;
                let gradient = marginal_slope_baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &fit.baseline_offset_residuals,
                )?
                .ok_or_else(|| {
                    "workflow survival marginal-slope baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                let hessian = marginal_slope_baseline_chain_rule_hessian(
                    age_entry.view(),
                    age_exit.view(),
                    candidate,
                    &fit.baseline_offset_residuals,
                    &fit.baseline_offset_curvatures,
                )?
                .ok_or_else(|| {
                    "workflow survival marginal-slope baseline unexpectedly has no theta Hessian"
                        .to_string()
                })?;
                Ok((fit.fit.reml_score, gradient, hessian))
            },
        )?
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear
        && survival_mode == SurvivalLikelihoodMode::LocationScale
    {
        // Analytic θ-gradient path. The baseline configuration enters the
        // location-scale fit only through the three additive time-block
        // offsets (entry η, exit η, exit ∂η/∂t); at the converged β the
        // envelope theorem gives
        //
        //   d(NLL)/dθ_k = Σ_i r^(E)_i ∂o_E_i/∂θ_k
        //               + r^(X)_i ∂o_X_i/∂θ_k
        //               + r^(D)_i ∂o_D_i/∂θ_k
        //
        // where r^(*) are populated by
        // `SurvivalLocationScaleFamily::offset_channel_geometry` and the
        // partials by `baseline_offset_theta_partials`. When the inverse
        // link is probit/SAS/Mixture/etc., the location-scale family uses
        // the probit-channel baseline q(t) instead, so we contract against
        // `marginal_slope_baseline_offset_theta_partials` exactly as the
        // marginal-slope path does. BFGS w/ this analytic gradient
        // typically converges in ≲10 outer evaluations — one order of
        // magnitude fewer probes than the gradient-free compass sweep that
        // used to run on this path.
        let probit_channel =
            location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
        // Catch errors at the optimizer-call site so a single bad θ
        // candidate doesn't blow up the whole `gam.fit()` call. Specific
        // failure mode: when the inner ρ-ARC stalls on a near-flat REML
        // direction (smoothing param running to exp(20+) on an
        // under-identified covariate at small n), the subsequent inner
        // refit can produce a fit whose family methods see empty
        // `block_states` and crash with "expects 3 blocks, got 0". The
        // crash message originates from `validate_joint_states` and can
        // bubble up from any of ~9 callers in the survival family. Rather
        // than enumerating them all, catch the wrapper error here and
        // fall back to the seed baseline_cfg — the gradient path made no
        // progress, but the rest of the fit can proceed at the user's
        // initial GM (α, λ, γ).
        let baseline_outcome = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow survival location-scale baseline",
            |candidate| {
                let fit_result = fit_survival_location_scale_model(build_location_scale_request(
                    candidate, false,
                )?)
                .map_err(|e| format!("survival location-scale fit failed: {e}"))?;
                // Warm-start the next probe's threshold / log-σ smoothing parameters
                // at the converged values for this probe.
                let threshold_rho = fit_result.fit.fit.lambdas_threshold().mapv(f64::ln);
                let log_sigma_rho = fit_result.fit.fit.lambdas_log_sigma().mapv(f64::ln);
                *location_scale_smoothing_warm_start.borrow_mut() =
                    Some((threshold_rho, log_sigma_rho));
                let residuals = &fit_result.fit.baseline_offset_residuals;
                let gradient = if probit_channel {
                    marginal_slope_baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        residuals,
                    )?
                } else {
                    baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        residuals,
                    )?
                }
                .ok_or_else(|| {
                    "workflow survival location-scale baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                // The envelope-theorem residual contraction is the exact
                // θ-gradient of the *profile penalized NLL* −ℓ + ½βᵀSβ at
                // converged (β̂, ρ̂). Optimizing `reml_score` (which includes
                // ½ log|S_λ| − ½ log|H| LAML corrections) against this
                // gradient would mismatch the cost surface, because the
                // log-determinant terms have their own θ-dependence through
                // H(β̂, θ). Use the matching profile-NLL cost here; the final
                // model refit downstream still picks ρ via the full REML
                // surface at the converged baseline θ.
                let profile_cost = -fit_result.fit.fit.log_likelihood
                    + 0.5 * fit_result.fit.fit.stable_penalty_term;
                if !profile_cost.is_finite() {
                    return Err(format!(
                        "workflow survival location-scale baseline: non-finite profile cost \
                         (log_likelihood={}, stable_penalty_term={}, cost={})",
                        fit_result.fit.fit.log_likelihood,
                        fit_result.fit.fit.stable_penalty_term,
                        profile_cost
                    ));
                }
                Ok((profile_cost, gradient))
            },
        );
        match baseline_outcome {
            Ok(baseline) => baseline,
            Err(e)
                if e.contains("expects 3 blocks, got 0")
                    || e.contains("expects 4 blocks, got 0")
                    || (e.contains("block_states") && e.contains("got 0"))
                    || e.contains("blockwise fit requires at least one block state")
                    || e.contains(SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER) =>
            {
                log::warn!(
                    "workflow survival location-scale baseline: gradient-only BFGS \
                     failed at an empty-block_states candidate ({e}); falling back \
                     to the seed baseline_cfg as-is"
                );
                baseline_cfg.clone()
            }
            Err(e) => return Err(e.into()),
        }
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        optimize_survival_baseline_config(
            &baseline_cfg,
            "workflow survival baseline",
            |candidate| {
                match survival_mode {
                SurvivalLikelihoodMode::LocationScale => Err(
                    "internal: location-scale baseline profiling uses analytic chain-rule gradient and should not reach the workflow scalar profile closure"
                        .to_string(),
                ),
                SurvivalLikelihoodMode::MarginalSlope => Err(
                    "internal: marginal-slope baseline profiling uses analytic GM-probit gradient and should not reach the workflow scalar profile closure"
                        .to_string(),
                ),
                SurvivalLikelihoodMode::Latent => Ok(fit_latent_survival_model(
                    build_latent_survival_request(candidate)?,
                )
                .map_err(|e| format!("latent survival fit failed: {e}"))?
                .fit
                .reml_score),
                SurvivalLikelihoodMode::LatentBinary => Ok(fit_latent_binary_model(
                    build_latent_binary_request(candidate)?,
                )
                .map_err(|e| format!("latent binary fit failed: {e}"))?
                .fit
                .reml_score),
                SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                    Err(
                        "internal: Transformation/Weibull survival baseline should not enter the workflow scalar profile closure (outer guard filters them above)"
                            .to_string(),
                    )
                }
            }
            },
        )?
    } else {
        baseline_cfg
    };

    let request = match survival_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
            if config.noise_offset_column.is_some() {
                return Err(WorkflowError::InvalidConfig {
                    reason:
                        "noise_offset_column is supported only for survival location-scale or marginal-slope"
                            .to_string(),
                }
                .into());
            }
            let weibull_seed = if survival_mode == SurvivalLikelihoodMode::Weibull
                && effective_timewiggle.is_none()
            {
                let scale = config
                    .baseline_scale
                    .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
                let shape = config.baseline_shape.unwrap_or(1.0);
                if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                    return Err(WorkflowError::InvalidConfig {
                        reason:
                            "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                                .to_string(),
                    }
                    .into());
                }
                Some((scale, shape))
            } else {
                None
            };
            FitRequest::SurvivalTransformation(SurvivalTransformationFitRequest {
                data: data.values.view(),
                spec: SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_codes.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: threshold_offset.clone(),
                    baseline_cfg,
                    likelihood_mode: survival_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    ridge_lambda: config.ridge_lambda,
                    penalty_block_gamma_priors: config.penalty_block_gamma_priors.clone(),
                },
                cache_session: None,
            })
        }
        SurvivalLikelihoodMode::LocationScale => {
            FitRequest::SurvivalLocationScale(build_location_scale_request(&baseline_cfg, true)?)
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            FitRequest::SurvivalMarginalSlope(build_marginal_slope_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::Latent => {
            FitRequest::LatentSurvival(build_latent_survival_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::LatentBinary => {
            FitRequest::LatentBinary(build_latent_binary_request(&baseline_cfg)?)
        }
    };

    Ok(MaterializedModel {
        request,
        inference_notes,
    })
}

fn materialize_transformation_normal<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if parsed.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported for the transformation-normal family".to_string(),
        }
        .into());
    }
    if parsed.linkwiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) is not supported for the transformation-normal family"
                .to_string(),
        }
        .into());
    }
    if config.noise_offset_column.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "noise_offset_column is not supported for transformation-normal models"
                .to_string(),
        }
        .into());
    }
    if config.frailty.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "frailty is not supported for transformation-normal models".to_string(),
        }
        .into());
    }

    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let policy = resolved_resource_policy(config, data, marginal_slope_hints(config));
    let covariate_spec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;

    Ok(MaterializedModel {
        request: FitRequest::TransformationNormal(TransformationNormalFitRequest {
            data: data.values.view(),
            response: y,
            weights,
            offset,
            covariate_spec,
            config: TransformationNormalConfig::default(),
            options: BlockwiseFitOptions::default(),
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            warm_start: None,
        }),
        inference_notes,
    })
}

/// Apply basis parsimony to a *secondary* (distributional) predictor's smooths.
///
/// In a location-scale / GAMLSS fit the mean is identified directly by the
/// response and warrants the generous default basis, but the scale (log-σ) and
/// other distributional predictors are identified only through (noisy) squared
/// residuals. Handing their radial spatial smooths a basis sized for the mean
/// lets REML over-fit them (#501). For each spatial smooth (thin-plate /
/// Matérn / Duchon) the user did not size explicitly, cap the *default* center
/// count via the private [`SECONDARY_CENTER_CAP_OPTION`]. The cap lowers the
/// default while preserving the `Auto` center strategy, so the basis is still
/// softly reduced when the data can't support the count (rather than erroring
/// like an explicit count would). Smooths the user sized explicitly, and the
/// non-radial bases (B-spline, cyclic, tensor) which already default modestly
/// via knot counts, are deliberately left untouched.
fn apply_secondary_predictor_basis_parsimony(terms: &mut [ParsedTerm], n_rows: usize) {
    for term in terms.iter_mut() {
        if let ParsedTerm::Smooth {
            vars,
            kind,
            options,
            ..
        } = term
        {
            let canonical = resolve_smooth_type_name(*kind, vars.len(), options);
            if !smooth_type_uses_spatial_center_heuristic(&canonical)
                || has_explicit_countwith_basis_alias(options, "centers")
            {
                continue;
            }
            let cap = crate::terms::basis::conservative_secondary_centers(n_rows, vars.len());
            options.insert(SECONDARY_CENTER_CAP_OPTION.to_string(), cap.to_string());
        }
    }
}

fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    let mut noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;
    apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation, owned by the family type.
    // See `ResponseFamily::validate_response_support`.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331, #332). The
    // location-scale path has its own σ-model so a near-constant Gaussian
    // mean response is even more pathological here than in the standard
    // path; same typed check, same family-owned classifier.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let policy = resolved_resource_policy(config, data, crate::resource::ProblemHints::default());
    let meanspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    let log_sigmaspec = build_termspec_with_geometry_and_overrides(
        &noise_parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
    )?;
    // Sample size vs basis rank, summed across the mean and log-σ smooths
    // (#309). Both designs share the same n_rows.
    check_smooth_capacity(&meanspec, y.len(), &parsed.response)?;
    check_smooth_capacity(&log_sigmaspec, y.len(), &parsed.response)?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let mean_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let options = BlockwiseFitOptions::default();

    let wiggle_cfg = effective_linkwiggle.map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    if family.is_latent_cloglog() {
        return Err(WorkflowError::InvalidConfig {
            reason: "latent-cloglog-binomial is not implemented for location-scale fitting"
                .to_string(),
        }
        .into());
    }

    if family.is_binomial() {
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                Err(e) => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "binomial location-scale fitting cannot route link `{}` through `InverseLink::Standard`: {e}",
                            c.link.name()
                        ),
                    }
                    .into());
                }
            },
            None => InverseLink::Standard(StandardLink::Logit),
        };
        Ok(MaterializedModel {
            request: FitRequest::BinomialLocationScale(BinomialLocationScaleFitRequest {
                data: data.values.view(),
                spec: BinomialLocationScaleTermSpec {
                    y,
                    weights,
                    link_kind,
                    thresholdspec: meanspec,
                    log_sigmaspec,
                    threshold_offset: mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    } else if let Some(kind) = dispersion_location_scale_kind(&family.response) {
        // Genuine-dispersion mean families (NegativeBinomial / Gamma / Beta /
        // Tweedie): `noise_formula` models the overdispersion channel (#913).
        // A link-wiggle is mean-only and not defined here.
        if wiggle_cfg.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "link-wiggle is not supported for {} location-scale models",
                    kind.family_tag()
                ),
            }
            .into());
        }
        Ok(MaterializedModel {
            request: FitRequest::DispersionLocationScale(DispersionLocationScaleFitRequest {
                data: data.values.view(),
                spec: DispersionGlmLocationScaleTermSpec {
                    kind,
                    y,
                    weights,
                    meanspec,
                    log_dispspec: log_sigmaspec,
                    mean_offset,
                    log_disp_offset: noise_offset,
                },
                options,
                kappa_options,
            }),
            inference_notes,
        })
    } else {
        Ok(MaterializedModel {
            request: FitRequest::GaussianLocationScale(GaussianLocationScaleFitRequest {
                data: data.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y,
                    weights,
                    meanspec,
                    log_sigmaspec,
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    }
}

/// Map a [`ResponseFamily`] to the dispersion-GAM kind whose overdispersion
/// channel can carry a `noise_formula` (#913), or `None` for families handled
/// by the Gaussian/Binomial location-scale paths.
fn dispersion_location_scale_kind(response: &ResponseFamily) -> Option<DispersionFamilyKind> {
    match response {
        ResponseFamily::NegativeBinomial { .. } => Some(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Some(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Some(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Some(DispersionFamilyKind::Tweedie { p: *p }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{DuchonNullspaceOrder, minimum_duchon_power_for_operator_penalties};
    use crate::inference::data::load_dataset_projected;
    use crate::inference::formula_dsl::{
        default_linkwiggle_formulaspec, parse_linkwiggle_formulaspec,
    };
    use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
    use crate::smooth::SmoothBasisSpec;
    use crate::solver::outer_strategy::{HessianSource, OuterPlan, OuterResult, Solver};
    use ndarray::Array2;
    use std::fs;
    use tempfile::tempdir;

    fn load_survival_dataset() -> crate::inference::data::EncodedDataset {
        let td = tempdir().expect("tempdir");
        let data_path = td.path().join("survival.csv");
        fs::write(
            &data_path,
            "entry,exit,event,x,z\n0.0,1.0,1,0.2,-0.4\n0.3,1.6,0,-0.1,0.6\n",
        )
        .expect("write survival csv");
        load_dataset_projected(
            &data_path,
            &[
                "entry".to_string(),
                "exit".to_string(),
                "event".to_string(),
                "x".to_string(),
                "z".to_string(),
            ],
        )
        .expect("load survival dataset")
    }

    #[test]
    fn competing_risks_baseline_seed_replicates_to_match_cause_specific_beta_length() {
        // Regression for #378's downstream break: the cause-specific assembly in
        // `fit_cause_specific_survival_transformation_custom` requires exactly
        // `p * cause_count` initial coefficients (it slices `cause * p..(cause +
        // 1) * p` per cause). The pooled baseline working model returns a
        // length-`p` seed, so without per-cause replication every `cause_count >
        // 1` fit aborts with a `SchemaMismatch` length mismatch. This pins that
        // the replication helper produces the exact length the assembly checks
        // for, and seeds each cause from the same pooled baseline.
        let pooled = Array1::from_vec(vec![-1.5_f64, 0.8, 0.0]);
        let p = pooled.len();

        for cause_count in [1usize, 2, 3] {
            let flat = replicate_pooled_baseline_seed_per_cause(pooled.view(), cause_count);
            // The exact invariant the cause-specific length guard enforces.
            assert_eq!(
                flat.len(),
                p * cause_count,
                "replicated seed must satisfy the `p * cause_count` length contract"
            );
            // Every per-cause slice must equal the shared pooled baseline seed.
            for cause in 0..cause_count {
                let slice = flat.slice(s![cause * p..(cause + 1) * p]);
                assert_eq!(
                    slice.to_owned(),
                    pooled,
                    "cause {cause} block must be seeded from the pooled baseline"
                );
            }
        }
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_main_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("main formula should reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("main formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_logslope_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1 + z".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x", &data, &config)
            .err()
            .expect("logslope formula should reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("logslope_formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_when_logslope_defaults_to_main_spec() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("defaulted logslope spec should still reject z-column reuse");

        assert!(
            err.to_string()
                .contains("survival marginal-slope reserves z column 'z'")
        );
        assert!(err.to_string().contains("main formula"));
    }

    #[test]
    fn survival_marginal_slope_matern_logslope_penalties_keep_surface_width() {
        let n = 24usize;
        let mut values = Array2::<f64>::zeros((n, 8));
        for i in 0..n {
            let u = i as f64 / (n - 1) as f64;
            values[[i, 0]] = 0.0;
            values[[i, 1]] = 0.25 + 8.0 * u;
            values[[i, 2]] = if i % 3 == 0 { 1.0 } else { 0.0 };
            values[[i, 3]] = ((i * 17 % 23) as f64 - 11.0) / 7.0;
            values[[i, 4]] = (2.0 * std::f64::consts::PI * u).sin();
            values[[i, 5]] = (2.0 * std::f64::consts::PI * u).cos();
            values[[i, 6]] = 2.0 * u - 1.0;
            values[[i, 7]] = if i % 2 == 0 { 0.0 } else { 1.0 };
        }
        let data = Dataset {
            headers: vec![
                "t0".to_string(),
                "t1".to_string(),
                "event".to_string(),
                "z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
                "sex".to_string(),
            ],
            values,
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "t0".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "t1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "sex".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
            ],
        };
        for (case, formula) in [
            (
                "with parametric sex term",
                "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6) + sex",
            ),
            (
                "without parametric sex term",
                "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6)",
            ),
        ] {
            let config = FitConfig {
                survival_likelihood: "marginal-slope".to_string(),
                logslope_formula: Some("matern(PC1, PC2, PC3, centers=6)".to_string()),
                z_column: Some("z".to_string()),
                ..FitConfig::default()
            };

            let materialized = materialize(formula, &data, &config).unwrap_or_else(|err| {
                panic!(
                    "survival marginal-slope materialization should keep block-local penalties \
                     {case}: {err}"
                )
            });
            let FitRequest::SurvivalMarginalSlope(request) = materialized.request else {
                panic!("expected survival marginal-slope request for {case}");
            };
            let specs = vec![
                request.spec.marginalspec.clone(),
                request.spec.logslopespec.clone(),
            ];
            let (designs, frozen_specs) =
                crate::smooth::build_term_collection_designs_and_freeze_joint(
                    data.values.view(),
                    &specs,
                )
                .unwrap_or_else(|err| {
                    panic!("joint freeze should preserve per-block penalty geometry {case}: {err}")
                });
            let (rebuilt, _) = crate::smooth::build_term_collection_designs_and_freeze_joint(
                data.values.view(),
                &frozen_specs,
            )
            .unwrap_or_else(|err| {
                panic!("frozen rebuild should preserve per-block penalty geometry {case}: {err}")
            });

            for (label, design) in [
                ("raw marginal", &designs[0]),
                ("raw logslope", &designs[1]),
                ("frozen marginal", &rebuilt[0]),
                ("frozen logslope", &rebuilt[1]),
            ] {
                let width = design.design.ncols();
                assert!(
                    width > 2,
                    "{case}: {label} design should be surface-width, not sex/intercept-width; \
                     width={width}"
                );
                for (idx, penalty) in design.penalties_as_penalty_matrix().iter().enumerate() {
                    assert_eq!(
                        penalty.shape(),
                        (width, width),
                        "{case}: {label} penalty {idx} must be block-local at the surface width"
                    );
                }
            }
        }
    }

    fn workflow_test_dataset() -> Dataset {
        Dataset {
            headers: vec![
                "age_entry".to_string(),
                "age_exit".to_string(),
                "event".to_string(),
                "bmi".to_string(),
                "z".to_string(),
            ],
            values: Array2::from_shape_vec(
                (4, 5),
                vec![
                    40.0, 43.0, 1.0, 22.0, -1.0, 41.0, 46.0, 0.0, 24.0, -0.2, 42.0, 47.0, 1.0,
                    27.0, 0.3, 44.0, 49.0, 0.0, 29.0, 1.2,
                ],
            )
            .expect("workflow test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "age_entry".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "age_exit".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "bmi".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        }
    }

    #[test]
    fn issue_789_transformation_normal_rejects_marginal_slope_controls_before_dispatch() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            transformation_normal: true,
            family: Some("bernoulli-marginal-slope".to_string()),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };

        let err = materialize("event ~ bmi", &data, &config)
            .err()
            .expect("transformation_normal must not steal marginal-slope fits");

        assert!(
            err.to_string()
                .contains("transformation_normal cannot be combined with marginal-slope")
        );
    }

    #[test]
    fn survival_marginal_slope_rejects_zero_event_data_before_fit() {
        let mut data = workflow_test_dataset();
        data.values.column_mut(2).fill(0.0);
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };

        let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
            .err()
            .expect("zero-event survival marginal-slope data must fail before optimization");

        assert!(err.to_string().contains("at least one target event"));
    }

    fn workflow_test_outer_result(converged: bool, rho: Array1<f64>) -> OuterResult {
        let mut result = OuterResult::new(
            rho,
            1.25,
            7,
            converged,
            OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        );
        result.final_grad_norm = Some(0.5);
        result
    }

    fn duchon_workflow_dataset() -> Dataset {
        let n = 72usize;
        let mut values = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            values[[i, 0]] = 0.5 * t.sin() + 0.15 * (3.0 * t).cos();
            values[[i, 1]] = t.cos();
            values[[i, 2]] = t.sin();
        }
        Dataset {
            headers: vec!["y".to_string(), "ct".to_string(), "st".to_string()],
            values,
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "y".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "ct".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "st".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        }
    }

    #[test]
    fn materialize_standard_keeps_adaptive_regularization_off_by_default_for_duchon() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12)",
            &data,
            &FitConfig::default(),
        )
        .expect("Duchon standard materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        assert!(request.options.adaptive_regularization.is_none());
    }

    #[test]
    fn materialize_standard_honors_adaptive_regularization_enable() {
        let data = duchon_workflow_dataset();
        let config = FitConfig {
            adaptive_regularization: Some(true),
            ..FitConfig::default()
        };
        let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
            .expect("Duchon materialization should allow enabling adaptive regularization");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let opts = request
            .options
            .adaptive_regularization
            .expect("Duchon should enable adaptive regularization when requested");
        assert!(opts.enabled);
    }

    #[test]
    fn materialize_standard_honors_adaptive_regularization_disable() {
        let data = duchon_workflow_dataset();
        let config = FitConfig {
            adaptive_regularization: Some(false),
            ..FitConfig::default()
        };
        let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
            .expect("Duchon materialization should allow disabling adaptive regularization");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        assert!(request.options.adaptive_regularization.is_none());
    }

    #[test]
    fn materialize_standard_duchon_defaults_to_pure_scale_free_basis() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12)",
            &data,
            &FitConfig::default(),
        )
        .expect("Duchon materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
            panic!("expected Duchon smooth");
        };
        assert_eq!(spec.length_scale, None);
        assert!(matches!(spec.nullspace_order, DuchonNullspaceOrder::Linear));
        assert_eq!(spec.power, 0.5);
    }

    #[test]
    fn materialize_standard_duchon_length_scale_opts_into_hybrid_basis() {
        let data = duchon_workflow_dataset();
        let materialized = materialize(
            "y ~ duchon(ct, st, centers=12, length_scale=1.0)",
            &data,
            &FitConfig::default(),
        )
        .expect("hybrid Duchon materialization should succeed");
        let FitRequest::Standard(request) = materialized.request else {
            panic!("expected standard request");
        };
        let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
            panic!("expected Duchon smooth");
        };
        assert_eq!(spec.length_scale, Some(1.0));
        assert_eq!(spec.nullspace_order, DuchonNullspaceOrder::Linear);
        // The hybrid Matérn-blended kernel requires an INTEGER power. The cubic
        // structural default's fractional s=(d-1)/2 = 0.5 (d=2) is resolved at the
        // request layer to the smallest admissible integer (here s=0, the d=2
        // thin-plate order) rather than carried in as 0.5 and silently truncated
        // to 0 by the basis builder (#750). The pure path above still keeps 0.5.
        assert_eq!(spec.power, 0.0);
    }

    #[test]
    fn workflow_survival_marginal_slope_routes_logslope_linkwiggle_into_score_warp_only() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some(
                "1 + linkwiggle(degree=5, internal_knots=7, penalty_order=\"2,3\")".to_string(),
            ),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ s(bmi) + linkwiggle(degree=4, internal_knots=9, penalty_order=\"1\")",
            &data,
            &config,
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel {
            request,
            inference_notes,
        } = materialized;
        let FitRequest::SurvivalMarginalSlope(request) = request else {
            panic!("expected survival marginal-slope request");
        };

        let link_dev = request.spec.link_dev.expect("main-formula link-dev");
        let score_warp = request.spec.score_warp.expect("logslope score-warp");
        assert_eq!(link_dev.degree, 4);
        assert_eq!(link_dev.num_internal_knots, 9);
        assert_eq!(link_dev.penalty_order, 1);
        assert_eq!(link_dev.penalty_orders, vec![1]);
        assert_eq!(score_warp.degree, 5);
        assert_eq!(score_warp.num_internal_knots, 7);
        assert_eq!(score_warp.penalty_order, 3);
        assert_eq!(score_warp.penalty_orders, vec![2, 3]);
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("link-deviation block")),
            "workflow notes should mention main-formula linkwiggle routing"
        );
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("score-warp block")),
            "workflow notes should mention logslope_formula linkwiggle routing"
        );
    }

    #[test]
    fn materialize_routes_bernoulli_marginal_slope_when_logslope_and_z_are_set() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize("event ~ bmi", &data, &config)
            .expect("Bernoulli marginal-slope materialization should succeed");
        assert!(matches!(
            materialized.request,
            FitRequest::BernoulliMarginalSlope(_)
        ));
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_prunes_redundant_scalar_term() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "x".to_string(),
                "constant_spline_col".to_string(),
                "prs_z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
            ],
            values: Array2::from_shape_vec(
                (6, 7),
                vec![
                    0.0, -2.0, 1.0, -1.2, -1.0, 0.2, 0.7, 1.0, -1.0, 1.0, -0.4, -0.4, -0.3, 0.5,
                    0.0, 0.0, 1.0, 0.1, 0.1, 0.4, -0.2, 1.0, 1.0, 1.0, 0.5, 0.7, -0.6, 0.3, 0.0,
                    2.0, 1.0, 1.1, 1.2, 0.9, 0.0, 1.0, 3.0, 1.0, 1.7, 1.6, -0.8, -0.4,
                ],
            )
            .expect("BMS redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "event ~ matern(PC1, PC2, PC3, centers=3) + x + constant_spline_col",
            &data,
            &config,
        )
        .expect("BMS materialization should prune the redundant scalar term");
        let MaterializedModel {
            request,
            inference_notes,
        } = materialized;
        let FitRequest::BernoulliMarginalSlope(request) = request else {
            panic!("expected Bernoulli marginal-slope request");
        };
        let kept: Vec<&str> = request
            .spec
            .marginalspec
            .linear_terms
            .iter()
            .map(|term| term.name.as_str())
            .collect();
        assert_eq!(kept, vec!["x"]);
        assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
        assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("constant_spline_col")),
            "materialization should report the removed redundant scalar term; notes={inference_notes:?}"
        );
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_prunes_binary_outcome_style_scalar_alias() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "sex".to_string(),
                "entry_age_z".to_string(),
                "current_age_ns_1".to_string(),
                "current_age_ns_2".to_string(),
                "current_age_ns_3".to_string(),
                "current_age_ns_4".to_string(),
                "prs_z".to_string(),
                "PC1".to_string(),
                "PC2".to_string(),
                "PC3".to_string(),
            ],
            values: Array2::from_shape_vec(
                (8, 11),
                vec![
                    0.0, 0.0, -1.4, 1.0, -0.6, 0.36, -0.216, -1.3, -1.0, 0.2, 0.7, 1.0, 1.0, -0.9,
                    1.0, -0.2, 0.04, -0.008, -0.8, -0.5, -0.3, 0.5, 0.0, 0.0, -0.5, 1.0, 0.1, 0.01,
                    0.001, -0.2, 0.1, 0.4, -0.2, 1.0, 1.0, -0.1, 1.0, 0.4, 0.16, 0.064, 0.3, 0.7,
                    -0.6, 0.3, 0.0, 0.0, 0.3, 1.0, 0.7, 0.49, 0.343, 0.8, 1.2, 0.9, 0.0, 1.0, 1.0,
                    0.7, 1.0, 1.0, 1.0, 1.0, 1.2, 1.6, -0.8, -0.4, 0.0, 0.0, 1.1, 1.0, 1.3, 1.69,
                    2.197, 1.6, -1.4, 0.8, -0.9, 1.0, 1.0, 1.5, 1.0, 1.6, 2.56, 4.096, 2.0, 0.3,
                    -1.1, 0.6,
                ],
            )
            .expect("binary-outcome-style BMS scalar-alias test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "sex".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "entry_age_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "current_age_ns_4".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "PC3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "event ~ matern(PC1, PC2, PC3, centers=3) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
            &data,
            &config,
        )
        .expect("BMS materialization should prune the local-column-3 scalar alias");
        let FitRequest::BernoulliMarginalSlope(request) = materialized.request else {
            panic!("expected Bernoulli marginal-slope request");
        };
        let kept: Vec<&str> = request
            .spec
            .marginalspec
            .linear_terms
            .iter()
            .map(|term| term.name.as_str())
            .collect();
        assert_eq!(
            kept,
            vec![
                "sex",
                "entry_age_z",
                "current_age_ns_2",
                "current_age_ns_3",
                "current_age_ns_4"
            ]
        );
        assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
        assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
        assert!(
            materialized
                .inference_notes
                .iter()
                .any(|note| note.contains("current_age_ns_1")),
            "materialization should report the removed binary-outcome-style scalar alias; notes={:?}",
            materialized.inference_notes
        );
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_rejects_constrained_redundant_scalar_term() {
        let data = Dataset {
            headers: vec![
                "event".to_string(),
                "x".to_string(),
                "constant_spline_col".to_string(),
                "prs_z".to_string(),
            ],
            values: Array2::from_shape_vec(
                (6, 4),
                vec![
                    0.0, -2.0, 1.0, -1.2, 1.0, -1.0, 1.0, -0.4, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 1.0,
                    0.5, 0.0, 2.0, 1.0, 1.1, 1.0, 3.0, 1.0, 1.7,
                ],
            )
            .expect("BMS constrained redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };
        let err = match materialize(
            "event ~ x + linear(constant_spline_col, min=0.0)",
            &data,
            &config,
        ) {
            Ok(_) => panic!("constrained duplicate scalar term must be rejected, not pruned"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("constrained linear term 'constant_spline_col' is redundant"),
            "error should explain that the constrained duplicate scalar cannot be pruned: {msg}"
        );
    }

    #[test]
    fn bernoulli_marginal_slope_prune_rejects_penalized_redundant_scalar_term() {
        let data = Dataset {
            headers: vec!["event".to_string(), "constant_spline_col".to_string()],
            values: Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
                .expect("BMS penalized redundant scalar test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "constant_spline_col".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![ColumnKindTag::Binary, ColumnKindTag::Continuous],
        };
        let mut spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "constant_spline_col".to_string(),
                feature_col: 1,
                feature_cols: vec![1],
                double_penalty: true,
                coefficient_geometry: crate::smooth::LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let mut notes = Vec::new();
        let err = prune_unidentified_linear_terms_for_marginal_slope(
            &mut spec,
            &data,
            "test BMS formula",
            &mut notes,
        )
        .expect_err("explicitly penalized duplicate scalar term must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("explicitly penalized linear term 'constant_spline_col' is redundant"),
            "error should reject ridge-identification of duplicate scalar directions: {msg}"
        );
        assert_eq!(spec.linear_terms.len(), 1);
        assert!(notes.is_empty());
    }

    #[test]
    fn materialize_bernoulli_marginal_slope_names_constant_z_column() {
        let data = Dataset {
            headers: vec!["event".to_string(), "bmi".to_string(), "prs_z".to_string()],
            values: Array2::from_shape_vec(
                (4, 3),
                vec![
                    0.0, 22.0, -0.58, 1.0, 24.0, -0.58, 0.0, 27.0, -0.58, 1.0, 29.0, -0.58,
                ],
            )
            .expect("constant z test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "bmi".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "prs_z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let config = FitConfig {
            logslope_formula: Some("1".to_string()),
            z_column: Some("prs_z".to_string()),
            ..FitConfig::default()
        };

        let err = match materialize("event ~ bmi", &data, &config) {
            Ok(_) => panic!("constant z_column should be rejected before BMS integration"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("z_column 'prs_z' has zero weighted variance"),
            "error should name the constant z_column and diagnose weighted variance: {msg}"
        );
        assert!(
            msg.contains("all 4 values ~= -0.580000"),
            "error should summarize the observed constant value: {msg}"
        );
        assert!(
            msg.contains("weighted_sd=0.000000e0") && msg.contains("n=4"),
            "error should report weighted_sd and n: {msg}"
        );
        assert!(
            msg.contains(
                "bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score"
            ),
            "error should explain why the input is invalid: {msg}"
        );
        assert!(
            !msg.contains("requires z with positive finite weighted standard deviation"),
            "workflow should surface the input-style message instead of the generic BMS normalization error: {msg}"
        );
    }

    #[test]
    fn linkwiggle_defaults_are_consistent_across_formula_and_runtime() {
        let parsed = parse_linkwiggle_formulaspec(&Default::default(), "linkwiggle()")
            .expect("default linkwiggle should parse");
        let formula_default = default_linkwiggle_formulaspec();
        let runtime_default = DeviationBlockConfig::default();
        assert_eq!(parsed.degree, formula_default.degree);
        assert_eq!(
            parsed.num_internal_knots,
            formula_default.num_internal_knots
        );
        assert_eq!(parsed.penalty_orders, formula_default.penalty_orders);
        assert_eq!(parsed.double_penalty, formula_default.double_penalty);
        assert_eq!(runtime_default.degree, formula_default.degree);
        assert_eq!(
            runtime_default.num_internal_knots,
            formula_default.num_internal_knots
        );
        assert_eq!(
            runtime_default.penalty_orders,
            formula_default.penalty_orders
        );
        assert_eq!(
            runtime_default.double_penalty,
            formula_default.double_penalty
        );
    }

    #[test]
    fn survival_marginal_slope_accepts_explicit_probit_link() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let ok = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + link(type=probit)",
            &data,
            &config,
        );
        assert!(ok.is_ok(), "explicit probit should be accepted");

        let err = match materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + link(type=logit)",
            &data,
            &config,
        ) {
            Ok(_) => panic!("non-probit link should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("only link(type=probit)"));
    }

    #[test]
    fn high_dimensional_duchon_default_power_is_admissible() {
        let dim = 16;
        let power = minimum_duchon_power_for_operator_penalties(dim, DuchonNullspaceOrder::Zero, 2);
        assert!(2 * (1 + power) > dim + 2);
    }

    #[test]
    fn survival_location_scale_wiggle_rejects_unsupported_inverse_link() {
        let data = workflow_test_dataset();
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + linkwiggle(degree=4, internal_knots=3, penalty_order=\"1\")",
            &data,
            &FitConfig::default(),
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel { request, .. } = materialized;
        let FitRequest::SurvivalLocationScale(mut request) = request else {
            panic!("expected survival location-scale request");
        };
        request.spec.inverse_link = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: 0.1,
                initial_log_delta: 0.0,
            })
            .expect("valid SAS state"),
        );
        request.optimize_inverse_link = false;

        let err = match fit_survival_location_scale_model(request) {
            Ok(_) => panic!("survival link wiggle should reject unsupported inverse links"),
            Err(e) => e,
        };

        assert!(err.contains("survival link wiggle"));
        assert!(err.contains("does not support"));
    }

    #[test]
    fn survival_inverse_link_result_requires_convergence() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(false, Array1::from_vec(vec![0.1, -0.2])),
            "survival inverse-link optimization (SAS, dim=2)",
            |_| Some(InverseLink::Standard(StandardLink::Logit)),
        )
        .expect_err("non-converged inverse-link search should fail");

        assert!(err.contains("did not converge"));
        assert!(err.contains("final_objective"));
    }

    #[test]
    fn survival_inverse_link_result_requires_recoverable_state() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(true, Array1::from_vec(vec![9.0, 8.0])),
            "survival inverse-link optimization (mixture, dim=2)",
            |_| None,
        )
        .expect_err("unrecoverable inverse-link state should fail");

        assert!(err.contains("produced an invalid inverse-link state"));
        assert!(err.contains("9.0"));
    }

    // #371: survival-only / binomial-only DSL controls must be *rejected* in a
    // non-survival main formula, not parsed-and-silently-dropped. The bug was
    // that `parsed.timewiggle` / `parsed.survivalspec` are consumed only by
    // `materialize_survival`, and an explicit `linkwiggle(...)` is wired into
    // the fit only on the binomial arm, so a Gaussian formula carrying any of
    // these accepted the term and then ignored it — the user got an ordinary
    // GAM while believing they had configured a time-varying / wiggled model.

    #[test]
    fn timewiggle_rejected_in_nonsurvival_main_formula() {
        // `bmi` is a continuous response -> Gaussian standard path, no Surv(...).
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + timewiggle(internal_knots=4)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("timewiggle in a non-survival formula must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("timewiggle(...)") && msg.contains("survival"),
            "error should explain timewiggle is survival-only, got: {msg}"
        );
    }

    #[test]
    fn survmodel_rejected_in_nonsurvival_main_formula() {
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + survmodel(spec=net)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("survmodel in a non-survival formula must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("survmodel(...)") && msg.contains("survival"),
            "error should explain survmodel is survival-only, got: {msg}"
        );
    }

    #[test]
    fn linkwiggle_rejected_for_nonbinomial_response() {
        // `bmi` is continuous -> Gaussian; an explicit `linkwiggle(...)` corrects
        // a binomial link and would otherwise be dropped on the floor here.
        let data = workflow_test_dataset();
        let err = materialize(
            "bmi ~ z + linkwiggle(internal_knots=4)",
            &data,
            &FitConfig::default(),
        )
        .err()
        .expect("linkwiggle on a non-binomial response must be rejected, not silently ignored");
        let msg = err.to_string();
        assert!(
            msg.contains("linkwiggle(...)") && msg.contains("binomial"),
            "error should explain linkwiggle is binomial-only, got: {msg}"
        );
    }

    #[test]
    fn timewiggle_still_accepted_in_survival_formula() {
        // Guard must not regress the legitimate survival path: a Surv(...)
        // response still consumes timewiggle(...) without hitting the
        // non-survival rejection. We assert it does not error with the
        // non-survival "only supported in the main survival formula" message.
        let data = load_survival_dataset();
        let result = materialize(
            "Surv(entry, exit, event) ~ x + timewiggle(internal_knots=2)",
            &data,
            &FitConfig::default(),
        );
        if let Err(err) = result {
            let msg = err.to_string();
            assert!(
                !(msg.contains("timewiggle(...)") && msg.contains("meaningless")),
                "survival timewiggle wrongly rejected by the non-survival guard: {msg}"
            );
        }
    }

    // ---- #430 location-scale wiggle-pilot unification: parity tests ---------
    //
    // The Gaussian and binomial location-scale model entry points are now thin
    // adapters over the single `fit_location_scale_with_optional_wiggle` engine.
    // The tests below pin that the unified engine reproduces, coefficient for
    // coefficient, the exact per-family reference sequence it replaced — both
    // with and without a wiggle config — so the deslop cannot silently change
    // any fitted result. The reference replays the *old* hand-rolled flow
    // (pilot fit → select link-wiggle basis from the pilot → refit with that
    // basis → extract `beta_link_wiggle` from block 2) directly against the
    // family functions, with no shared code path with the engine other than
    // those leaf family functions.

    fn gaussian_location_scale_dataset() -> Dataset {
        // A mildly heteroscedastic, monotone-in-x signal with enough rows for a
        // stable mean+scale fit and a small wiggle basis.
        let n = 48usize;
        let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
        for i in 0..n {
            let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            // Deterministic, smooth response; the σ-model is intercept-only so
            // the test stays small while still exercising both blocks.
            let y = 0.7 * x + 0.3 * (1.3 * x).sin();
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
            ]));
        }
        crate::inference::data::encode_recordswith_inferred_schema(
            vec!["y".to_string(), "x".to_string()],
            records,
        )
        .expect("encode gaussian location-scale dataset")
    }

    fn binomial_location_scale_dataset() -> Dataset {
        // Balanced 0/1 response with a clear monotone gradient in x so the
        // threshold/log-σ blocks are well posed.
        let n = 60usize;
        let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
        for i in 0..n {
            let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            let y = if i % 2 == 0 { 1.0 } else { 0.0 };
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
            ]));
        }
        crate::inference::data::encode_recordswith_inferred_schema(
            vec!["y".to_string(), "x".to_string()],
            records,
        )
        .expect("encode binomial location-scale dataset")
    }

    fn small_wiggle_cfg() -> LinkWiggleConfig {
        LinkWiggleConfig {
            degree: 3,
            num_internal_knots: 3,
            penalty_orders: vec![2],
            double_penalty: false,
        }
    }

    fn assert_block_states_match(label: &str, lhs: &UnifiedFitResult, rhs: &UnifiedFitResult) {
        assert_eq!(
            lhs.block_states.len(),
            rhs.block_states.len(),
            "{label}: block count mismatch (engine {} vs reference {})",
            lhs.block_states.len(),
            rhs.block_states.len()
        );
        for (i, (a, b)) in lhs
            .block_states
            .iter()
            .zip(rhs.block_states.iter())
            .enumerate()
        {
            assert_eq!(
                a.beta.len(),
                b.beta.len(),
                "{label}: block {i} coefficient length mismatch"
            );
            for (j, (&av, &bv)) in a.beta.iter().zip(b.beta.iter()).enumerate() {
                // The engine and reference share the same leaf family functions
                // and feed them identical inputs, so the fitted coefficients
                // must agree to full numerical precision — this is a refactor,
                // not an approximation. A loose tolerance here would let a real
                // orchestration bug slip through, so the bound stays at the
                // bit-noise floor of an exact replay.
                assert!(
                    (av - bv).abs() <= 1e-12 * (1.0 + bv.abs()),
                    "{label}: block {i} coef {j} diverged: engine {av:.17e} vs reference {bv:.17e}"
                );
            }
        }
    }

    fn assert_beta_link_wiggle_match(
        label: &str,
        engine: &Option<Vec<f64>>,
        reference: &Option<Vec<f64>>,
    ) {
        match (engine, reference) {
            (Some(e), Some(r)) => {
                assert_eq!(
                    e.len(),
                    r.len(),
                    "{label}: beta_link_wiggle length mismatch (engine {} vs reference {})",
                    e.len(),
                    r.len()
                );
                for (j, (&ev, &rv)) in e.iter().zip(r.iter()).enumerate() {
                    // Same exact-replay floor as the block-state comparison: the
                    // engine reads block 2 off the very fit the reference refit
                    // produced, so any divergence beyond bit noise is a bug.
                    assert!(
                        (ev - rv).abs() <= 1e-12 * (1.0 + rv.abs()),
                        "{label}: beta_link_wiggle coef {j} diverged: \
                         engine {ev:.17e} vs reference {rv:.17e}"
                    );
                }
            }
            (None, None) => {}
            (e, r) => panic!(
                "{label}: beta_link_wiggle presence mismatch (engine is_some={}, reference is_some={})",
                e.is_some(),
                r.is_some()
            ),
        }
    }

    #[test]
    fn gaussian_location_scale_engine_matches_reference_flow() {
        let data = gaussian_location_scale_dataset();
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        };
        let materialized =
            materialize("y ~ x", &data, &config).expect("gaussian location-scale materialization");
        let FitRequest::GaussianLocationScale(request) = materialized.request else {
            panic!("expected a Gaussian location-scale request");
        };
        let GaussianLocationScaleFitRequest {
            data: req_data,
            spec,
            options,
            kappa_options,
            ..
        } = request;

        // --- no-wiggle parity ------------------------------------------------
        let engine_plain = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: None,
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine gaussian no-wiggle fit");
        let reference_plain =
            fit_gaussian_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference gaussian no-wiggle fit");
        assert_block_states_match(
            "gaussian/no-wiggle",
            &engine_plain.fit.fit,
            &reference_plain.fit,
        );
        assert!(engine_plain.wiggle_knots.is_none());
        assert!(engine_plain.wiggle_degree.is_none());
        assert!(engine_plain.beta_link_wiggle.is_none());

        // --- wiggle parity ---------------------------------------------------
        let wiggle_cfg = small_wiggle_cfg();
        let engine_wiggle = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: Some(wiggle_cfg.clone()),
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine gaussian wiggle fit");

        // Reference: the exact pre-unification hand-rolled sequence.
        let ref_pilot =
            fit_gaussian_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference gaussian pilot");
        let ref_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            &ref_pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )
        .expect("reference gaussian wiggle basis selection");
        let ref_solved = fit_gaussian_location_scale_terms_with_selected_wiggle(
            req_data,
            spec.clone(),
            ref_basis,
            &options,
            &kappa_options,
        )
        .expect("reference gaussian wiggle refit");

        assert_block_states_match(
            "gaussian/wiggle",
            &engine_wiggle.fit.fit,
            &ref_solved.fit.fit,
        );
        assert_eq!(
            engine_wiggle.wiggle_degree,
            Some(ref_solved.wiggle_degree),
            "gaussian wiggle degree must match the reference refit"
        );
        let engine_knots = engine_wiggle
            .wiggle_knots
            .as_ref()
            .expect("engine gaussian wiggle knots present");
        assert_eq!(
            engine_knots.len(),
            ref_solved.wiggle_knots.len(),
            "gaussian wiggle knot count must match the reference refit"
        );
        for (k, (&ek, &rk)) in engine_knots
            .iter()
            .zip(ref_solved.wiggle_knots.iter())
            .enumerate()
        {
            assert!(
                (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
                "gaussian wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
            );
        }
        // `beta_link_wiggle` is block 2 of the refit; the engine must extract it
        // exactly as the reference would read it off the same fit.
        let ref_beta_link_wiggle = ref_solved
            .fit
            .fit
            .block_states
            .get(2)
            .map(|b| b.beta.to_vec());
        assert_beta_link_wiggle_match(
            "gaussian",
            &engine_wiggle.beta_link_wiggle,
            &ref_beta_link_wiggle,
        );
        assert!(
            engine_wiggle.beta_link_wiggle.is_some(),
            "a wiggle refit must populate beta_link_wiggle (block 2 present)"
        );
    }

    #[test]
    fn binomial_location_scale_engine_matches_reference_flow() {
        let data = binomial_location_scale_dataset();
        let config = FitConfig {
            family: Some("binomial".to_string()),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        };
        let materialized =
            materialize("y ~ x", &data, &config).expect("binomial location-scale materialization");
        let FitRequest::BinomialLocationScale(request) = materialized.request else {
            panic!("expected a binomial location-scale request");
        };
        let BinomialLocationScaleFitRequest {
            data: req_data,
            spec,
            options,
            kappa_options,
            ..
        } = request;

        // --- no-wiggle parity ------------------------------------------------
        let engine_plain = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: None,
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine binomial no-wiggle fit");
        let reference_plain =
            fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference binomial no-wiggle fit");
        assert_block_states_match(
            "binomial/no-wiggle",
            &engine_plain.fit.fit,
            &reference_plain.fit,
        );
        assert!(engine_plain.wiggle_knots.is_none());
        assert!(engine_plain.wiggle_degree.is_none());
        assert!(engine_plain.beta_link_wiggle.is_none());

        // --- wiggle parity ---------------------------------------------------
        let wiggle_cfg = small_wiggle_cfg();
        let engine_wiggle = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
            data: req_data,
            spec: spec.clone(),
            wiggle: Some(wiggle_cfg.clone()),
            options: options.clone(),
            kappa_options: kappa_options.clone(),
        })
        .expect("engine binomial wiggle fit");

        // Reference: the exact pre-unification hand-rolled sequence, including
        // the binomial-only link compatibility guard.
        require_inverse_link_supports_joint_wiggle(
            &spec.link_kind,
            "binomial location-scale link wiggle",
        )
        .expect("logit link supports joint wiggle");
        let ref_pilot =
            fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
                .expect("reference binomial pilot");
        let ref_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            &ref_pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )
        .expect("reference binomial wiggle basis selection");
        let ref_solved = fit_binomial_location_scale_terms_with_selected_wiggle(
            req_data,
            spec.clone(),
            ref_basis,
            &options,
            &kappa_options,
        )
        .expect("reference binomial wiggle refit");

        assert_block_states_match(
            "binomial/wiggle",
            &engine_wiggle.fit.fit,
            &ref_solved.fit.fit,
        );
        assert_eq!(
            engine_wiggle.wiggle_degree,
            Some(ref_solved.wiggle_degree),
            "binomial wiggle degree must match the reference refit"
        );
        let engine_knots = engine_wiggle
            .wiggle_knots
            .as_ref()
            .expect("engine binomial wiggle knots present");
        assert_eq!(
            engine_knots.len(),
            ref_solved.wiggle_knots.len(),
            "binomial wiggle knot count must match the reference refit"
        );
        for (k, (&ek, &rk)) in engine_knots
            .iter()
            .zip(ref_solved.wiggle_knots.iter())
            .enumerate()
        {
            assert!(
                (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
                "binomial wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
            );
        }
        let ref_beta_link_wiggle = ref_solved
            .fit
            .fit
            .block_states
            .get(2)
            .map(|b| b.beta.to_vec());
        assert_beta_link_wiggle_match(
            "binomial",
            &engine_wiggle.beta_link_wiggle,
            &ref_beta_link_wiggle,
        );
        assert!(
            engine_wiggle.beta_link_wiggle.is_some(),
            "a wiggle refit must populate beta_link_wiggle (block 2 present)"
        );
    }
}
