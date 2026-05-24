use crate::custom_family::{
    BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix, fit_custom_family,
    fit_custom_family_with_rho_prior,
};
use crate::estimate::{
    AdaptiveRegularizationOptions, EstimationError, FitOptions, FittedLinkState, UnifiedFitResult,
};
use crate::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    fit_bernoulli_marginal_slope_terms,
};
use crate::families::gamlss::{
    BinomialLocationScaleFitResult, BinomialLocationScaleTermSpec, BlockwiseTermFitResult,
    BlockwiseTermFitResultParts, GaussianLocationScaleFitResult, GaussianLocationScaleTermSpec,
    WiggleBlockConfig, fit_binomial_location_scale_terms,
    fit_binomial_location_scale_terms_with_selected_wiggle,
    fit_binomial_mean_wiggle_terms_with_selected_basis, fit_gaussian_location_scale_terms,
    fit_gaussian_location_scale_terms_with_selected_wiggle,
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
use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::solver::latent_cache::LatentRetractionRegistry;
use crate::solver::riemannian_retraction::{ProductRetraction, RetractionKind};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, CoefficientGroupSpec, SpatialLengthScaleOptimizationOptions,
    StandardLatentCoordConfig, TermCollectionDesign, TermCollectionSpec,
    build_term_collection_design, fit_term_collectionwith_latent_coord_optimization,
    fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use crate::terms::latent_coord::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};
use crate::terms::{
    ARDPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, RowPrecisionPriorPenalty,
    BlockOrthogonalityPenalty, BlockSparsityPenalty, DifferenceOpKind, GumbelTemperatureSchedule,
    IBPAssignmentPenalty, IsometryPenalty, IvaeRidgeMeanGauge, NuclearNormPenalty, OrthogonalityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice, ScadMcpPenalty,
    SoftmaxAssignmentSparsityPenalty, ScalarWeightSchedule, ScheduleKind, SparsityPenalty,
    TotalVariationPenalty,
};
use crate::survival::PenaltyBlock;
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodFamily, LinkFunction, MixtureLinkSpec, SasLinkSpec,
    WigglePenaltyConfig,
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
use serde_json::Value as JsonValue;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowError::InvalidConfig { reason }
            | WorkflowError::SchemaMismatch { reason }
            | WorkflowError::MissingDependency { reason }
            | WorkflowError::IntegrationFailed { reason } => f.write_str(reason),
            WorkflowError::FormulaDsl { context, source } => write!(f, "{context}: {source}"),
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
            | WorkflowError::IntegrationFailed { .. } => None,
        }
    }
}

impl From<WorkflowError> for String {
    fn from(err: WorkflowError) -> String {
        err.to_string()
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

#[derive(Clone, Debug)]
pub struct LinkWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

#[derive(Clone, Debug)]
pub struct StandardBinomialWiggleConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleConfig,
}

pub struct StandardFitRequest<'a> {
    pub data: Array2<f64>,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodFamily,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleConfig>,
    pub wiggle_options: Option<BlockwiseFitOptions>,
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
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    SurvivalTransformation(SurvivalTransformationFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest<'a>),
    LatentSurvival(LatentSurvivalFitRequest<'a>),
    LatentBinary(LatentBinaryFitRequest<'a>),
    TransformationNormal(TransformationNormalFitRequest<'a>),
}

impl<'a> FitRequest<'a> {
    /// Short stable string identifying the family variant. Used as one
    /// segment of the warm-start cache key — every fit of this family
    /// shape shares the same family tag, so a hierarchical lookup can
    /// drop trailing key segments to find near-match seeds from prior
    /// fits of the same family on related data.
    pub const fn family_tag(&self) -> &'static str {
        match self {
            FitRequest::Standard(_) => "standard",
            FitRequest::GaussianLocationScale(_) => "gaussian-location-scale",
            FitRequest::BinomialLocationScale(_) => "binomial-location-scale",
            FitRequest::SurvivalLocationScale(_) => "survival-location-scale",
            FitRequest::SurvivalTransformation(_) => "survival-transformation",
            FitRequest::BernoulliMarginalSlope(_) => "bernoulli-marginal-slope",
            FitRequest::SurvivalMarginalSlope(_) => "survival-marginal-slope",
            FitRequest::LatentSurvival(_) => "latent-survival",
            FitRequest::LatentBinary(_) => "latent-binary",
            FitRequest::TransformationNormal(_) => "transformation-normal",
        }
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
        // Family-shape hash: a 64-bit StableHasher-derived integer that
        // captures dimensions / formula structure unique to this family
        // variant. Two fits with identical family-shape hashes have
        // compatible θ shapes and can warm-start from each other.
        let mut shape = crate::solver::persistent_warm_start::StableHasher::new();
        match self {
            FitRequest::Standard(req) => {
                shape.write_str("standard");
                shape.write_str(&format!("{:?}", req.family));
                shape.write_usize(req.y.len());
                shape.write_usize(req.data.ncols());
            }
            FitRequest::GaussianLocationScale(req) => {
                shape.write_str("gauss-ls");
                shape.write_usize(req.spec.y.len());
                shape.write_usize(req.data.ncols());
            }
            FitRequest::BinomialLocationScale(req) => {
                shape.write_str("binom-ls");
                shape.write_usize(req.spec.y.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.link_kind));
            }
            FitRequest::SurvivalLocationScale(req) => {
                shape.write_str("surv-ls");
                shape.write_usize(req.spec.age_entry.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.inverse_link));
            }
            FitRequest::SurvivalTransformation(req) => {
                shape.write_str("surv-tn");
                shape.write_usize(req.spec.age_entry.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.likelihood_mode));
                shape.write_str(&req.spec.time_build.basisname);
            }
            FitRequest::BernoulliMarginalSlope(req) => {
                shape.write_str("bern-ms");
                shape.write_usize(req.spec.y.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.base_link));
            }
            FitRequest::SurvivalMarginalSlope(req) => {
                shape.write_str("surv-ms");
                shape.write_usize(req.spec.age_entry.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.base_link));
                shape.write_str(&format!("{:?}", req.spec.frailty));
            }
            FitRequest::LatentSurvival(req) => {
                shape.write_str("lat-surv");
                shape.write_usize(req.spec.age_entry.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.frailty));
            }
            FitRequest::LatentBinary(req) => {
                shape.write_str("lat-bin");
                shape.write_usize(req.spec.age_entry.len());
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.frailty));
            }
            FitRequest::TransformationNormal(req) => {
                shape.write_str("tn");
                shape.write_usize(req.response.len());
                shape.write_usize(req.data.ncols());
            }
        }
        let shape_hash = shape.finish_hex();
        let (nrows, ncols) = match self {
            FitRequest::Standard(req) => (req.y.len(), req.data.ncols()),
            FitRequest::GaussianLocationScale(req) => (req.spec.y.len(), req.data.ncols()),
            FitRequest::BinomialLocationScale(req) => (req.spec.y.len(), req.data.ncols()),
            FitRequest::SurvivalLocationScale(req) => (req.spec.age_entry.len(), req.data.ncols()),
            FitRequest::SurvivalTransformation(req) => (req.spec.age_entry.len(), req.data.ncols()),
            FitRequest::BernoulliMarginalSlope(req) => (req.spec.y.len(), req.data.ncols()),
            FitRequest::SurvivalMarginalSlope(req) => (req.spec.age_entry.len(), req.data.ncols()),
            FitRequest::LatentSurvival(req) => (req.spec.age_entry.len(), req.data.ncols()),
            FitRequest::LatentBinary(req) => (req.spec.age_entry.len(), req.data.ncols()),
            FitRequest::TransformationNormal(req) => (req.response.len(), req.data.ncols()),
        };
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
        let mut shape = crate::solver::persistent_warm_start::StableHasher::new();
        match self {
            FitRequest::Standard(req) => {
                shape.write_str("standard-seed");
                shape.write_str(&format!("{:?}", req.family));
                shape.write_usize(req.data.ncols());
            }
            FitRequest::GaussianLocationScale(req) => {
                shape.write_str("gauss-ls-seed");
                shape.write_usize(req.data.ncols());
            }
            FitRequest::BinomialLocationScale(req) => {
                shape.write_str("binom-ls-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.link_kind));
            }
            FitRequest::SurvivalLocationScale(req) => {
                shape.write_str("surv-ls-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.inverse_link));
            }
            FitRequest::SurvivalTransformation(req) => {
                shape.write_str("surv-tn-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.likelihood_mode));
                shape.write_str(&req.spec.time_build.basisname);
            }
            FitRequest::BernoulliMarginalSlope(req) => {
                shape.write_str("bern-ms-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.base_link));
            }
            FitRequest::SurvivalMarginalSlope(req) => {
                shape.write_str("surv-ms-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.spec.base_link));
                shape.write_str(&format!("{:?}", req.spec.frailty));
            }
            FitRequest::LatentSurvival(req) => {
                shape.write_str("lat-surv-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.frailty));
            }
            FitRequest::LatentBinary(req) => {
                shape.write_str("lat-bin-seed");
                shape.write_usize(req.data.ncols());
                shape.write_str(&format!("{:?}", req.frailty));
            }
            FitRequest::TransformationNormal(req) => {
                shape.write_str("tn-seed");
                shape.write_usize(req.data.ncols());
            }
        }
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
        match self {
            FitRequest::Standard(_) => {}
            FitRequest::GaussianLocationScale(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::BinomialLocationScale(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::SurvivalLocationScale(req) => {
                req.spec.cache_mirror_sessions.push(mirror);
            }
            FitRequest::SurvivalTransformation(_) => {
                // SurvivalTransformation uses an internal WorkingModelPirlsOptions
                // path with its own warm-start key. Mirror finalize is a no-op
                // here — the path's own cache fires on exact match.
            }
            FitRequest::BernoulliMarginalSlope(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::SurvivalMarginalSlope(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::LatentSurvival(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::LatentBinary(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
            FitRequest::TransformationNormal(req) => {
                req.options.cache_mirror_sessions.push(mirror);
            }
        }
    }

    /// Attach a warm-start cache session to this request.
    ///
    /// Threads the session into the variant's BlockwiseFitOptions
    /// (`request.options.cache_session`) or top-level `cache_session`
    /// field, whichever is the variant's natural slot. Idempotent —
    /// existing sessions are not overwritten so callers can pre-attach.
    pub fn attach_cache_session(&mut self, session: std::sync::Arc<crate::cache::Session>) {
        match self {
            FitRequest::Standard(_) => {
                // The standard REML path opens its own session inside the
                // outer optimizer via `reml_state.outer_cache_session()`
                // (see `solver/estimate.rs:2701`). The session here would
                // be a duplicate keyed on the same fingerprint, so we
                // skip it to avoid double-checkpointing.
            }
            FitRequest::GaussianLocationScale(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::BinomialLocationScale(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::SurvivalLocationScale(req) => {
                // The request-level slot is mirrored into the spec slot
                // so the family fit fn sees the session when it
                // constructs its internal BlockwiseFitOptions.
                if req.cache_session.is_none() {
                    req.cache_session = Some(session.clone());
                }
                if req.spec.cache_session.is_none() {
                    req.spec.cache_session = Some(session);
                }
            }
            FitRequest::SurvivalTransformation(req) => {
                // SurvivalTransformation uses WorkingModelPirlsOptions
                // (different mechanism) for its inner solve; the cache
                // session here is parked at the request level so future
                // wiring through `persistent_survival_transformation_key`
                // can be unified. For now, the path's own
                // `persistent_survival_transformation_key` mechanism
                // handles exact-match warm-start.
                req.cache_session.get_or_insert(session);
            }
            FitRequest::BernoulliMarginalSlope(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::SurvivalMarginalSlope(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::LatentSurvival(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::LatentBinary(req) => {
                req.options.cache_session.get_or_insert(session);
            }
            FitRequest::TransformationNormal(req) => {
                req.options.cache_session.get_or_insert(session);
            }
        }
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
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    SurvivalTransformation(SurvivalTransformationFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitResult),
    LatentSurvival(LatentSurvivalTermFitResult),
    LatentBinary(LatentBinaryTermFitResult),
    TransformationNormal(TransformationNormalFitResult),
}

fn resolved_wiggle_inverse_link(
    family: LikelihoodFamily,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, String> {
    let resolved = match fit.fitted_link_state(family).map_err(|e| e.to_string())? {
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
        if !request.coefficient_groups.is_empty() || !request.penalty_block_gamma_priors.is_empty() {
            return Err("latent-coordinate standard fits do not support coefficient_groups or penalty_block_gamma_priors in the same request".to_string());
        }
        fit_term_collectionwith_latent_coord_optimization(
            request.data.view(),
            request.y.clone(),
            request.weights.clone(),
            request.offset.clone(),
            &request.spec,
            latent_coord,
            request.family,
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
            request.family,
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
            request.family,
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
    let wiggle_options = request
        .wiggle_options
        .ok_or_else(|| "standard wiggle workflow requires blockwise wiggle options".to_string())?;
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(request.family, &result.fit, &wiggle.link_kind)?;
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

    let solved = fit_binomial_mean_wiggle_terms_with_selected_basis(
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
    )?;

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

fn fit_gaussian_location_scale_model(
    request: GaussianLocationScaleFitRequest<'_>,
) -> Result<GaussianLocationScaleFitResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        let pilot = fit_gaussian_location_scale_terms(
            request.data,
            GaussianLocationScaleTermSpec {
                y: request.spec.y.clone(),
                weights: request.spec.weights.clone(),
                meanspec: request.spec.meanspec.clone(),
                log_sigmaspec: request.spec.log_sigmaspec.clone(),
                mean_offset: request.spec.mean_offset.clone(),
                log_sigma_offset: request.spec.log_sigma_offset.clone(),
            },
            &request.options,
            &request.kappa_options,
        )?;
        let selected_wiggle_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            &pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        let solved = fit_gaussian_location_scale_terms_with_selected_wiggle(
            request.data,
            request.spec,
            selected_wiggle_basis,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(GaussianLocationScaleFitResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            beta_link_wiggle,
        })
    } else {
        let fit = fit_gaussian_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        })
    }
}

fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleFitRequest<'_>,
) -> Result<BinomialLocationScaleFitResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        require_inverse_link_supports_joint_wiggle(
            &request.spec.link_kind,
            "binomial location-scale link wiggle",
        )?;
        let pilot = fit_binomial_location_scale_terms(
            request.data,
            BinomialLocationScaleTermSpec {
                y: request.spec.y.clone(),
                weights: request.spec.weights.clone(),
                link_kind: request.spec.link_kind.clone(),
                thresholdspec: request.spec.thresholdspec.clone(),
                log_sigmaspec: request.spec.log_sigmaspec.clone(),
                threshold_offset: request.spec.threshold_offset.clone(),
                log_sigma_offset: request.spec.log_sigma_offset.clone(),
            },
            &request.options,
            &request.kappa_options,
        )?;
        let selected_wiggle_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            &pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        let solved = fit_binomial_location_scale_terms_with_selected_wiggle(
            request.data,
            request.spec,
            selected_wiggle_basis,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(BinomialLocationScaleFitResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            beta_link_wiggle,
        })
    } else {
        let solved = fit_binomial_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(BinomialLocationScaleFitResult {
            fit: solved,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        })
    }
}

fn survival_working_reml_score(state: &crate::pirls::WorkingState) -> f64 {
    0.5 * (state.deviance + state.penalty_term)
}

fn fitted_weibull_baseline_from_linear_time_beta(
    beta: &Array1<f64>,
) -> Option<crate::families::survival_construction::SurvivalBaselineConfig> {
    if beta.len() < 2 {
        return None;
    }
    let shape = beta[1];
    if !shape.is_finite() || shape <= 0.0 {
        return None;
    }
    let scale = (-beta[0] / shape).exp();
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
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

fn survival_unified_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    summary: &crate::pirls::WorkingModelPirlsResult,
    state: &crate::pirls::WorkingState,
) -> Result<UnifiedFitResult, String> {
    let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
    let reml_score = survival_working_reml_score(state);
    crate::estimate::validate_all_finite("survival fit beta", beta.iter().copied())?;
    crate::estimate::validate_all_finite("survival fit lambdas", lambdas.iter().copied())?;
    crate::estimate::ensure_finite_scalar("survival fit log_likelihood", state.log_likelihood)?;
    crate::estimate::ensure_finite_scalar("survival fit deviance", state.deviance)?;
    crate::estimate::ensure_finite_scalar("survival fit penalty", state.penalty_term)?;
    crate::estimate::ensure_finite_scalar("survival fit reml_score", reml_score)?;
    crate::estimate::ensure_finite_scalar("survival fit gradient_norm", summary.lastgradient_norm)?;
    crate::estimate::ensure_finite_scalar("survival fit max_abs_eta", summary.max_abs_eta)?;

    UnifiedFitResult::try_from_parts(crate::estimate::UnifiedFitResultParts {
        blocks: vec![crate::estimate::FittedBlock {
            beta: beta.clone(),
            role: crate::estimate::BlockRole::Mean,
            edf: 0.0,
            lambdas: lambdas.clone(),
        }],
        log_lambdas,
        lambdas,
        likelihood_family: Some(LikelihoodFamily::RoystonParmar),
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
        inference: None,
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

fn fit_cause_specific_survival_transformation_custom(
    spec: &SurvivalTransformationTermSpec,
    resolvedspec: TermCollectionSpec,
    baseline_cfg: crate::families::survival_construction::SurvivalBaselineConfig,
    prepared: PreparedWorkflowSurvivalTimeStack,
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
    let dense_time_derivative = prepared.time_design_derivative.to_dense();
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
            initial_log_lambdas[penalty_idx] = block.lambda.max(1e-300).ln();
        }
        let beta_start = beta0_flat.slice(s![cause * p..(cause + 1) * p]).to_owned();
        block_specs.push(ParameterBlockSpec {
            name: format!("time_cause_{}", cause + 1),
            design: crate::matrix::DesignMatrix::from(x_exit.clone()),
            offset: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            penalties,
            nullspace_dims,
            initial_log_lambdas,
            initial_beta: Some(beta_start),
        });
    }

    let family = crate::survival::CauseSpecificRoystonParmarFamily::new(family_blocks)?;
    let fit_options = BlockwiseFitOptions {
        compute_covariance: false,
        ..Default::default()
    };
    let rho_prior =
        cause_specific_survival_rho_prior(penalty_blocks.len(), penalty_block_gamma_priors)?;
    let mut fit = if matches!(rho_prior, crate::types::RhoPrior::Flat) {
        fit_custom_family(&family, &block_specs, &fit_options)
    } else {
        fit_custom_family_with_rho_prior(&family, &block_specs, &fit_options, rho_prior)
    }
    .map_err(|err| format!("cause-specific survival custom-family fit failed: {err}"))?;
    fit.likelihood_family = Some(LikelihoodFamily::RoystonParmar);
    let time_basis = crate::families::survival_construction::SavedSurvivalTimeBasis::from_build(
        &spec.time_build,
        spec.time_anchor,
    );
    Ok(SurvivalTransformationFitResult {
        fit,
        resolvedspec,
        baseline_cfg,
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

fn hash_workflow_array_view(
    hasher: &mut crate::solver::persistent_warm_start::StableHasher,
    array: ArrayView1<'_, f64>,
) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_f64(value);
    }
}

fn hash_workflow_u8_array(
    hasher: &mut crate::solver::persistent_warm_start::StableHasher,
    array: ArrayView1<'_, u8>,
) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_usize(usize::from(value));
    }
}

fn hash_workflow_array2(
    hasher: &mut crate::solver::persistent_warm_start::StableHasher,
    array: ArrayView2<'_, f64>,
) {
    hasher.write_usize(array.nrows());
    hasher.write_usize(array.ncols());
    for row in array.rows() {
        for &value in row {
            hasher.write_f64(value);
        }
    }
}

fn hash_workflow_design_matrix(
    hasher: &mut crate::solver::persistent_warm_start::StableHasher,
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
        .map(|block| block.lambda.max(1e-300).ln())
        .collect()
}

fn persistent_survival_transformation_key(
    spec: &SurvivalTransformationTermSpec,
    baseline_cfg: &crate::families::survival_construction::SurvivalBaselineConfig,
    dense_cov_design: ArrayView2<'_, f64>,
    prepared: &PreparedWorkflowSurvivalTimeStack,
    penalty_blocks: &[crate::survival::PenaltyBlock],
    opts: &crate::pirls::WorkingModelPirlsOptions,
    n_cols: usize,
) -> String {
    let mut hasher = crate::solver::persistent_warm_start::StableHasher::new();
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
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_derivative);
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
            let prepared = prepare_workflow_survival_time_stack(
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
            let dense_time_derivative = prepared.time_design_derivative.to_dense();
            let mut model =
                crate::families::royston_parmar::working_model_from_time_covariateshared(
                    PenaltyBlocks::new(penalty_blocks.clone()),
                    MonotonicityPenalty { tolerance: 0.0 },
                    SurvivalSpec::Net,
                    crate::families::royston_parmar::RoystonParmarSharedTimeCovariateInputs {
                        age_entry: spec.age_entry.view(),
                        age_exit: spec.age_exit.view(),
                        event_target: spec.event_target.view(),
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
            ))
        };

    if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        baseline_cfg = optimize_survival_baseline_config(
            &baseline_cfg,
            "workflow survival transformation baseline",
            |candidate| {
                let (_, _, beta0, structural_lower_bounds, mut model) =
                    build_working_model(candidate)?;
                let opts = crate::pirls::WorkingModelPirlsOptions {
                    max_iterations: 400,
                    convergence_tolerance: 1e-6,
                    adaptive_kkt_tolerance: None,
                    max_step_halving: 40,
                    min_step_size: 1e-12,
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

    let (prepared, penalty_blocks, beta0, structural_lower_bounds, mut model) =
        build_working_model(&baseline_cfg)?;
    if cause_count > 1 || !spec.penalty_block_gamma_priors.is_empty() {
        return fit_cause_specific_survival_transformation_custom(
            &spec,
            resolvedspec,
            baseline_cfg,
            prepared,
            &dense_cov_design,
            penalty_blocks,
            beta0,
            exact_derivative_guard,
            &spec.penalty_block_gamma_priors,
        );
    }
    let opts = crate::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        adaptive_kkt_tolerance: None,
        max_step_halving: 40,
        min_step_size: 1e-12,
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
            fitted_weibull_baseline_from_linear_time_beta(&time_beta).ok_or_else(|| {
                "failed to recover fitted Weibull scale/shape from the linear time coefficients"
                    .to_string()
            })?
        } else {
            baseline_cfg
        };
    let fit = survival_unified_fit_result(beta, lambdas, &summary, &state)?;

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
    optimize_survival_baseline_config_with_gradient, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config, positive_survival_time_seed,
    require_structural_survival_time_basis, resolve_survival_time_anchor_value,
    resolved_survival_time_basis_config_from_build, survival_derivative_guard_for_likelihood,
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
    build_termspec, column_map_with_alias, enable_scale_dimensions, resolve_role_col,
};

/// Non-formula configuration for model fitting. All fields have sensible defaults.
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Family: "gaussian", "binomial", "poisson", "negative-binomial", "gamma",
    /// or None for auto-detect.
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
            scale_dimensions: false,
            adaptive_regularization: None,
            ridge_lambda: 1e-6,
            transformation_normal: false,
            firth: false,
            gpu_policy: crate::gpu::GpuPolicy::Auto,
            resource_policy: None,
            group_metadata: None,
            coefficient_groups: Vec::new(),
            penalty_block_gamma_priors: Vec::new(),
            latents: None,
            analytic_penalties: None,
            topology_auto_selector: None,
        }
    }
}

/// Resolve the [`crate::resource::ResourcePolicy`] backing term construction
/// for a given [`FitConfig`] + dataset.
///
/// If the caller hasn't supplied an explicit policy override, derive one from
/// the shape of the problem via
/// [`crate::resource::ResourcePolicy::for_problem`]. At biobank scale (n_rows
/// >= 100k or the marginal-slope biobank path active) this returns
/// `analytic_operator_required` so that any silent dense materialization in
/// the term-construction layer fails fast rather than allocating tens of GiB;
/// at small scale it falls through to the permissive default-library policy
/// so that non-operator bases still build cleanly.
///
/// `p_estimate = 0` because the per-block coefficient count isn't known until
/// the spec has been built; the n_rows and hints triggers are sufficient to
/// flip strict mode for every shape that needs it.
fn resolved_resource_policy(
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
        marginal_slope_biobank_active: config.logslope_formula.is_some()
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
) -> Result<MaterializedModel<'a>, String> {
    crate::gpu::configure_global_policy(config.gpu_policy);
    let parsed = parse_formula(formula)?;
    let col_map = data.column_map();

    if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        if config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with a Surv(...) response"
                    .to_string(),
            }
            .into());
        }
        materialize_survival(
            &parsed, data, &col_map, config, &entry_col, &exit_col, &event_col,
        )
        .map_err(|reason| WorkflowError::IntegrationFailed { reason })
    } else if config.transformation_normal {
        if config.noise_formula.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with noise_formula".to_string(),
            }
            .into());
        }
        materialize_transformation_normal(&parsed, data, &col_map, config)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })
    } else if config.logslope_formula.is_some() || config.z_column.is_some() {
        materialize_bernoulli_marginal_slope(&parsed, data, &col_map, config)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })
    } else if config.noise_formula.is_some() {
        materialize_location_scale(&parsed, data, &col_map, config)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })
    } else {
        materialize_standard(&parsed, data, &col_map, config)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })
    }
}

/// Detect whether a response column is binary (0/1 only).
pub fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Resolve a family from an optional name, optional link choice, and response data.
pub fn resolve_family(
    family: Option<&str>,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
) -> Result<LikelihoodFamily, String> {
    let nb_theta = negative_binomial_theta.unwrap_or(1.0);
    if !nb_theta.is_finite() || nb_theta <= 0.0 {
        return Err(format!(
            "negative-binomial theta must be finite and > 0; got {nb_theta}"
        ));
    }
    let explicit = match family {
        Some(name) => {
            let resolved = match name.to_ascii_lowercase().as_str() {
                "gaussian" => LikelihoodFamily::GaussianIdentity,
                "binomial" | "binomial-logit" => LikelihoodFamily::BinomialLogit,
                "binomial-probit" => LikelihoodFamily::BinomialProbit,
                "binomial-cloglog" => LikelihoodFamily::BinomialCLogLog,
                "latent-cloglog-binomial" => LikelihoodFamily::BinomialLatentCLogLog,
                "poisson" => LikelihoodFamily::PoissonLog,
                "nb" | "negbin" | "negative_binomial" | "negative-binomial"
                | "negative-binomial-log" => {
                    LikelihoodFamily::NegativeBinomial { theta: nb_theta }
                }
                "beta" | "beta-logit" | "beta-regression" | "beta-regression-logit" => {
                    LikelihoodFamily::BetaLogit { phi: 1.0 }
                }
                "gamma" => LikelihoodFamily::GammaLog,
                other => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!("unknown family '{other}'"),
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
        let from_link = if choice.mixture_components.is_some() {
            LikelihoodFamily::BinomialMixture
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodFamily::GaussianIdentity,
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodFamily::PoissonLog
                    } else {
                        LikelihoodFamily::GammaLog
                    }
                }
                LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
                LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
                LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
                LinkFunction::Sas => LikelihoodFamily::BinomialSas,
                LinkFunction::BetaLogistic => LikelihoodFamily::BinomialBetaLogistic,
            }
        };
        if let Some(explicit_family) = explicit {
            let compatible_log_nb = matches!(
                (explicit_family, choice.link, choice.mixture_components.as_ref()),
                (LikelihoodFamily::NegativeBinomial { .. }, LinkFunction::Log, None)
            );
            if explicit_family != from_link && !compatible_log_nb {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!("family '{}' conflicts with link", explicit_family.name()),
                }
                .into());
            }
        }
        return Ok(explicit.unwrap_or(from_link));
    }

    if let Some(f) = explicit {
        return Ok(f);
    }

    // Auto-detect
    if is_binary_response(y) {
        Ok(LikelihoodFamily::BinomialLogit)
    } else {
        Ok(LikelihoodFamily::GaussianIdentity)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_termspec_with_geometry(
    terms: &[ParsedTerm],
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
    scale_dimensions: bool,
    policy: &crate::resource::ResourcePolicy,
) -> Result<TermCollectionSpec, String> {
    let mut spec = build_termspec(terms, data, col_map, inference_notes, policy)?;
    if scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    Ok(spec)
}

fn standard_adaptive_regularization_options(
    config: &FitConfig,
    spec: &TermCollectionSpec,
) -> Option<AdaptiveRegularizationOptions> {
    drop(spec);
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
        return Ok(InverseLink::Standard(LinkFunction::Probit));
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
        LinkFunction::Probit => Ok(InverseLink::Standard(LinkFunction::Probit)),
        other => Err(WorkflowError::InvalidConfig {
            reason: format!(
                "survival marginal-slope currently supports only link(type=probit), got {other:?}"
            ),
        }
        .into()),
    }
}

struct PreparedWorkflowSurvivalTimeStack {
    eta_offset_entry: Array1<f64>,
    eta_offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    unloaded_mass_entry: Array1<f64>,
    unloaded_mass_exit: Array1<f64>,
    unloaded_hazard_exit: Array1<f64>,
    time_design_entry: crate::matrix::DesignMatrix,
    time_design_exit: crate::matrix::DesignMatrix,
    time_design_derivative: crate::matrix::DesignMatrix,
    time_penalties: Vec<Array2<f64>>,
    time_nullspace_dims: Vec<usize>,
    timewiggle_block: Option<TimeWiggleBlockInput>,
}

fn prepare_workflow_survival_time_stack(
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
) -> Result<PreparedWorkflowSurvivalTimeStack, String> {
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
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_time_offsets_for_likelihood(
                age_entry,
                age_exit,
                baseline_cfg,
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
    let mut time_design_derivative = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(wiggle) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative,
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
    Ok(PreparedWorkflowSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative,
        time_penalties,
        time_nullspace_dims,
        timewiggle_block,
    })
}

fn resolve_continuous_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: &str,
    role: &str,
) -> Result<Array1<f64>, String> {
    let col_idx = resolve_role_col(col_map, column_name, role)?;
    let values = data.values.column(col_idx).to_owned();
    for (row_idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "{role} column '{column_name}' contains non-finite value at row {row_idx}: {value}"
                ),
            }
            .into());
        }
    }
    Ok(values)
}

pub fn resolve_offset_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, String> {
    let Some(column_name) = column_name else {
        return Ok(Array1::zeros(data.values.nrows()));
    };
    resolve_continuous_column(data, col_map, column_name, "offset")
}

pub fn resolve_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, String> {
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
            }
            .into());
        }
    }
    Ok(values)
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
    explicit_none_mode: bool,
}

#[derive(Clone)]
struct LatentPenaltyTarget {
    name: String,
    n: usize,
    d: usize,
}

fn latent_penalty_targets(
    latents: Option<&JsonValue>,
) -> Result<Vec<LatentPenaltyTarget>, String> {
    let Some(raw) = latents.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = raw
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut out = Vec::with_capacity(map.len());
    for (key, raw_block) in map {
        let obj = raw_block
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let name = obj
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
        out.push(LatentPenaltyTarget { name, n, d });
    }
    Ok(out)
}

fn penalty_target_for_descriptor<'a>(
    targets: &'a [LatentPenaltyTarget],
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<&'a LatentPenaltyTarget, String> {
    let raw = descriptor
        .get("target")
        .ok_or_else(|| format!("{context}.target is required"))?;
    if let Some(name) = raw.as_str() {
        return targets
            .iter()
            .find(|target| target.name == name)
            .ok_or_else(|| {
                format!(
                    "{context}.target references latent block {name:?}, but latents declares [{}]",
                    targets
                        .iter()
                        .map(|target| target.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    if let Some(index) = raw.as_u64() {
        return targets.get(index as usize).ok_or_else(|| {
            format!(
                "{context}.target references latent index {index}, but latents declares {} block(s)",
                targets.len()
            )
        });
    }
    Err(format!("{context}.target must be a latent block name or index"))
}

fn analytic_descriptor_f64(
    descriptor: &serde_json::Map<String, JsonValue>,
    key: &str,
    default: f64,
) -> Result<f64, String> {
    let value = descriptor
        .get(key)
        .and_then(JsonValue::as_f64)
        .unwrap_or(default);
    if !(value.is_finite() && value > 0.0) {
        return Err(format!("analytic penalty {key} must be finite and > 0"));
    }
    Ok(value)
}

fn analytic_descriptor_usize(
    descriptor: &serde_json::Map<String, JsonValue>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let value = descriptor
        .get(key)
        .and_then(JsonValue::as_u64)
        .unwrap_or(default as u64) as usize;
    if value == 0 {
        return Err(format!("analytic penalty {key} must be > 0"));
    }
    Ok(value)
}

fn analytic_descriptor_weight_value(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<(), String> {
    let Some(value) = descriptor.get("weight") else {
        return Ok(());
    };
    if value.as_str() == Some("auto") {
        return Ok(());
    }
    let Some(weight) = value.as_f64() else {
        return Err(format!(
            "{context}.weight must be 'auto' or a finite positive float"
        ));
    };
    if !(weight.is_finite() && weight > 0.0) {
        return Err(format!("{context}.weight must be finite and > 0"));
    }
    Ok(())
}

fn analytic_descriptor_weight_sequence(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<(), String> {
    let Some(value) = descriptor.get("weight") else {
        return Ok(());
    };
    if value.as_str() == Some("auto") {
        return Ok(());
    }
    let values = value
        .as_array()
        .ok_or_else(|| format!("{context}.weight must be 'auto' or a list of positive floats"))?;
    if values.is_empty() {
        return Err(format!("{context}.weight must have at least one entry"));
    }
    for (idx, raw) in values.iter().enumerate() {
        let Some(weight) = raw.as_f64() else {
            return Err(format!("{context}.weight[{idx}] must be a finite positive float"));
        };
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!("{context}.weight[{idx}] must be finite and > 0"));
        }
    }
    Ok(())
}

fn analytic_descriptor_difference_op(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<DifferenceOpKind, String> {
    let op = descriptor
        .get("difference_op")
        .and_then(JsonValue::as_str)
        .unwrap_or("forward_1d")
        .to_ascii_lowercase()
        .replace('-', "_");
    match op.as_str() {
        "forward_1d" => Ok(DifferenceOpKind::ForwardDiff1D),
        "graph_edges" => {
            let raw_edges = descriptor
                .get("edges")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| format!("{context}.edges is required for graph_edges"))?;
            let mut edges = Vec::with_capacity(raw_edges.len());
            for (edge_idx, raw_edge) in raw_edges.iter().enumerate() {
                let pair = raw_edge
                    .as_array()
                    .ok_or_else(|| format!("{context}.edges[{edge_idx}] must be a two-item list"))?;
                if pair.len() != 2 {
                    return Err(format!(
                        "{context}.edges[{edge_idx}] must contain exactly two row indices"
                    ));
                }
                let from = pair[0].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][0] must be a non-negative integer")
                })? as usize;
                let to = pair[1].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][1] must be a non-negative integer")
                })? as usize;
                edges.push((from, to));
            }
            Ok(DifferenceOpKind::GraphEdges(edges))
        }
        other => Err(format!(
            "{context}.difference_op must be forward_1d or graph_edges; got {other:?}"
        )),
    }
}

fn analytic_descriptor_weight_schedule(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<Option<ScalarWeightSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("weight_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.weight_schedule must be an object"))?;
    let w_start = schedule
        .get("w_start")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_start must be a finite number"))?;
    let w_end = schedule
        .get("w_end")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_end must be a finite number"))?;
    let kind_name = schedule
        .get("kind")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| format!("{context}.weight_schedule.kind is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let kind = match kind_name.as_str() {
        "geometric" => {
            let rate = schedule
                .get("rate")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    format!("{context}.weight_schedule.rate is required for geometric")
                })?;
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| {
                    format!("{context}.weight_schedule.steps is required for linear")
                })?;
            ScheduleKind::Linear {
                steps: steps as usize,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.weight_schedule.kind must be geometric, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = ScalarWeightSchedule::new(w_start, w_end, kind)
        .map_err(|err| format!("{context}.weight_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        parsed.iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.weight_schedule.iter_count must be a non-negative integer")
        })? as usize;
    }
    Ok(Some(parsed))
}

fn analytic_descriptor_temperature_schedule(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<Option<GumbelTemperatureSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("temperature_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.temperature_schedule must be an object"))?;
    let tau_start = schedule
        .get("tau_start")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.temperature_schedule.tau_start must be a finite number"))?;
    let tau_min = schedule
        .get("tau_min")
        .or_else(|| schedule.get("tau_end"))
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| {
            format!("{context}.temperature_schedule.tau_min must be a finite number")
        })?;
    let decay_name = schedule
        .get("decay")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| format!("{context}.temperature_schedule.decay is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let decay = match decay_name.as_str() {
        "geometric" | "exponential" => {
            let rate = schedule
                .get("rate")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.9);
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| {
                    format!("{context}.temperature_schedule.steps is required for linear")
                })?;
            ScheduleKind::Linear {
                steps: steps as usize,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.temperature_schedule.decay must be geometric, exponential, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = GumbelTemperatureSchedule::new(tau_start, tau_min, decay)
        .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        parsed.iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.temperature_schedule.iter_count must be a non-negative integer")
        })? as usize;
        parsed
            .validate()
            .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    }
    Ok(Some(parsed))
}

fn build_standard_latent_analytic_penalty_registry(
    latents: Option<&JsonValue>,
    penalties: Option<&JsonValue>,
) -> Result<AnalyticPenaltyRegistry, String> {
    let mut registry = AnalyticPenaltyRegistry::new();
    let Some(raw) = penalties.filter(|value| !value.is_null()) else {
        return Ok(registry);
    };
    let items = raw
        .as_array()
        .ok_or_else(|| "penalties must be a list of analytic penalty descriptors".to_string())?;
    let targets = latent_penalty_targets(latents)?;
    if !items.is_empty() && targets.is_empty() {
        return Err("penalties requires latents with at least one latent block".to_string());
    }
    for (idx, raw_item) in items.iter().enumerate() {
        let context = format!("penalties[{idx}]");
        let descriptor = raw_item
            .as_object()
            .ok_or_else(|| format!("{context} must be an object"))?;
        let target = penalty_target_for_descriptor(&targets, descriptor, &context)?;
        let slice = PsiSlice::full(target.n * target.d, Some(target.d));
        let kind = descriptor
            .get("kind")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("{context}.kind is required"))?
            .to_ascii_lowercase()
            .replace('-', "_");
        let weight_schedule = analytic_descriptor_weight_schedule(descriptor, &context)?;
        match kind.as_str() {
            "isometry" => {
                analytic_descriptor_weight_value(descriptor, &context)?;
                let penalty = IsometryPenalty::new_euclidean(slice, target.d);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Isometry(Arc::new(penalty)));
            }
            "ard" => {
                analytic_descriptor_weight_sequence(descriptor, &context)?;
                let penalty = ARDPenalty::new(slice, target.d);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Ard(Arc::new(penalty)));
            }
            "orthogonality" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty =
                    OrthogonalityPenalty::new(slice, target.d, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Orthogonality(Arc::new(penalty)));
            }
            "sparsity" => {
                analytic_descriptor_weight_value(descriptor, &context)?;
                let sparsity_kind = descriptor
                    .get("sparsity_kind")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("smooth_l1")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let eps = analytic_descriptor_f64(descriptor, "eps", 1.0e-3)?;
                let penalty = match sparsity_kind.as_str() {
                    "smooth_l1" | "smoothed_l1" => {
                        SparsityPenalty::smoothed_l1(PenaltyTier::Psi, eps)
                    }
                    "log" => SparsityPenalty::log(PenaltyTier::Psi, eps),
                    "hoyer" => Ok(SparsityPenalty::hoyer(PenaltyTier::Psi)),
                    other => Err(format!(
                        "{context}.sparsity_kind must be smooth_l1, hoyer, or log; got {other:?}"
                    )),
                }?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Sparsity(Arc::new(penalty)));
            }
            "scad_mcp" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let variant = descriptor
                    .get("variant")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("mcp")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let (variant, gamma_default) = match variant.as_str() {
                    "mcp" => (PenaltyConcavity::Mcp, 2.5),
                    "scad" => (PenaltyConcavity::Scad, 3.7),
                    other => {
                        return Err(format!(
                            "{context}.variant must be 'mcp' or 'scad'; got {other:?}"
                        ));
                    }
                };
                let gamma = analytic_descriptor_f64(descriptor, "gamma", gamma_default)?;
                let smoothing_eps =
                    analytic_descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = ScadMcpPenalty::new(
                    slice,
                    weight,
                    n_eff,
                    gamma,
                    smoothing_eps,
                    variant,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(penalty)));
            }
            "block_orthogonality" => {
                let raw_groups = descriptor
                    .get("groups")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| format!("{context}.groups is required"))?;
                let mut groups = Vec::with_capacity(raw_groups.len());
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_axes = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.groups[{group_idx}] must be a list of latent axes")
                    })?;
                    let mut group = Vec::with_capacity(raw_axes.len());
                    for (axis_idx, raw_axis) in raw_axes.iter().enumerate() {
                        let axis = raw_axis.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.groups[{group_idx}][{axis_idx}] must be a non-negative integer"
                            )
                        })? as usize;
                        group.push(axis);
                    }
                    groups.push(group);
                }
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty =
                    BlockOrthogonalityPenalty::new(slice, groups, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::BlockOrthogonality(Arc::new(penalty)));
            }
            "ibp_assignment" | "ibp_assignment_penalty" => {
                let k_max = analytic_descriptor_usize(descriptor, "k_max", target.d)?;
                let alpha = analytic_descriptor_f64(descriptor, "alpha", 1.0)?;
                let tau = analytic_descriptor_f64(descriptor, "tau", 1.0)?;
                let temperature_schedule =
                    analytic_descriptor_temperature_schedule(descriptor, &context)?;
                let learnable_alpha = descriptor
                    .get("learnable_alpha")
                    .or_else(|| descriptor.get("learnable"))
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = IBPAssignmentPenalty::new(k_max, alpha, tau, learnable_alpha);
                let penalty = match temperature_schedule {
                    Some(schedule) => penalty.with_temperature_schedule(schedule),
                    None => penalty,
                };
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::IBPAssignment(Arc::new(penalty)));
            }
            "softmax_assignment_sparsity" => {
                let k_atoms = analytic_descriptor_usize(descriptor, "k_atoms", target.d)?;
                let temperature = analytic_descriptor_f64(descriptor, "temperature", 1.0)?;
                let penalty = SoftmaxAssignmentSparsityPenalty::new(k_atoms, temperature);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(penalty)));
            }
            "total_variation" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let difference_op = analytic_descriptor_difference_op(descriptor, &context)?;
                let smoothing_eps = analytic_descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = TotalVariationPenalty::new(
                    weight,
                    n_eff,
                    difference_op,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::TotalVariation(Arc::new(penalty)));
            }
            "nuclear_norm" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = analytic_descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let max_rank = match descriptor.get("max_rank") {
                    None | Some(JsonValue::Null) => None,
                    Some(value) => {
                        let raw = value.as_u64().ok_or_else(|| {
                            format!("{context}.max_rank must be null or a positive integer")
                        })?;
                        if raw == 0 {
                            return Err(format!("{context}.max_rank must be > 0"));
                        }
                        Some(raw as usize)
                    }
                };
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty =
                    NuclearNormPenalty::new(slice, weight, n_eff, smoothing_eps, max_rank, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));
            }
            "block_sparsity" => {
                let raw_groups = descriptor
                    .get("groups")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| format!("{context}.groups is required"))?;
                let mut groups = Vec::with_capacity(raw_groups.len());
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_axes = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.groups[{group_idx}] must be a list of latent axes")
                    })?;
                    let mut group = Vec::with_capacity(raw_axes.len());
                    for (axis_idx, raw_axis) in raw_axes.iter().enumerate() {
                        let axis = raw_axis.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.groups[{group_idx}][{axis_idx}] must be a non-negative integer"
                            )
                        })? as usize;
                        group.push(axis);
                    }
                    groups.push(group);
                }
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = analytic_descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = BlockSparsityPenalty::new(
                    slice,
                    groups,
                    weight,
                    n_eff,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::BlockSparsity(Arc::new(penalty)));
            }
            "row_precision_prior" | "aux_conditional_prior" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let lambda_per_row = analytic_descriptor_array3_flat(
                    descriptor,
                    "lambda_per_row",
                    "lambda_per_row_shape",
                    &context,
                )?;
                let penalty = RowPrecisionPriorPenalty::new(
                    slice,
                    lambda_per_row,
                    weight,
                    n_eff,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::RowPrecisionPrior(Arc::new(penalty)));
            }
            "ivae_ridge_mean_gauge" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let ridge_eps = analytic_descriptor_f64(descriptor, "ridge_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let aux =
                    analytic_descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let penalty =
                    IvaeRidgeMeanGauge::new(slice, aux, ridge_eps, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::IvaeRidgeMeanGauge(Arc::new(penalty)));
            }
            "parametric_row_precision_prior" | "parametric_aux_conditional_prior" => {
                let weight = analytic_descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = analytic_descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let aux =
                    analytic_descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let log_alpha =
                    analytic_descriptor_array1_flat(descriptor, "log_alpha", &context)?;
                let raw_beta =
                    analytic_descriptor_array1_flat(descriptor, "raw_beta", &context)?;
                let mu =
                    analytic_descriptor_array2_flat(descriptor, "mu", "mu_shape", &context)?;
                let penalty = ParametricRowPrecisionPriorPenalty::new(
                    slice, aux, log_alpha, raw_beta, mu, weight, n_eff, learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::ParametricRowPrecisionPrior(Arc::new(
                    penalty,
                )));
            }
            other => return Err(format!("{context}.kind has unsupported analytic penalty {other:?}")),
        }
    }
    Ok(registry)
}

fn analytic_descriptor_array3_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array3<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a three-item shape list"))?;
    if shape_values.len() != 3 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly three dimensions"
        ));
    }
    let mut shape = [0usize; 3];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim.as_u64().ok_or_else(|| {
            format!("{context}.{shape_key}[{idx}] must be a positive integer")
        })?;
        if dim == 0 {
            return Err(format!("{context}.{shape_key}[{idx}] must be > 0"));
        }
        shape[idx] = dim as usize;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .and_then(|v| v.checked_mul(shape[2]))
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

fn analytic_descriptor_array1_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    context: &str,
) -> Result<Array1<f64>, String> {
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.is_empty() {
        return Err(format!("{context}.{data_key} must be non-empty"));
    }
    let mut flat = Vec::with_capacity(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Ok(Array1::from(flat))
}

fn analytic_descriptor_array2_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array2<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a two-item shape list"))?;
    if shape_values.len() != 2 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly two dimensions"
        ));
    }
    let mut shape = [0usize; 2];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim.as_u64().ok_or_else(|| {
            format!("{context}.{shape_key}[{idx}] must be a positive integer")
        })?;
        if dim == 0 {
            return Err(format!("{context}.{shape_key}[{idx}] must be > 0"));
        }
        shape[idx] = dim as usize;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array2::from_shape_vec((shape[0], shape[1]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

fn json_array2(value: &JsonValue, context: &str) -> Result<Array2<f64>, String> {
    let rows = value.as_array().ok_or_else(|| {
        format!("{context} must be a two-dimensional numeric array")
    })?;
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
            let value = cell.as_f64().ok_or_else(|| {
                format!("{context}[{i}][{j}] must be a finite number")
            })?;
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
                if d == 1 {
                    Ok(LatentManifold::Circle)
                } else {
                    Ok(LatentManifold::Product(
                        (0..d).map(|_| LatentManifold::Circle).collect(),
                    ))
                }
            }
            "sphere" | "sn" => Ok(LatentManifold::Sphere { dim: d }),
            "torus" => Ok(LatentManifold::Product(
                (0..d).map(|_| LatentManifold::Circle).collect(),
            )),
            "cylinder" => {
                if d < 2 {
                    return Err(format!("{context}='cylinder' requires d >= 2"));
                }
                let mut parts = Vec::with_capacity(d);
                parts.push(LatentManifold::Circle);
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
            parts.push(parse_latent_manifold(
                Some(item),
                1,
                &format!("{context}[{idx}]"),
            )?.manifold);
        }
        LatentManifold::Product(parts)
    } else {
        return Err(format!("{context} must be a string, object, or product array"));
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
            Some(value) if value.as_str().is_some_and(|s| s.eq_ignore_ascii_case("pca")) => {
                LatentInitSpec::Pca
            }
            Some(value) if value.as_str().is_some_and(|s| s.eq_ignore_ascii_case("random")) => {
                LatentInitSpec::Random
            }
            Some(value) => LatentInitSpec::Explicit(json_array2(
                value,
                &format!("latents['{key}'].init"),
            )?),
        };
        let aux_prior = match obj.get("aux_prior").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => {
                let aux = value.as_object().ok_or_else(|| {
                    format!("latents['{key}'].aux_prior must be an object")
                })?;
                let u = json_array2(
                    aux.get("u").ok_or_else(|| {
                        format!("latents['{key}'].aux_prior.u is required")
                    })?,
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
        if dim_selection.is_some() && aux_prior.is_none() {
            return Err(format!(
                "latents['{key}'] uses dim_selection without aux_prior; ARD alone is not an identifiable latent-coordinate gauge"
            ));
        }
        let explicit_none_mode = obj
            .get("id_mode")
            .or_else(|| obj.get("mode"))
            .and_then(JsonValue::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("none"));
        if aux_prior.is_none() && dim_selection.is_none() && !explicit_none_mode {
            return Err(format!(
                "latents['{key}'] requires aux_prior for identifiable joint REML; pass id_mode='none' only when a separate gauge fix is supplied"
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
                let mut seed =
                    0xD1B54A32D192ED03_u64 ^ ((spec.n as u64) << 16) ^ spec.d as u64;
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
    match (&spec.aux_prior, &spec.dim_selection) {
        (Some(aux), Some(dim)) => {
            if let Some(init) = dim.init_log_precision.as_ref() {
                if init.len() != spec.d {
                    return Err(format!(
                        "latent '{}' dim_selection.init_log_precision has length {}, expected {}",
                        spec.target,
                        init.len(),
                        spec.d
                    ));
                }
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
    let analytic_penalties = build_standard_latent_analytic_penalty_registry(
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
    if let Some(aux) = spec.aux_prior.as_ref() {
        if aux.u.nrows() != spec.n {
            return Err(format!(
                "latent '{}' aux_prior.u has {} rows, expected {}",
                spec.target,
                aux.u.nrows(),
                spec.n
            ));
        }
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
        if let ParsedTerm::Smooth { vars, .. } = term {
            if vars.len() == 1 && vars[0] == spec.target {
                *vars = synthetic_vars.clone();
                matched = true;
            }
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
            term_index: usize::MAX,
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
        | crate::smooth::SmoothBasisSpec::Matern { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Duchon { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::Pca { feature_cols, .. }
        | crate::smooth::SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            Some(feature_cols.clone())
        }
        crate::smooth::SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_basis_feature_cols_for_latent(smooth)
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
            if matches!(
                &spec.knotspec,
                crate::basis::BSplineKnotSpec::PeriodicUniform { .. }
            ) {
                LatentManifold::Circle
            } else {
                LatentManifold::Euclidean
            }
        }
        crate::smooth::SmoothBasisSpec::Sphere { .. } => LatentManifold::Sphere { dim: d },
        crate::smooth::SmoothBasisSpec::Duchon { spec, .. } if spec.periodic && d == 1 => {
            LatentManifold::Circle
        }
        crate::smooth::SmoothBasisSpec::TensorBSpline { spec, .. } => {
            let parts: Vec<LatentManifold> = spec
                .marginalspecs
                .iter()
                .map(|margin| {
                    if matches!(
                        &margin.knotspec,
                        crate::basis::BSplineKnotSpec::PeriodicUniform { .. }
                    ) {
                        LatentManifold::Circle
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
        crate::smooth::SmoothBasisSpec::ThinPlate { .. }
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
) -> Result<MaterializedModel<'a>, String> {
    if config.noise_offset_column.is_some() {
        return Err(
            "noise_offset_column requires a location-scale model with noise_formula".to_string(),
        );
    }
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
    )?;

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

    let policy = resolved_resource_policy(config, term_data, crate::resource::ProblemHints::default());
    let spec = build_termspec_with_geometry(
        &term_parsed.terms,
        term_data,
        &term_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
    )?;
    if let Some(coord) = latent_coord.as_mut() {
        coord.term_index = spec
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
        if coord.manifold_auto {
            let inferred =
                natural_latent_manifold_for_basis(&spec.smooth_terms[coord.term_index].basis, coord.feature_cols.len());
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
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: config.firth,
        adaptive_regularization: standard_adaptive_regularization_options(config, &spec),
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
        let link_kind = link_choice
            .as_ref()
            .map(|c| InverseLink::Standard(c.link))
            .unwrap_or_else(|| {
                if let Some(state) = latent_cloglog {
                    InverseLink::LatentCLogLog(state)
                } else {
                    InverseLink::Standard(LinkFunction::Logit)
                }
            });
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders.clone(),
                double_penalty: cfg.double_penalty,
            },
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
            wiggle_options: None,
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
) -> Result<MaterializedModel<'a>, String> {
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
    let z_column = config
        .z_column
        .as_deref()
        .ok_or_else(|| "Bernoulli marginal-slope requires z_column".to_string())?;

    let (_, parsed_logslope) =
        parse_matching_auxiliary_formula(logslope_formula, &parsed.response, "logslope_formula")?;
    if parsed_logslope.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported inside logslope_formula".to_string(),
        }
        .into());
    }
    validate_marginal_slope_z_column_exclusion(
        parsed,
        &parsed_logslope,
        z_column,
        "Bernoulli marginal-slope",
        "logslope_formula",
    )?;

    let mut inference_notes = Vec::new();
    // Bernoulli marginal-slope: structurally operator-only at biobank scale, so
    // flip the hint regardless of n to keep dense fallbacks blocked.
    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            marginal_slope_biobank_active: true,
        },
    );
    let aliased_col_map = column_map_with_alias(col_map, "z", z_column);
    let marginalspec = build_termspec_with_geometry(
        &parsed.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
    )?;
    let logslopespec = build_termspec_with_geometry(
        &parsed_logslope.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
    )?;
    let z_idx = resolve_role_col(col_map, z_column, "z")?;
    let z = data.values.column(z_idx).to_owned();
    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let marginal_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let logslope_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let routing = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
    )?;
    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(LinkFunction::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
        score_warp: routing.score_warp,
        link_dev: routing.link_dev,
        latent_z_policy: Default::default(),
    };

    Ok(MaterializedModel {
        request: FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
            data: data.values.view(),
            spec,
            options: BlockwiseFitOptions {
                compute_covariance: true,
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
    entry_col: &str,
    exit_col: &str,
    event_col: &str,
) -> Result<MaterializedModel<'a>, String> {
    let mut inference_notes = Vec::new();

    // Extract columns
    let entry_idx = resolve_role_col(col_map, entry_col, "entry")?;
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
            normalize_survival_time_pair(data.values[[i, entry_idx]], data.values[[i, exit_idx]], i)
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
                .to_string(),
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
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
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
                .to_string(),
        );
    }

    let policy = resolved_resource_policy(
        config,
        data,
        crate::resource::ProblemHints {
            // Survival marginal-slope shares the operator-only invariant with
            // the Bernoulli path; flag it as such so strict mode is selected
            // even at small n.
            marginal_slope_biobank_active: survival_mode == SurvivalLikelihoodMode::MarginalSlope,
        },
    );
    let marginal_slope_aliased_col_map = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        Some(column_map_with_alias(
            col_map,
            "z",
            config.z_column.as_deref().ok_or_else(|| {
                "marginal-slope survival requires z_column in FitConfig".to_string()
            })?,
        ))
    } else {
        None
    };
    let termspec_col_map = marginal_slope_aliased_col_map.as_ref().unwrap_or(col_map);
    let termspec = build_termspec_with_geometry(
        &parsed.terms,
        data,
        termspec_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
    )?;

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
        let noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        // Use the same aliased col_map as the main termspec — survival
        // marginal-slope reserves `z` as a placeholder for `--z-column`,
        // and the logslope/noise formula may reference it too.
        build_termspec_with_geometry(
            &noise_parsed.terms,
            data,
            termspec_col_map,
            &mut inference_notes,
            config.scale_dimensions,
            &policy,
        )?
    } else if survival_mode == SurvivalLikelihoodMode::LocationScale {
        termspec.clone()
    } else {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };
    let marginal_z_column_name =
        if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
            Some(config.z_column.as_deref().ok_or_else(|| {
                "marginal-slope survival requires z_column in FitConfig".to_string()
            })?)
        } else {
            None
        };
    let (
        marginal_z,
        marginal_logslopespec,
        marginal_logslopespecs,
        marginal_slope_deviation_routing,
    ) = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        drop(resolve_survival_marginal_slope_base_link(parsed.linkspec.as_ref())?);
        let default_z_column =
            marginal_z_column_name.expect("marginal-slope z column should be available");
        if let Some(ls_formula) = config.logslope_formula.as_deref() {
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
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
                specs.push(build_termspec_with_geometry(
                    &surface.terms,
                    data,
                    &aliased_col_map,
                    &mut inference_notes,
                    config.scale_dimensions,
                    &policy,
                )?);
            }
            (
                Some(z),
                specs.first().cloned(),
                Some(specs),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                )?,
            )
        } else {
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
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;
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
            let prepared = prepare_workflow_survival_time_stack(
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
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
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
            let (prepared, time_block) = build_time_block(candidate)?;
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
                    base_link: resolve_survival_marginal_slope_base_link(parsed.linkspec.as_ref())?,
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
                },
                options: BlockwiseFitOptions {
                    compute_covariance: false,
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
            let prepared = prepare_workflow_survival_time_stack(
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
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
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
            let prepared = prepare_workflow_survival_time_stack(
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
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
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
                    return Err(WorkflowError::IntegrationFailed {
                        reason: format!(
                            "workflow survival location-scale baseline: non-finite profile cost \
                             (log_likelihood={}, stable_penalty_term={}, cost={})",
                            fit_result.fit.fit.log_likelihood,
                            fit_result.fit.fit.stable_penalty_term,
                            profile_cost
                        ),
                    }
                    .into());
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
            Err(e) => return Err(e),
        }
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        optimize_survival_baseline_config(
            &baseline_cfg,
            "workflow survival baseline",
            |candidate| match survival_mode {
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
) -> Result<MaterializedModel<'a>, String> {
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
    let mut covariate_spec =
        build_termspec(&parsed.terms, data, col_map, &mut inference_notes, &policy)?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut covariate_spec);
    }

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

fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, String> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    let noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
    )?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let policy = resolved_resource_policy(config, data, crate::resource::ProblemHints::default());
    let mut meanspec = build_termspec(&parsed.terms, data, col_map, &mut inference_notes, &policy)?;
    let mut log_sigmaspec = build_termspec(
        &noise_parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        &policy,
    )?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut meanspec);
        enable_scale_dimensions(&mut log_sigmaspec);
    }

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
        let link_kind = link_choice
            .as_ref()
            .map(|c| InverseLink::Standard(c.link))
            .unwrap_or(InverseLink::Standard(LinkFunction::Logit));
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

    fn workflow_test_outer_result(converged: bool, rho: Array1<f64>) -> OuterResult {
        OuterResult {
            rho,
            final_value: 1.25,
            iterations: 7,
            final_grad_norm: Some(0.5),
            final_gradient: None,
            final_hessian: None,
            converged,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
            operator_trust_radius: None,
            operator_stop_reason: None,
        }
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
        assert!(matches!(
            spec.nullspace_order,
            DuchonNullspaceOrder::Degree(2)
        ));
        assert_eq!(spec.power, 0.0);
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
        assert_eq!(spec.nullspace_order, DuchonNullspaceOrder::Zero);
        assert_eq!(spec.power, 2.0);
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
            |_| Some(InverseLink::Standard(LinkFunction::Logit)),
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
}
