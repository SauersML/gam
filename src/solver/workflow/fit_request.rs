
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


/// Map a converged outer-optimizer result to its recovered inverse-link state,
/// or surface a convergence/recovery failure as a `WorkflowError`. Used by the
/// survival location-scale inverse-link profiling path to turn the optimized
/// `rho` into the concrete `InverseLink` before the final fixed-link refit.
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
    /// Exact O(n) state-space cubic/linear/quintic smoothing-spline scan
    /// (#1030/#1034). A scan-bearing model IS a Gaussian-identity model with a
    /// different (exact) representation: rather than a dense design + coefficient
    /// vector it carries the Durbin–Koopman smoother posterior directly (knots,
    /// smoothed states, pointwise variances, σ², log λ, exact diffuse-REML EDF,
    /// and an exact per-row `predict`). Library callers that want the fitted
    /// posterior get it here without paying the dense O(n·k²)+O(k³) route; the
    /// CLI/FFI save paths build the persistence payload from the same
    /// `SplineScanFit` via `assemble_spline_scan_payload`.
    SplineScan(crate::solver::spline_scan::SplineScanFit),
    /// O(n log n) multiresolution residual-cascade smooth (#1032). UNLIKE the
    /// 1-D scan, the cascade is NOT the same posterior as the Duchon/Matérn term
    /// it stands in for (a different finite basis — the multilevel Wendland
    /// frame), so it is never a silent swap: this variant is produced only when
    /// the structural detector [`residual_cascade_fast_path`] fires on an
    /// eligible scattered-low-d Gaussian fit past the dense-kernel cliff AND the
    /// in-cascade quasi-uniformity guard certifies the metric; every other shape
    /// (and a rejected metric) falls through to the dense `fit_model` path. The
    /// cascade-bearing model carries the
    /// [`ResidualCascadeFit`](crate::solver::residual_cascade::ResidualCascadeFit)
    /// directly — knots-free nested geometry, coefficients, the factored
    /// precision, and an exact per-row `predict`; the CLI/FFI save paths build
    /// the persistence payload from its `to_state` snapshot.
    ResidualCascade(crate::solver::residual_cascade::ResidualCascadeFit),
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
    // The joint (mean + log-precision) posterior covariance / EDF is requested
    // unconditionally inside `fit_dispersion_glm_location_scale_terms`, which is
    // the shared entry for all four genuine-dispersion mean families (gam#1119),
    // so no per-request override is needed here.
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
    use crate::solver::outer_strategy::{Derivative, HessianResult, OuterEval, OuterProblem};
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
        // Analytic-gradient BFGS over the baseline shape params (weibull
        // scale/shape; gompertz rate/shape; gompertz-makeham rate/shape/makeham).
        //
        // The cost optimized here is the *profile penalized NLL*
        //   V(θ) = 0.5·deviance(β̂(θ); o(θ)) + 0.5·β̂ᵀSβ̂   (= survival_working_reml_score),
        // where the baseline θ enters the transformation working model only
        // through the three additive time-block offsets
        //   o_E(θ) = η_target(age_entry), o_X(θ) = η_target(age_exit),
        //   o_D(θ) = d/dt η_target |_{age_exit}.
        // β̂(θ) is the (constrained) PIRLS optimum, so ∂V/∂β = 0 there and by the
        // envelope theorem dV/dθ_k = ∂V/∂θ_k|_{β=β̂} — the explicit partial holding
        // β̂ fixed. The active-set inequality constraints {β_j ≥ 0} carry no
        // θ-dependence, so the constrained envelope identity is unchanged. That
        // explicit partial is exactly the residual×offset-partial contraction
        //   dV/dθ_k = Σ_i r^X_i ∂o_X_i/∂θ_k + r^E_i ∂o_E_i/∂θ_k + r^D_i ∂o_D_i/∂θ_k,
        // with r^* = WorkingModelSurvival::offset_channel_residuals(β̂) and the
        // η-channel offset partials supplied by baseline_offset_theta_partials
        // (contracted by baseline_chain_rule_gradient). See the derivation header
        // on baseline_chain_rule_gradient. BFGS over this exact gradient converges
        // in ≲10 outer evaluations on the 2–3 dim surface.
        baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
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
                let cost = survival_working_reml_score(&state);
                let residuals = model.offset_channel_residuals(&beta).map_err(|err| {
                    format!("failed to form survival baseline offset residuals: {err}")
                })?;
                let gradient = baseline_chain_rule_gradient(
                    spec.age_entry.view(),
                    spec.age_exit.view(),
                    // RP transformation has no interval upper-bound channel;
                    // `residuals.right` is all-zero so `age_exit` is an unconsulted
                    // placeholder for `age_right`.
                    spec.age_exit.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "workflow survival transformation baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                Ok((cost, gradient))
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

    /// Profile the survival location-scale fit at a fixed inverse-link state:
    /// substitutes `inverse_link` into the spec and runs the full penalized fit.
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
        // Analytic-gradient BFGS over the inverse-link parameters θ_link
        // (SAS ε/log_δ, BetaLogistic ε/log_δ, Mixture ρ). The link enters the
        // location-scale likelihood through the standardized residual it maps,
        // and the EXACT data-fit θ-gradient
        //   ∂(−ℓ)/∂θ_link = −Σ_i w_i·( event_mix(d, ∂logφ(u1), ∂logS(u1)) − ∂logS(u0) )
        // is formed analytically from the inverse-link param partials
        // (`SurvivalLocationScaleFamily::link_param_data_fit_gradient`, carried
        // on the fit result as `link_param_data_fit_gradient`). We optimize the
        // *profile penalized NLL* `−ℓ + ½βᵀSβ` — not the LAML `reml_score` whose
        // ½log|H+S_λ| term has its own θ_link dependence through H(β̂,θ) — so the
        // envelope-theorem gradient matches the cost surface; the final fit
        // downstream still picks ρ on the full REML surface at the converged link.
        fn optimize_link_parameters<R>(
            data: ArrayView2<'_, f64>,
            spec: &SurvivalLocationScaleTermSpec,
            kappa_options: &SpatialLengthScaleOptimizationOptions,
            init: Array1<f64>,
            name: &str,
            final_wiggle: Option<LinkWiggleConfig>,
            wiggle_cfg: Option<LinkWiggleConfig>,
            make_link: impl Fn(&Array1<f64>) -> Result<InverseLink, String> + Clone,
            recover: R,
        ) -> Result<SurvivalLocationScaleProfile, String>
        where
            R: Fn(&Array1<f64>) -> Option<InverseLink>,
        {
            use crate::solver::outer_strategy::{
                DeclaredHessianForm, Derivative, HessianResult, OuterEval, OuterProblem,
            };
            let dim = init.len();
            // Box bounds keep line-search probes inside a physically admissible
            // region (|ε|, |log δ| ≤ 6 gives the SAS link a finite range on both
            // tails; mixture logits stay in a numerically sane band). With an
            // analytic gradient and no declared Hessian the planner routes this
            // to BFGS.
            let lower = init.mapv(|v| v - 6.0);
            let upper = init.mapv(|v| v + 6.0);
            let problem = OuterProblem::new(dim)
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Unavailable)
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
            // The objective returns the profile-NLL cost and the exact analytic
            // θ_link-gradient from the converged fit at this candidate link.
            let eval_link = move |theta: &Array1<f64>| -> Result<(f64, Array1<f64>), String> {
                let link = make_link(theta)?;
                let profile = profile_survival_location_scale_with_inverse_link(
                    data,
                    spec,
                    link,
                    wiggle_cfg.clone(),
                    kappa_options,
                )?;
                let cost = -profile.fit.fit.log_likelihood
                    + 0.5 * profile.fit.fit.stable_penalty_term;
                if !cost.is_finite() {
                    return Err(format!(
                        "survival inverse-link ({name}): non-finite profile cost \
                         (log_likelihood={}, stable_penalty_term={})",
                        profile.fit.fit.log_likelihood, profile.fit.fit.stable_penalty_term
                    ));
                }
                let gradient = profile
                    .fit
                    .link_param_data_fit_gradient
                    .clone()
                    .ok_or_else(|| {
                        format!(
                            "survival inverse-link ({name}): fit reported no link-parameter \
                             data-fit gradient"
                        )
                    })?;
                if gradient.len() != theta.len() {
                    return Err(format!(
                        "survival inverse-link ({name}): gradient dim {} != theta dim {}",
                        gradient.len(),
                        theta.len()
                    ));
                }
                Ok((cost, gradient))
            };
            let cost_eval = eval_link.clone();
            let cost_fn = move |_: &mut (), theta: &Array1<f64>| {
                cost_eval(theta)
                    .map(|(cost, _)| cost)
                    .map_err(crate::estimate::EstimationError::InvalidInput)
            };
            let eval_fn = move |_: &mut (), theta: &Array1<f64>| {
                let (cost, gradient) =
                    eval_link(theta).map_err(crate::estimate::EstimationError::InvalidInput)?;
                Ok(OuterEval {
                    cost,
                    gradient,
                    hessian: HessianResult::Unavailable,
                    inner_beta_hint: None,
                })
            };
            let mut obj = problem.build_objective(
                (),
                cost_fn,
                eval_fn,
                None::<fn(&mut ())>,
                None::<
                    fn(
                        &mut (),
                        &Array1<f64>,
                    )
                        -> Result<
                            crate::solver::outer_strategy::EfsEval,
                            crate::estimate::EstimationError,
                        >,
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
            InverseLink::Sas(state0) => optimize_link_parameters(
                data,
                spec,
                kappa_options,
                Array1::from_vec(vec![state0.epsilon, state0.log_delta]),
                "SAS",
                wiggle.clone(),
                wiggle.clone(),
                |theta| {
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: theta[0],
                        initial_log_delta: theta[1],
                    })
                    .map(InverseLink::Sas)
                },
                |rho| {
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: rho[0],
                        initial_log_delta: rho[1],
                    })
                    .ok()
                    .map(InverseLink::Sas)
                },
            ),
            InverseLink::BetaLogistic(state0) => optimize_link_parameters(
                data,
                spec,
                kappa_options,
                Array1::from_vec(vec![state0.epsilon, state0.log_delta]),
                "BetaLogistic",
                wiggle.clone(),
                wiggle.clone(),
                |theta| {
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: theta[0],
                        initial_log_delta: theta[1],
                    })
                    .map(InverseLink::BetaLogistic)
                },
                |rho| {
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: rho[0],
                        initial_log_delta: rho[1],
                    })
                    .ok()
                    .map(InverseLink::BetaLogistic)
                },
            ),
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                let components_recover = components.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    state0.rho.clone(),
                    "mixture",
                    wiggle.clone(),
                    wiggle.clone(),
                    move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components.clone(),
                            initial_rho: rho.clone(),
                        })
                        .map(InverseLink::Mixture)
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
    normalize_survival_time_pair, optimize_survival_baseline_config_with_gradient,
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
    parse_matching_auxiliary_formula, parse_surv_interval_response, parse_surv_response,
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
    /// Expectile asymmetry `τ ∈ (0, 1)` for `family = "expectile"`.
    ///
    /// When `family` resolves to `"expectile"` the fit minimizes the
    /// Newey–Powell asymmetric squared loss `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with
    /// `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`, tracing the conditional
    /// `τ`-expectile — the smooth analogue of the `τ`-quantile. `τ = 0.5`
    /// reduces exactly to the Gaussian-identity mean fit. The whole penalized
    /// smooth + REML `λ`-selection machinery is reused via a Least
    /// Asymmetrically Weighted Squares (LAWS) outer loop. `None` defaults to
    /// the median expectile `τ = 0.5` when the family is `"expectile"`; it is
    /// ignored for every other family. The asymmetry may also be written inline
    /// as `family = "expectile(0.9)"`, which fills this field at resolve time.
    pub expectile_tau: Option<f64>,
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
    /// Engage the cross-process ON-DISK persistent warm-start layer (#1082).
    ///
    /// Default `false`: only the always-on in-memory warm start runs, so a
    /// single fit and throwaway/replicate/CI-coverage loops pay zero disk I/O
    /// (no `WarmStartStore` dir/eviction scan, no record load/store). Set
    /// `true` to engage cross-process / repeat-fit resume: the flag threads
    /// `FitConfig → FitOptions → ExternalOptimOptions` down to the standard
    /// `RemlState`, which then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
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
            expectile_tau: None,
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
            persist_warm_start_disk: false,
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
/// Resolve the expectile asymmetry `τ` requested by `config`, if any.
///
/// Returns `Ok(Some(τ))` when `config.family` is `"expectile"` (optionally with
/// an inline asymmetry, `"expectile(0.9)"`), `Ok(None)` for every other family,
/// and `Err` when an expectile request carries an out-of-range `τ`. The inline
/// form takes precedence over the explicit [`FitConfig::expectile_tau`] field
/// only when both are present and disagree is rejected as a contradiction; when
/// neither pins `τ`, the median expectile `τ = 0.5` (the ordinary mean fit) is
/// the default.
fn expectile_tau_for_config(config: &FitConfig) -> Result<Option<f64>, WorkflowError> {
    let Some(raw) = config.family.as_deref() else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !(lower == "expectile" || lower.starts_with("expectile(")) {
        return Ok(None);
    }
    let invalid = |reason: String| WorkflowError::InvalidConfig { reason };
    // Optional inline asymmetry: `expectile(0.9)`.
    let inline_tau = if let Some(rest) = lower.strip_prefix("expectile(") {
        let inner = rest.strip_suffix(')').ok_or_else(|| {
            invalid(format!(
                "expectile family asymmetry must be written as `expectile(τ)`; got `{trimmed}`"
            ))
        })?;
        let value: f64 = inner.trim().parse().map_err(|_| {
            invalid(format!(
                "expectile asymmetry `{}` is not a finite number",
                inner.trim()
            ))
        })?;
        Some(value)
    } else {
        None
    };
    let tau = match (inline_tau, config.expectile_tau) {
        (Some(a), Some(b)) if (a - b).abs() > 0.0 => {
            return Err(invalid(format!(
                "expectile asymmetry given both inline (`expectile({a})`) and via expectile_tau \
                 ({b}); supply exactly one"
            )));
        }
        (Some(a), _) => a,
        (None, Some(b)) => b,
        (None, None) => 0.5,
    };
    if !(tau.is_finite() && tau > 0.0 && tau < 1.0) {
        return Err(invalid(format!(
            "expectile asymmetry τ must be finite and strictly in (0, 1); got {tau}"
        )));
    }
    Ok(Some(tau))
}


/// Per-row asymmetric LAWS weight `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`, scaled
/// by the base prior weight. At the boundary `yᵢ = μᵢ` the two half-weights
/// agree in the limit only at `τ = 0.5`; the convention `yᵢ > μᵢ ⇒ τ` (strict)
/// matches Newey–Powell's lower-closed asymmetric loss and is what `expectreg`
/// uses. The fixed point is independent of the tie convention because ties form
/// a measure-zero set under any continuous response.
fn expectile_row_weights(
    y: ArrayView1<f64>,
    mu: ArrayView1<f64>,
    base: ArrayView1<f64>,
    tau: f64,
) -> Array1<f64> {
    Array1::from_shape_fn(y.len(), |i| {
        let asym = if y[i] > mu[i] { tau } else { 1.0 - tau };
        base[i] * asym
    })
}


pub fn fit_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FitResult, WorkflowError> {
    // Expectile regression (Newey–Powell asymmetric least squares): when the
    // family resolves to "expectile", the τ-expectile of `y | x` is the
    // minimizer of `Σ wᵢ(τ)·(yᵢ − μᵢ)²`, `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`
    // — the smooth analogue of the τ-quantile. The minimizer is a Least
    // Asymmetrically Weighted Squares (LAWS) fixed point: iterate the penalized
    // Gaussian-identity GAM with `wᵢ(τ)` recomputed from the current `μᵢ` until
    // the residual-sign pattern stabilizes. REML λ-selection runs inside each
    // inner Gaussian solve, so every gam smooth/tensor/spatial basis becomes a
    // penalized expectile smooth with data-driven smoothing for free. This is a
    // genuine estimator route, not a silent swap: it fires only on the explicit
    // `family = "expectile"`. Every other family falls through unchanged.
    if let Some(tau) = expectile_tau_for_config(config)? {
        return fit_expectile_laws(formula, data, config, tau);
    }
    let mat = materialize(formula, data, config)?;
    // Exact O(n) spline-scan fast path (#1030): when the materialized request
    // is the single 1-D Gaussian-identity penalized-smooth shape the
    // state-space scan solves exactly, route through it and return the
    // scan-bearing model directly — the same penalized posterior at O(n) per
    // λ-trial instead of the dense design/Gram route. Detection is structural
    // and conservative (see `spline_scan_fast_path`); every other shape falls
    // through to the dense `fit_model` path unchanged. Mirrors the CLI
    // (main.rs run_fit) and FFI consumers, which build the persistence payload
    // from this same `SplineScanFit`.
    if let FitRequest::Standard(request) = &mat.request {
        if let Some(inputs) = spline_scan_fast_path(request) {
            let scan = crate::solver::spline_scan::fit_spline_scan(
                &inputs.x,
                &inputs.y,
                &inputs.w,
                inputs.order,
            )
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?;
            return Ok(FitResult::SplineScan(scan));
        }
        // O(n log n) multiresolution residual-cascade fast path (#1032): a
        // scattered low-d Gaussian-identity Duchon/Matérn smooth past the
        // dense-kernel cliff. UNLIKE the scan, the cascade is a DIFFERENT
        // posterior from the dense radial term, so it only ever fires as an
        // explicit alternative estimator on the exact structural signature
        // (`residual_cascade_fast_path`) AND when the in-cascade quasi-uniformity
        // guard certifies the metric — a rejected metric or any ineligible shape
        // falls through to the dense `fit_model` path (a genuine estimator
        // choice, never a silent swap). The save paths build the persistence
        // payload from this `ResidualCascadeFit`'s `to_state` snapshot.
        if let Some(inputs) = residual_cascade_fast_path(request) {
            let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
            if let Ok(fit) = crate::solver::residual_cascade::fit_residual_cascade(
                &coord_refs,
                &inputs.y,
                &inputs.w,
                &inputs.metric,
                inputs.sobolev_s,
            ) {
                return Ok(FitResult::ResidualCascade(fit));
            }
            // The quasi-uniformity guard (caveat 2) or any degenerate-design
            // signal surfaces as a build/solve error; fall through to the dense
            // kernel path rather than failing the fit outright.
        }
    }
    // `fit_model` already returns `WorkflowError` end-to-end; propagate it
    // directly instead of stringifying then re-wrapping.
    fit_model(mat.request)
}


/// Least Asymmetrically Weighted Squares (LAWS) driver for expectile GAMs.
///
/// The τ-expectile surface minimizes `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with the residual-
/// sign asymmetric weight `wᵢ(τ)`. Because that weight is piecewise-constant in
/// `sign(yᵢ − μᵢ)`, the objective is the supremum of a finite family of
/// weighted least-squares problems and its minimizer is the unique fixed point
/// of: *solve the penalized WLS with weights frozen at the current sign
/// pattern, then recompute the sign pattern from the new fit*. The asymmetric
/// loss is strictly convex (weights bounded in `[min(τ,1−τ), max(τ,1−τ)] > 0`),
/// so this monotone-descent iteration converges, and since the sign pattern
/// takes finitely many values it stabilizes in finitely many steps (Schnabel &
/// Eilers 2009; the same Newton/IRLS-for-expectiles `expectreg` runs).
///
/// Each inner solve is the FULL standard Gaussian-identity GAM: any basis,
/// tensor, spatial smooth, by-variable, random effect, plus REML λ-selection on
/// the current asymmetric weights. The returned fit is an ordinary
/// [`FitResult::Standard`] whose coefficients ARE the penalized τ-expectile —
/// every downstream consumer (predict, posterior bands, persistence) works
/// unchanged. The reported scale is the asymmetric working variance, so
/// expectile standard errors are the sandwich-free Gaussian-form bands of the
/// converged weighted problem (a deliberate first-rung choice; see #1100).
fn fit_expectile_laws(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
    tau: f64,
) -> Result<FitResult, WorkflowError> {
    use crate::linalg::matrix::LinearOperator;

    // Inner fits are ordinary Gaussian-identity GAMs; the τ asymmetry lives
    // entirely in the per-iteration prior weights this driver injects.
    let gaussian_config = FitConfig {
        family: Some("gaussian".to_string()),
        link: Some("identity".to_string()),
        expectile_tau: None,
        ..config.clone()
    };

    // Materialize once to capture the fixed training design, response, offset,
    // and base prior weights. The design (basis, penalties, identifiability
    // transforms) does not depend on the prior weights, so it is reused across
    // every LAWS iteration; only the weight vector and the resulting β change.
    let base_mat = materialize(formula, data, &gaussian_config)?;
    let FitRequest::Standard(base_request) = base_mat.request else {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression is only defined for standard (non-survival, \
                     non-location-scale) responses"
                .to_string(),
        });
    };
    let StandardFitRequest {
        data: design_data,
        y,
        weights: base_weights,
        offset,
        spec,
        family: materialized_family,
        options,
        kappa_options,
        wiggle,
        coefficient_groups,
        penalty_block_gamma_priors,
        latent_coord,
        _marker,
    } = base_request;
    // The materializer already resolved the inner family to Gaussian-identity
    // from `gaussian_config`; assert it so a future materializer change that
    // silently picked a different family for `"gaussian"` is caught here rather
    // than producing a non-expectile fit.
    if !materialized_family.is_gaussian_identity() {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "expectile LAWS requires a Gaussian-identity inner family; materializer produced {}",
                materialized_family.name()
            ),
        });
    }

    if wiggle.is_some() || latent_coord.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "expectile regression does not support flexible-link wiggle or latent \
                     coordinates"
                .to_string(),
        });
    }

    let n = y.len();
    let gaussian_family = LikelihoodSpec::gaussian_identity();
    // Cold start: τ = 0.5 (symmetric) weights ⇒ the first inner fit is the OLS
    // mean GAM, the natural warm start for any τ.
    let mut weights = base_weights.clone();
    let mut last_sign: Option<Vec<bool>> = None;
    let mut last_result: Option<StandardFitResult> = None;

    // The sign pattern has 2ⁿ values but LAWS visits a monotone-descent subset;
    // empirically a handful of iterations suffice. The cap is a safety guard:
    // on the rare oscillation between two equal-objective sign patterns (only
    // possible when rows sit exactly on the fitted surface) the last fit is a
    // valid τ-expectile of the perturbation-stable problem, so returning it is
    // correct rather than an error.
    const MAX_LAWS_ITERS: usize = 50;

    for _iter in 0..MAX_LAWS_ITERS {
        let request = StandardFitRequest {
            data: design_data.clone(),
            y: y.clone(),
            weights: weights.clone(),
            offset: offset.clone(),
            spec: spec.clone(),
            family: gaussian_family.clone(),
            options: options.clone(),
            kappa_options: kappa_options.clone(),
            wiggle: None,
            coefficient_groups: coefficient_groups.clone(),
            penalty_block_gamma_priors: penalty_block_gamma_priors.clone(),
            latent_coord: None,
            _marker,
        };
        let result = fit_standard_model(request)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?;

        // Training-scale fitted mean μ = X·β (identity link, zero-checked
        // offset folded by the design path). The design columns match the
        // combined coefficient vector exactly (the same contract `predict`
        // and the safety tests rely on).
        let mu = result.design.design.apply(&result.fit.beta);
        if mu.len() != n {
            return Err(WorkflowError::IntegrationFailed {
                reason: format!(
                    "expectile LAWS: fitted mean length {} disagrees with response length {n}",
                    mu.len()
                ),
            });
        }
        let mut mu_off = mu;
        mu_off += &offset;

        let sign: Vec<bool> = (0..n).map(|i| y[i] > mu_off[i]).collect();
        let converged = last_sign.as_ref().is_some_and(|prev| prev == &sign);
        weights = expectile_row_weights(y.view(), mu_off.view(), base_weights.view(), tau);
        last_sign = Some(sign);
        last_result = Some(result);
        if converged {
            break;
        }
    }

    let result = last_result.ok_or_else(|| WorkflowError::IntegrationFailed {
        reason: "expectile LAWS produced no fit".to_string(),
    })?;
    Ok(FitResult::Standard(result))
}


/// Inputs extracted by [`spline_scan_fast_path`] for the exact O(n)
/// state-space cubic-smoothing-spline scan
/// ([`crate::solver::spline_scan::fit_spline_scan`]).
pub struct SplineScanInputs {
    /// Abscissae of the single 1-D smooth (training rows of its feature column).
    pub x: Vec<f64>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Smoothing-spline order `m = penalty_order ∈ {1, 2, 3}`: `m = 1` the
    /// random-walk/linear smoother (penalty `λ∫f′²`), `m = 2` the cubic
    /// smoother (penalty `λ∫f″²`), `m = 3` the quintic smoother (penalty
    /// `λ∫(f‴)²`).
    pub order: usize,
}


/// Detection seam for the exact O(n) cubic-smoothing-spline fast path.
///
/// This is the EARLIEST point in the standard workflow where a materialized
/// fit request carries everything needed to prove the model is exactly the
/// problem the scan solves: a Gaussian likelihood with identity link over
/// `intercept + one 1-D cubic-class penalized smooth` — i.e. the penalized
/// least-squares problem `min Σ w_i (y_i − f(x_i))² + λ∫f″²` with an
/// unpenalized `{1, x}` null space. The Kalman/RTS scan computes that
/// posterior (mean, pointwise variance, exact diffuse REML for λ) in O(n) per
/// λ-trial instead of the dense design/Gram O(n·k²) + O(k³) route.
///
/// Returns `Some` only when ALL of the following hold; everything else falls
/// through to the dense path:
/// - family is Gaussian + identity link;
/// - no link wiggle, no latent coordinates, no coefficient groups, no penalty
///   hyperpriors, no linear/box constraints, no Firth, no adaptive
///   regularization, no Kronecker systems, no externally injected null-space
///   dims;
/// - the term collection is exactly one smooth term — no linear terms, no
///   random effects, no by-variables / factor interactions;
/// - that smooth is a plain 1-D cubic (degree 3) B-spline with a single
///   order-2 penalty (`double_penalty=false` — the default `s(x)` adds a
///   second null-space shrinkage penalty, which the scan's diffuse `{1, x}`
///   null space deliberately does NOT shrink, so it is not the same
///   posterior and is excluded), open (non-cyclic) boundary, free endpoint
///   conditions, and no shape constraint;
/// - the offset is identically zero and every weight is finite and positive;
/// - at least 3 distinct finite abscissae (the scan's diffuse rank plus one).
///
/// λ-mapping note: the scan's penalty is exactly `λ∫f″²` (state-space
/// `q = 1/λ` at unit σ²). The dense 1-D B-spline path penalizes the same
/// cubic class through a reduced-rank discrete-difference Gram whose
/// normalization differs by a basis-dependent constant, so a λ selected by
/// one parameterization does not transfer numerically to the other. The scan
/// therefore always re-selects λ by its own exact diffuse REML criterion
/// (the optimizer of the same restricted likelihood, expressed in the scan's
/// parameterization); user-pinned smoothing parameters are not representable
/// at this seam (the formula DSL exposes none for this term class), so no
/// pinned-λ mapping arises.
///
/// Identifiability transforms on the smooth (centering / linear-trend
/// removal / orthogonality-to-intercept) are accepted as eligible: they only
/// re-coordinate the unpenalized null space against the implicit intercept
/// and do not change the fitted posterior of `E[y|x]`, which is what the
/// scan returns directly.
pub fn spline_scan_fast_path(request: &StandardFitRequest<'_>) -> Option<SplineScanInputs> {
    if !request.family.is_gaussian_identity() {
        return None;
    }
    if request.wiggle.is_some()
        || request.latent_coord.is_some()
        || !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        return None;
    }
    let options = &request.options;
    if options.latent_cloglog.is_some()
        || options.mixture_link.is_some()
        || options.sas_link.is_some()
        || options.linear_constraints.is_some()
        || options.adaptive_regularization.is_some()
        || options.kronecker_penalty_system.is_some()
        || options.kronecker_factored.is_some()
        || options.firth_bias_reduction
        || !options.nullspace_dims.is_empty()
    {
        return None;
    }
    let spec = &request.spec;
    if !spec.linear_terms.is_empty()
        || !spec.random_effect_terms.is_empty()
        || spec.smooth_terms.len() != 1
    {
        return None;
    }
    let term = &spec.smooth_terms[0];
    if !matches!(term.shape, crate::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    let crate::smooth::SmoothBasisSpec::BSpline1D {
        feature_col,
        spec: bspec,
    } = &term.basis
    else {
        return None;
    };
    // Smoothing-spline order m = penalty_order ∈ {1, 2, 3}. The exact scan
    // integrates the order-m integrated-Wiener prior whose natural spline has
    // degree 2m−1 (m=1 → linear, m=2 → cubic, m=3 → quintic), so require that
    // degree to match user intent. The de Jong exact diffuse leading-block
    // smoother (#1044) handles the m−1 partially-diffuse leading nodes for all
    // m ≤ MAX_ORDER; m > MAX_ORDER falls through to the dense path.
    let order = bspec.penalty_order;
    if !(1..=3).contains(&order)
        || bspec.degree != 2 * order - 1
        || bspec.double_penalty
        || !bspec.boundary_conditions.is_free()
        || !matches!(bspec.boundary, crate::basis::OneDimensionalBoundary::Open)
        || matches!(
            bspec.knotspec,
            crate::basis::BSplineKnotSpec::PeriodicUniform { .. }
        )
    {
        return None;
    }
    if request.offset.iter().any(|&v| v != 0.0) {
        return None;
    }
    if request.weights.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return None;
    }
    if *feature_col >= request.data.ncols() || request.y.len() != request.data.nrows() {
        return None;
    }
    let x: Vec<f64> = request.data.column(*feature_col).iter().copied().collect();
    let y: Vec<f64> = request.y.iter().copied().collect();
    let w: Vec<f64> = request.weights.iter().copied().collect();
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // The diffuse polynomial null space consumes `order` innovations; the scan
    // needs at least one proper innovation beyond them to profile σ².
    let mut sorted = x.clone();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    if sorted.len() < order + 1 {
        return None;
    }
    Some(SplineScanInputs { x, y, w, order })
}


/// Formula-level entry for the exact O(n) cubic-smoothing-spline fast path.
///
/// Materializes the formula exactly like [`fit_from_formula`], then runs the
/// [`spline_scan_fast_path`] detection on the resulting standard request.
/// When detection fires the fit is routed through
/// [`crate::solver::spline_scan::fit_spline_scan`] — the exact diffuse
/// REML Kalman/RTS scan — and the full in-memory posterior
/// ([`crate::solver::spline_scan::SplineScanFit`]: knots, smoothed
/// states, pointwise variances, lag-one gains, σ², log λ, exact EDF, and an
/// exact `predict`) is returned. `Ok(None)` means the model is not the
/// scan-eligible shape and the caller should use the dense
/// [`fit_from_formula`] path; this keeps every persistence-bearing consumer
/// (model save, CLI, FFI) transparently on the dense fit, whose saved payload
/// the scan does not yet have a schema for.
pub fn fit_spline_scan_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<crate::solver::spline_scan::SplineScanFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = spline_scan_fast_path(&request) else {
        return Ok(None);
    };
    crate::solver::spline_scan::fit_spline_scan(&inputs.x, &inputs.y, &inputs.w, inputs.order)
        .map(Some)
        .map_err(|reason| WorkflowError::IntegrationFailed { reason })
}


/// Inputs extracted by [`residual_cascade_fast_path`] for the O(n log n)
/// multiresolution residual-cascade smooth
/// ([`crate::solver::residual_cascade::fit_residual_cascade`]).
pub struct ResidualCascadeInputs {
    /// One slice per coordinate axis (2 or 3) of the single scattered smooth.
    pub coords: Vec<Vec<f64>>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Per-axis positive metric scaling `diag(metric)` of `z = diag(metric)·x`.
    pub metric: Vec<f64>,
    /// Sobolev smoothness order `s` of the multilevel Wendland-(3,1) prior,
    /// clamped into the native-space window `(d/2, (d+3)/2]` (issue caveat 1).
    pub sobolev_s: f64,
}


/// Derived dense-kernel cliff: the cascade auto-route fires only once the dense
/// radial basis the smooth would otherwise use has SATURATED at its center cap
/// (`default_num_centers == K_MAX`), so the dense `O(n·K² + K³)` kernel solve
/// can no longer grow resolution with `n` and the streaming cascade's
/// `O(n·polylog)` is the only path that keeps improving. This is the structural
/// "past the dense-kernel cliff" condition the issue names — derived from the
/// dense sizing rule, NOT a magic n constant or a user flag.
fn past_dense_kernel_cliff(n: usize, d: usize) -> bool {
    // `default_num_centers` clamps to K_MAX = 2000; equality means the dense
    // basis is pinned at the cap and cannot densify further with n.
    const DENSE_CENTER_CAP: usize = 2000;
    crate::terms::basis::default_num_centers(n, d) >= DENSE_CENTER_CAP
}


/// Map a Duchon/Matérn smoothness order onto the cascade's Sobolev order,
/// clamped into the Wendland-(3,1) native window `(d/2, (d+3)/2]` (issue
/// caveat 1: the multilevel frame can only represent up to `H^{(d+3)/2}`).
fn cascade_sobolev_order(requested: f64, d: usize) -> f64 {
    let lo = d as f64 / 2.0;
    let hi = (d as f64 + 3.0) / 2.0;
    // Nudge strictly inside the open lower bound when the request lands on it.
    let eps = 1e-6 * (hi - lo);
    requested.clamp(lo + eps, hi)
}


/// Detection seam for the O(n log n) multiresolution residual-cascade fast path
/// (issue #1032).
///
/// This mirrors [`spline_scan_fast_path`] in shape but carries one CRITICAL
/// difference dictated by the issue: the cascade is **not** the same posterior
/// as the Duchon/Matérn term it stands in for (a different finite basis — the
/// multilevel Wendland frame, not the reduced-rank radial kernel). So unlike
/// the 1-D scan, which silently swaps an identical posterior, this path must
/// only fire as an explicit alternative estimator on the structural signature
/// the issue names, never as a transparent replacement. It returns `Some` only
/// when ALL of the following hold:
/// - family is Gaussian + identity link (the scattered low-d smooth the
///   cascade solves);
/// - none of the exotic-link / constraint / Firth / Kronecker / coefficient-
///   group / hyperprior machinery is engaged;
/// - the model is exactly one smooth term — no linear terms, no random
///   effects, no by-variables;
/// - that smooth is a scattered radial spatial smooth (`Duchon` or `Matern`)
///   over `d ∈ {2, 3}` coordinates with no shape constraint;
/// - the offset is identically zero and every weight is finite and positive;
/// - `n` is past the derived dense-kernel cliff
///   ([`past_dense_kernel_cliff`]) — below it the dense radial path is both
///   exact-posterior and cheap, so there is no reason to change estimators.
///
/// The returned [`ResidualCascadeInputs`] carry a unit per-axis metric (the
/// spec's isotropic radial distance); the quasi-uniformity guard inside
/// [`crate::solver::residual_cascade::fit_residual_cascade`] (issue caveat 2)
/// is the no-regression gate that refuses the iterative solve — and forces the
/// caller back to the dense path — when a near-degenerate metric would break
/// the BPX iteration bound.
pub fn residual_cascade_fast_path(
    request: &StandardFitRequest<'_>,
) -> Option<ResidualCascadeInputs> {
    if !request.family.is_gaussian_identity() {
        return None;
    }
    if request.wiggle.is_some()
        || request.latent_coord.is_some()
        || !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        return None;
    }
    let options = &request.options;
    if options.latent_cloglog.is_some()
        || options.mixture_link.is_some()
        || options.sas_link.is_some()
        || options.linear_constraints.is_some()
        || options.adaptive_regularization.is_some()
        || options.kronecker_penalty_system.is_some()
        || options.kronecker_factored.is_some()
        || options.firth_bias_reduction
        || !options.nullspace_dims.is_empty()
    {
        return None;
    }
    let spec = &request.spec;
    if !spec.linear_terms.is_empty()
        || !spec.random_effect_terms.is_empty()
        || spec.smooth_terms.len() != 1
    {
        return None;
    }
    let term = &spec.smooth_terms[0];
    if !matches!(term.shape, crate::smooth::ShapeConstraint::None)
        || term.joint_null_rotation.is_some()
    {
        return None;
    }
    // Only scattered radial spatial smooths (Duchon / Matérn) over 2–3 axes.
    // The Duchon spectral power `p + s` and the Matérn order set the requested
    // Sobolev smoothness; both clamp into the Wendland native window.
    let (feature_cols, requested_s) = match &term.basis {
        crate::smooth::SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => {
            // Pure-Duchon native order is `p + s` (kernel exponent 2(p+s)−d);
            // the multilevel frame targets the same continuum smoothness. `p`
            // is the polynomial nullspace degree, `s` the spectral power.
            let p = match spec.nullspace_order {
                crate::basis::DuchonNullspaceOrder::Zero => 0.0,
                crate::basis::DuchonNullspaceOrder::Linear => 1.0,
                crate::basis::DuchonNullspaceOrder::Degree(k) => k as f64,
            };
            (feature_cols, spec.power + p)
        }
        crate::smooth::SmoothBasisSpec::Matern {
            feature_cols, spec, ..
        } => {
            // Matérn smoothness ν sets native Sobolev order ν + d/2; the cascade
            // frame represents up to (d+3)/2, so the clamp below applies the
            // ceiling. (d is known just below from feature_cols.)
            let nu = spec.nu.half_integer_value();
            (feature_cols, nu + feature_cols.len() as f64 / 2.0)
        }
        _ => return None,
    };
    let d = feature_cols.len();
    if !(2..=3).contains(&d) {
        return None;
    }
    if request.offset.iter().any(|&v| v != 0.0) {
        return None;
    }
    if request.weights.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return None;
    }
    let n = request.y.len();
    if n != request.data.nrows() || feature_cols.iter().any(|&c| c >= request.data.ncols()) {
        return None;
    }
    if !past_dense_kernel_cliff(n, d) {
        return None;
    }
    let coords: Vec<Vec<f64>> = feature_cols
        .iter()
        .map(|&c| request.data.column(c).iter().copied().collect())
        .collect();
    let y: Vec<f64> = request.y.iter().copied().collect();
    let w: Vec<f64> = request.weights.iter().copied().collect();
    if coords
        .iter()
        .any(|axis| axis.iter().any(|v| !v.is_finite()))
        || y.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let metric = vec![1.0_f64; d];
    let sobolev_s = cascade_sobolev_order(requested_s, d);
    Some(ResidualCascadeInputs {
        coords,
        y,
        w,
        metric,
        sobolev_s,
    })
}


/// Formula-level library entry for the O(n log n) residual-cascade fast path
/// (issue #1032).
///
/// Materializes the formula exactly like [`fit_from_formula`], runs the
/// [`residual_cascade_fast_path`] detection, and — when it fires AND the
/// quasi-uniformity guard inside the cascade certifies the metric — returns the
/// certified [`ResidualCascadeFit`](crate::solver::residual_cascade::ResidualCascadeFit).
/// `Ok(None)` means EITHER the model is not the cascade-eligible shape OR the
/// quasi-uniformity guard rejected the metric; in both cases the caller falls
/// back to the dense [`fit_from_formula`] path (the cascade is a different
/// posterior, so the fallback is a genuine estimator choice, never a silent
/// swap). This keeps every persistence-bearing consumer on the dense fit until
/// the cascade payload schema lands.
pub fn fit_residual_cascade_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<Option<crate::solver::residual_cascade::ResidualCascadeFit>, WorkflowError> {
    let mat = materialize(formula, data, config)?;
    let FitRequest::Standard(request) = mat.request else {
        return Ok(None);
    };
    let Some(inputs) = residual_cascade_fast_path(&request) else {
        return Ok(None);
    };
    let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
    match crate::solver::residual_cascade::fit_residual_cascade(
        &coord_refs,
        &inputs.y,
        &inputs.w,
        &inputs.metric,
        inputs.sobolev_s,
    ) {
        Ok(fit) => Ok(Some(fit)),
        // The quasi-uniformity guard (caveat 2) and any degenerate-design
        // signal both surface as a build/solve error; treat them as "not
        // cascade-eligible" so the caller falls back to the dense kernel path
        // rather than failing the fit outright.
        Err(_) => Ok(None),
    }
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

    if let Some((left_col, right_col, event_col)) = parse_surv_interval_response(&parsed.response)? {
        if config.transformation_normal {
            return Err(WorkflowError::InvalidConfig {
                reason: "transformation_normal cannot be combined with a SurvInterval(...) response"
                    .to_string(),
            });
        }
        // Interval censoring `T ∈ (L, R]` is only defined for the latent
        // hazard-window survival likelihood, whose kernel carries the
        // `log[S(L) − S(R)]` interval contribution. Route the left boundary `L`
        // through the standard exit channel and the right boundary `R` through
        // the dedicated interval-right channel; `event_col` distinguishes
        // bracketed (interval) rows from right-censored rows beyond the last
        // inspection (which carry an infinite/sentinel `R`).
        materialize_survival(
            &parsed,
            data,
            &col_map,
            config,
            None,
            &left_col,
            &event_col,
            Some(&right_col),
        )
    } else if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
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
            None,
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
