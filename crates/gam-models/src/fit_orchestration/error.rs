use std::sync::Arc;

use gam_problem::{CustomFamilyError, EstimationError, FailureCategory};

use crate::survival::marginal_slope::SurvivalMarginalSlopeError;

pub(crate) trait WorkflowCauseCountResult {
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

/// Why a marginal-slope fit refuses the link its main formula names. The calibrated
/// de-nested kernel is probit-only, so every other request is refused with the rule it
/// breaks rather than fitted as probit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalSlopeLinkRefusal {
    /// `link(type=flexible(...))`: link deviations are learned by `linkwiggle(...)` around a
    /// fixed base link.
    Flexible,
    /// A base link other than probit, or a blend of links.
    NonProbit,
    /// A link parameter that only `link(type=<requires>)` reads.
    ForeignParameter {
        parameter: &'static str,
        requires: &'static str,
    },
}

/// Why a transformation-normal fit refuses a request. The CTN model is its own response
/// model, so a request that also selects another response model is refused by the control
/// that selects it rather than fitted as either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformationNormalConflict {
    /// A `SurvInterval(...)` response selects the interval-censored survival likelihood.
    SurvIntervalResponse,
    /// A `Surv(...)` response selects a survival likelihood.
    SurvResponse,
    /// `noise_formula` selects a location-scale model.
    NoiseFormula,
    /// A marginal-slope family, `slope_formula`, `z_column` or `ctn_stage1` selects a
    /// marginal-slope model.
    MarginalSlopeControls,
}

/// Typed error category for the `solver::fit_orchestration` materialization and
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
    /// A fit's solve failed. The typed engine error that stopped it is kept
    /// whole, under the context each orchestration layer added (#2937).
    Fit(FitFailure),
    /// Training data failed the shared fit-boundary contract.
    InvalidData { column: String, problem: String },
    /// A spatial basis could not be certified at its current resolution and
    /// the next information-bearing expansion could not be fitted. Carries the
    /// attempted resolution and underlying evidence instead of returning the
    /// last under-resolved fit as if it were complete.
    SpatialUnderresolved {
        term: String,
        current_centers: usize,
        attempted_centers: usize,
        reason: String,
        /// The certification refit's own failure, when a refit is what failed.
        /// It decides the failure's category; `reason` renders it (#2937).
        refit_failure: Option<Box<WorkflowError>>,
    },
    /// Formula parsing / term-resolution failed before materialization; the
    /// source retains the parser-layer category and argument context.
    FormulaDsl {
        context: &'static str,
        source: gam_terms::inference::formula_dsl::FormulaDslError,
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
    /// A marginal-slope fit named a link its probit-only kernel cannot fit.
    MarginalSlopeLink {
        context: &'static str,
        refusal: MarginalSlopeLinkRefusal,
    },
    /// A transformation-normal fit named a control that selects another response model.
    TransformationNormalConflict {
        conflict: TransformationNormalConflict,
    },
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowError::InvalidConfig { reason }
            | WorkflowError::SchemaMismatch { reason }
            | WorkflowError::MissingDependency { reason } => f.write_str(reason),
            WorkflowError::Fit(failure) => std::fmt::Display::fmt(failure, f),
            WorkflowError::InvalidData { column, problem } => {
                write!(f, "column '{column}' {problem}")
            }
            WorkflowError::SpatialUnderresolved {
                term,
                current_centers,
                attempted_centers,
                reason,
                ..
            } => write!(
                f,
                "spatial term '{term}' remains under-resolution-uncertain at {current_centers} \
                 centers: the {attempted_centers}-center certification refit failed ({reason})"
            ),
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
            WorkflowError::MarginalSlopeLink { context, refusal } => match refusal {
                MarginalSlopeLinkRefusal::Flexible => write!(
                    f,
                    "{context} does not accept flexible(...) inside link(); use link(type=<base-link>) plus linkwiggle(...) to learn anchored link deviations"
                ),
                MarginalSlopeLinkRefusal::NonProbit => write!(
                    f,
                    "{context} requires link(type=probit); non-probit marginal-slope links are not supported by the calibrated de-nested probit kernel"
                ),
                MarginalSlopeLinkRefusal::ForeignParameter {
                    parameter,
                    requires,
                } => write!(
                    f,
                    "link({parameter}=...) requires link(type={requires}), which {context} does not support"
                ),
            },
            WorkflowError::TransformationNormalConflict { conflict } => {
                let control = match conflict {
                    TransformationNormalConflict::SurvIntervalResponse => {
                        "a SurvInterval(...) response"
                    }
                    TransformationNormalConflict::SurvResponse => "a Surv(...) response",
                    TransformationNormalConflict::NoiseFormula => "noise_formula",
                    TransformationNormalConflict::MarginalSlopeControls => {
                        "marginal-slope family controls"
                    }
                };
                write!(f, "transformation_normal cannot be combined with {control}")
            }
        }
    }
}

impl std::error::Error for WorkflowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WorkflowError::FormulaDsl { source, .. } => Some(source),
            // Renders exactly its failure, so it is transparent to the chain.
            WorkflowError::Fit(failure) => failure.source(),
            WorkflowError::InvalidConfig { .. }
            | WorkflowError::SchemaMismatch { .. }
            | WorkflowError::MissingDependency { .. }
            | WorkflowError::InvalidData { .. }
            | WorkflowError::SpatialUnderresolved { .. }
            | WorkflowError::ColumnNotFound { .. }
            | WorkflowError::MarginalSlopeLink { .. }
            | WorkflowError::TransformationNormalConflict { .. } => None,
        }
    }
}

impl From<WorkflowError> for String {
    fn from(err: WorkflowError) -> String {
        err.to_string()
    }
}

impl WorkflowError {
    /// The remediation a user can act on, when the failure has one; the single
    /// source of the advice every front end prints beside the error. The
    /// schema variant is the typed form of what `gam_data::DataError::advice`
    /// says for the same condition.
    #[must_use]
    pub fn advice(&self) -> Option<String> {
        match self {
            Self::SchemaMismatch { .. } => Some(
                "Verify the new data has the same columns and types as the training data \
                 and that the formula terms match."
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// The fixed category of this failure (#2937). Exhaustive with no wildcard
    /// arm: a new variant is categorized by whoever adds it.
    #[must_use]
    pub fn failure_category(&self) -> FailureCategory {
        match self {
            Self::Fit(failure) => failure.category(),
            Self::SpatialUnderresolved { refit_failure, .. } => refit_failure
                .as_deref()
                // No refit failed: the resolution search ran out of admissible
                // centers without certifying the basis.
                .map_or(FailureCategory::Convergence, Self::failure_category),
            Self::InvalidConfig { .. }
            | Self::SchemaMismatch { .. }
            | Self::MissingDependency { .. }
            | Self::InvalidData { .. }
            | Self::FormulaDsl { .. }
            | Self::ColumnNotFound { .. }
            // A marginal-slope link the fit cannot declare: a configuration refusal.
            | Self::MarginalSlopeLink { .. }
            // Controls that select another response model: a configuration refusal.
            | Self::TransformationNormalConflict { .. } => FailureCategory::Input,
        }
    }

    /// The `Enum::Variant` name of the error that decided this failure, the one
    /// a front end prints beside the message (#2937). A failure that only
    /// wraps another is named by what it wraps.
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Fit(failure) => failure.variant_name(),
            Self::SpatialUnderresolved {
                refit_failure: Some(refit_failure),
                ..
            } => refit_failure.variant_name(),
            Self::SpatialUnderresolved {
                refit_failure: None,
                ..
            } => "WorkflowError::SpatialUnderresolved",
            Self::InvalidConfig { .. } => "WorkflowError::InvalidConfig",
            Self::SchemaMismatch { .. } => "WorkflowError::SchemaMismatch",
            Self::MissingDependency { .. } => "WorkflowError::MissingDependency",
            Self::InvalidData { .. } => "WorkflowError::InvalidData",
            Self::FormulaDsl { .. } => "WorkflowError::FormulaDsl",
            Self::ColumnNotFound { .. } => "WorkflowError::ColumnNotFound",
            Self::MarginalSlopeLink { .. } => "WorkflowError::MarginalSlopeLink",
            Self::TransformationNormalConflict { .. } => {
                "WorkflowError::TransformationNormalConflict"
            }
        }
    }
}

/// A fit's solve failure with the typed error that stopped it (#2937).
///
/// The `fit_*_model` helpers returned `Result<_, String>`, and `fit_model`
/// wrapped every one of those strings as an integration failure, so a refused
/// start, a stalled outer search and a numerical refusal all reached Python as
/// `IntegrationError`. A `FitFailure` keeps the engine error whole under the
/// context the layers above it add, so a front end can name the variant and
/// select an exception class from its category without reading the text.
#[derive(Debug, Clone)]
pub enum FitFailure {
    /// A typed estimation error from the solver.
    Estimation(Arc<EstimationError>),
    /// A typed custom-family error.
    CustomFamily(CustomFamilyError),
    /// A typed survival marginal-slope error.
    SurvivalMarginalSlope(SurvivalMarginalSlopeError),
    /// A nested fit through the workflow boundary failed, e.g. a CTN stage fit.
    Workflow(Box<WorkflowError>),
    /// A failure raised in orchestration code, categorized where it is raised.
    /// There is no conversion from text: a helper's text whose failures span
    /// categories is raised as [`FailureCategory::Unclassified`] through
    /// [`Self::unclassified`], at a call site that names it (#2937).
    Raised {
        category: FailureCategory,
        reason: String,
    },
    /// Context a layer put in front of a failure it did not produce. Renders
    /// `"{context}: {source}"`, the text those layers used to `format!`.
    Context {
        context: String,
        source: Box<FitFailure>,
    },
    /// Evidence a layer appended after a failure it did not produce, such as a
    /// rescue that was attempted. Renders `"{source}; {note}"`.
    Annotated {
        source: Box<FitFailure>,
        note: String,
    },
}

impl FitFailure {
    /// A failure raised in orchestration code under the category it belongs to.
    #[must_use]
    pub fn raised(category: FailureCategory, reason: impl Into<String>) -> Self {
        Self::Raised {
            category,
            reason: reason.into(),
        }
    }

    /// The caller's configuration, data or problem size was refused.
    #[must_use]
    pub fn input(reason: impl Into<String>) -> Self {
        Self::raised(FailureCategory::Input, reason)
    }

    /// State the engine built from validated input disagreed with itself.
    #[must_use]
    pub fn invariant(reason: impl Into<String>) -> Self {
        Self::raised(FailureCategory::Invariant, reason)
    }

    /// A numerical step failed on the fit's own iterates.
    #[must_use]
    pub fn numerical(reason: impl Into<String>) -> Self {
        Self::raised(FailureCategory::Numerical, reason)
    }

    /// A quadrature or compression did not reach its tolerance.
    #[must_use]
    pub fn integration(reason: impl Into<String>) -> Self {
        Self::raised(FailureCategory::Integration, reason)
    }

    /// Text from a helper whose failures span categories. Each call site is
    /// named, so what is left untyped stays countable (#2937).
    #[must_use]
    pub fn unclassified(reason: impl Into<String>) -> Self {
        Self::raised(FailureCategory::Unclassified, reason)
    }

    /// Put `context` in front of this failure without changing what it is.
    #[must_use]
    pub fn context(self, context: impl Into<String>) -> Self {
        Self::Context {
            context: context.into(),
            source: Box::new(self),
        }
    }

    /// Append `note` after this failure without changing what it is.
    #[must_use]
    pub fn annotated(self, note: impl Into<String>) -> Self {
        Self::Annotated {
            source: Box::new(self),
            note: note.into(),
        }
    }

    /// This failure as the fit boundary hands it to the caller (gam#2943).
    ///
    /// An inner solve that never certified its mode is a trial-point refusal
    /// while an outer search can still step away from it. Once the fit has
    /// ended no search is left, so the boundary names that refusal
    /// [`CustomFamilyError::FitEndedWithoutCertifiedInnerMode`], whether it
    /// arrives bare or as the last inner refusal an outer smoothing failure
    /// carries. This is the only place that variant is minted. The outer
    /// verdict it was read through stays in front as context, and every other
    /// failure is returned unchanged.
    #[must_use]
    pub fn ending_the_fit(self) -> Self {
        match self {
            Self::Context { context, source } => Self::Context {
                context,
                source: Box::new(source.ending_the_fit()),
            },
            Self::Annotated { source, note } => Self::Annotated {
                source: Box::new(source.ending_the_fit()),
                note,
            },
            Self::CustomFamily(err) => match Self::terminal_inner_refusal(&err).cloned() {
                None => Self::CustomFamily(err),
                Some(refusal) => {
                    let bare = matches!(err, CustomFamilyError::InnerSolveNotConverged { .. });
                    Self::ended_by(refusal, (!bare).then(|| err.to_string()))
                }
            },
            Self::Estimation(err) => {
                let refusal = Self::custom_family_leaf(&err)
                    .and_then(Self::terminal_inner_refusal)
                    .cloned();
                match refusal {
                    None => Self::Estimation(err),
                    Some(refusal) => {
                        let bare = matches!(
                            err.as_ref(),
                            EstimationError::CustomFamily(
                                CustomFamilyError::InnerSolveNotConverged { .. }
                            )
                        );
                        Self::ended_by(refusal, (!bare).then(|| err.to_string()))
                    }
                }
            }
            Self::Workflow(err) => match *err {
                WorkflowError::Fit(failure) => {
                    Self::Workflow(Box::new(WorkflowError::Fit(failure.ending_the_fit())))
                }
                other => Self::Workflow(Box::new(other)),
            },
            other @ (Self::SurvivalMarginalSlope(_) | Self::Raised { .. }) => other,
        }
    }

    /// The inner refusal a custom-family failure ends in: the refusal itself,
    /// or for an outer smoothing failure the search's most recent uncertified
    /// inner solve (`search_inner_refusal`), else its last evaluation's refusal,
    /// read through nested outer failures. `None` when the fit did not end on
    /// one, including a refusal already minted as the fit-ending variant.
    fn terminal_inner_refusal(err: &CustomFamilyError) -> Option<&CustomFamilyError> {
        match err {
            CustomFamilyError::InnerSolveNotConverged { .. } => Some(err),
            CustomFamilyError::OuterSmoothingFailed {
                search_inner_refusal,
                last_refusal,
                ..
            } => search_inner_refusal
                .as_deref()
                .and_then(Self::terminal_inner_refusal)
                .or_else(|| last_refusal.as_deref().and_then(Self::terminal_inner_refusal)),
            _ => None,
        }
    }

    /// The custom-family error an engine error carries, read through fatal
    /// outer-evaluation wrappers. Unlike
    /// [`EstimationError::innermost_estimation_error`] it does not descend into an
    /// outer smoothing failure's own verdict, which would pass over the refusals
    /// that failure carries.
    fn custom_family_leaf(err: &EstimationError) -> Option<&CustomFamilyError> {
        match err {
            EstimationError::CustomFamily(family) => Some(family),
            EstimationError::OuterObjectiveEvaluationFailed { source, .. } => {
                source.estimation_error().and_then(Self::custom_family_leaf)
            }
            _ => None,
        }
    }

    /// The fit-ending variant for `refusal`, under the outer verdict it was
    /// read through when there is one.
    fn ended_by(refusal: CustomFamilyError, outer_verdict: Option<String>) -> Self {
        let ended =
            Self::CustomFamily(CustomFamilyError::fit_ended_without_certified_inner_mode(refusal));
        match outer_verdict {
            Some(verdict) => ended.context(verdict),
            None => ended,
        }
    }

    /// The fixed category of the error this failure ends in.
    #[must_use]
    pub fn category(&self) -> FailureCategory {
        match self {
            Self::Context { source, .. } | Self::Annotated { source, .. } => source.category(),
            Self::Estimation(err) => err.failure_category(),
            Self::CustomFamily(err) => err.failure_category(),
            Self::SurvivalMarginalSlope(err) => err.failure_category(),
            Self::Workflow(err) => err.failure_category(),
            Self::Raised { category, .. } => *category,
        }
    }

    /// The `Enum::Variant` name of the error this failure ends in.
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Context { source, .. } | Self::Annotated { source, .. } => source.variant_name(),
            Self::Estimation(err) => err.variant_name(),
            Self::CustomFamily(err) => err.variant_name(),
            Self::SurvivalMarginalSlope(err) => err.variant_name(),
            Self::Workflow(err) => err.variant_name(),
            Self::Raised { category, .. } => match category {
                FailureCategory::Convergence => "FitFailure::Convergence",
                FailureCategory::StartupSeeds => "FitFailure::StartupSeeds",
                FailureCategory::Invariant => "FitFailure::Invariant",
                FailureCategory::Input => "FitFailure::Input",
                FailureCategory::Numerical => "FitFailure::Numerical",
                FailureCategory::Integration => "FitFailure::Integration",
                FailureCategory::Unclassified => "FitFailure::Unclassified",
            },
        }
    }

    /// The typed estimation error this failure ends in, when it ends in one,
    /// seen through the wrappers that only carry it.
    #[must_use]
    pub fn estimation_error(&self) -> Option<&EstimationError> {
        match self {
            Self::Context { source, .. } | Self::Annotated { source, .. } => {
                source.estimation_error()
            }
            Self::Estimation(err) => Some(err.innermost_estimation_error()),
            Self::CustomFamily(CustomFamilyError::OuterSmoothingFailed { outer_error, .. }) => {
                Some(outer_error.innermost_estimation_error())
            }
            Self::Workflow(err) => match err.as_ref() {
                WorkflowError::Fit(failure) => failure.estimation_error(),
                _ => None,
            },
            Self::CustomFamily(_) | Self::SurvivalMarginalSlope(_) | Self::Raised { .. } => None,
        }
    }

    /// The typed facts of the uncertified inner solve this failure ends in,
    /// when the fit ended without a certified inner mode (gam#2943), seen
    /// through the wrappers that only carry it. The fit boundary mints that
    /// variant on a custom-family leaf, which [`Self::estimation_error`] does
    /// not see.
    #[must_use]
    pub fn terminal_inner_mode_evidence(&self) -> Option<gam_problem::TerminalInnerModeEvidence<'_>> {
        match self {
            Self::Context { source, .. } | Self::Annotated { source, .. } => {
                source.terminal_inner_mode_evidence()
            }
            Self::CustomFamily(err) => err.terminal_inner_mode_evidence(),
            Self::Estimation(err) => match err.innermost_estimation_error() {
                EstimationError::CustomFamily(err) => err.terminal_inner_mode_evidence(),
                _ => None,
            },
            Self::Workflow(err) => match err.as_ref() {
                WorkflowError::Fit(failure) => failure.terminal_inner_mode_evidence(),
                _ => None,
            },
            Self::SurvivalMarginalSlope(_) | Self::Raised { .. } => None,
        }
    }

    /// The message chain, outermost first: each layer's context, then the
    /// message of the error that stopped the fit, then the notes appended after
    /// it, innermost first.
    #[must_use]
    pub fn causes(&self) -> Vec<String> {
        let mut causes = Vec::new();
        let mut notes = Vec::new();
        let mut current = self;
        loop {
            match current {
                Self::Context { context, source } => {
                    causes.push(context.clone());
                    current = source;
                }
                Self::Annotated { source, note } => {
                    notes.push(note.clone());
                    current = source;
                }
                Self::Workflow(err) => match err.as_ref() {
                    WorkflowError::Fit(failure) => current = failure,
                    other => {
                        causes.push(other.to_string());
                        break;
                    }
                },
                leaf => {
                    causes.push(leaf.to_string());
                    break;
                }
            }
        }
        causes.extend(notes.into_iter().rev());
        causes
    }
}

impl std::fmt::Display for FitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Estimation(err) => std::fmt::Display::fmt(err, f),
            Self::CustomFamily(err) => std::fmt::Display::fmt(err, f),
            Self::SurvivalMarginalSlope(err) => std::fmt::Display::fmt(err, f),
            Self::Workflow(err) => std::fmt::Display::fmt(err, f),
            Self::Raised { reason, .. } => f.write_str(reason),
            Self::Context { context, source } => write!(f, "{context}: {source}"),
            Self::Annotated { source, note } => write!(f, "{source}; {note}"),
        }
    }
}

impl std::error::Error for FitFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Context { source, .. } | Self::Annotated { source, .. } => Some(source.as_ref()),
            // The leaf wrappers render exactly their error, so they are
            // transparent to the chain.
            Self::Estimation(err) => err.source(),
            Self::CustomFamily(err) => err.source(),
            Self::SurvivalMarginalSlope(err) => err.source(),
            Self::Workflow(err) => err.source(),
            Self::Raised { .. } => None,
        }
    }
}

impl From<EstimationError> for FitFailure {
    fn from(err: EstimationError) -> Self {
        Self::Estimation(Arc::new(err))
    }
}

impl From<CustomFamilyError> for FitFailure {
    fn from(err: CustomFamilyError) -> Self {
        Self::CustomFamily(err)
    }
}

impl From<SurvivalMarginalSlopeError> for FitFailure {
    fn from(err: SurvivalMarginalSlopeError) -> Self {
        Self::SurvivalMarginalSlope(err)
    }
}

/// A term-design construction refusal, under the category the engine gives it
/// as [`EstimationError::BasisError`], with the basis error's own text.
impl From<gam_problem::BasisError> for FitFailure {
    fn from(err: gam_problem::BasisError) -> Self {
        let reason = err.to_string();
        Self::raised(EstimationError::from(err).failure_category(), reason)
    }
}

/// A survival location-scale refusal, under its variant's category, with its
/// own text.
impl From<crate::survival::location_scale::SurvivalLocationScaleError> for FitFailure {
    fn from(err: crate::survival::location_scale::SurvivalLocationScaleError) -> Self {
        Self::raised(err.failure_category(), err.to_string())
    }
}

/// A latent survival or binary refusal, under its variant's category, with its
/// own text.
impl From<crate::survival::latent::LatentSurvivalError> for FitFailure {
    fn from(err: crate::survival::latent::LatentSurvivalError) -> Self {
        Self::raised(err.failure_category(), err.to_string())
    }
}

impl From<WorkflowError> for FitFailure {
    fn from(err: WorkflowError) -> Self {
        match err {
            WorkflowError::Fit(failure) => failure,
            other => Self::Workflow(Box::new(other)),
        }
    }
}

impl From<FitFailure> for WorkflowError {
    fn from(failure: FitFailure) -> Self {
        match failure {
            FitFailure::Workflow(err) => *err,
            failure => Self::Fit(failure),
        }
    }
}

#[cfg(test)]
mod fit_failure_tests {
    use super::*;

    fn seeds_refused() -> EstimationError {
        EstimationError::StartupSeedsRefused(
            "no candidate seeds passed outer startup validation (custom family):".to_string(),
        )
    }

    #[test]
    fn every_category_keeps_its_variant_through_the_fit_model_boundary_2937() {
        let cases = vec![
            (
                FitFailure::from(EstimationError::RemlOptimizationFailed("stalled".to_string())),
                FailureCategory::Convergence,
                "EstimationError::RemlOptimizationFailed",
            ),
            (
                FitFailure::from(seeds_refused()),
                FailureCategory::StartupSeeds,
                "EstimationError::StartupSeedsRefused",
            ),
            (
                FitFailure::from(EstimationError::FitResultInvariantViolated(
                    "UnifiedFitResult inference conditional covariance must match top-level \
                     covariance_conditional"
                        .to_string(),
                )),
                FailureCategory::Invariant,
                "EstimationError::FitResultInvariantViolated",
            ),
            (
                FitFailure::from(EstimationError::InvalidInput("bad weights".to_string())),
                FailureCategory::Input,
                "EstimationError::InvalidInput",
            ),
            (
                FitFailure::from(EstimationError::HessianNotPositiveDefinite {
                    min_eigenvalue: -1.0,
                }),
                FailureCategory::Numerical,
                "EstimationError::HessianNotPositiveDefinite",
            ),
            (
                FitFailure::from(SurvivalMarginalSlopeError::RootSolveFailed {
                    reason: "survival marginal-slope intercept solve failed".to_string(),
                }),
                FailureCategory::Numerical,
                "SurvivalMarginalSlopeError::RootSolveFailed",
            ),
            (
                FitFailure::from(SurvivalMarginalSlopeError::IntegrationFailed {
                    reason: "quadrature missed its tolerance".to_string(),
                }),
                FailureCategory::Integration,
                "SurvivalMarginalSlopeError::IntegrationFailed",
            ),
            (
                FitFailure::unclassified("a helper's prose"),
                FailureCategory::Unclassified,
                "FitFailure::Unclassified",
            ),
        ];
        for (failure, category, variant) in cases {
            let message = failure.to_string();
            // `fit_model` hands each helper's failure over as `WorkflowError::from`.
            let boundary = WorkflowError::from(failure);
            assert!(matches!(boundary, WorkflowError::Fit(_)), "{boundary}");
            assert_eq!(boundary.failure_category(), category, "{message}");
            assert_eq!(boundary.variant_name(), variant, "{message}");
            assert_eq!(boundary.to_string(), message, "the boundary rewrote the message");
        }
    }

    #[test]
    fn a_custom_family_search_failure_is_categorized_by_its_outer_verdict_2937() {
        let failure = FitFailure::from(CustomFamilyError::OuterSmoothingFailed {
            reason: format!(
                "outer smoothing optimization failed certified-fit validation after exhausting \
                 strategy fallbacks: {}",
                seeds_refused()
            ),
            last_refusal: None,
            search_inner_refusal: None,
            outer_error: Arc::new(seeds_refused()),
        });
        assert_eq!(failure.category(), FailureCategory::StartupSeeds);
        assert_eq!(failure.variant_name(), "EstimationError::StartupSeedsRefused");
        assert!(matches!(
            failure.estimation_error(),
            Some(EstimationError::StartupSeedsRefused(_))
        ));
    }

    /// A CTN prefit whose outer search declined a certified optimum and certified nothing
    /// in its place used to reach its caller as `IntegrationFailed` text (#2953, at
    /// 598f3da691). The transformation fit wraps the custom-family refusal under its
    /// context, and the variant must survive that and the fit-model boundary by type.
    #[test]
    fn a_ctn_prefit_dominated_plateau_refusal_keeps_its_variant_by_type_2953() {
        let refusal = EstimationError::DominatedCertifiedPlateau {
            context: "custom family".to_string(),
            kind: gam_problem::DominanceRefusalKind::IncumbentUnescapableSaddle,
            plateau_rho: vec![0.5],
            plateau_value: 73.02427,
            incumbent_rho: vec![-1.25],
            incumbent_value: 72.50521,
            incumbent_projected_grad_norm: Some(2.632e-1),
            gap: 9.541e-3,
            band: 1.088e-6,
            continuation: "declined another certified optimum at objective 7.302427e1".to_string(),
            terminal_refusal: Box::new(EstimationError::RemlOptimizationFailed(
                "not stationary".to_string(),
            )),
        };
        let failure = FitFailure::from(CustomFamilyError::OuterSmoothingFailed {
            reason: format!(
                "outer smoothing optimization failed certified-fit validation after exhausting \
                 strategy fallbacks: {refusal}"
            ),
            last_refusal: None,
            search_inner_refusal: None,
            outer_error: Arc::new(refusal),
        })
        .context("transformation fit failed");
        let boundary = WorkflowError::from(failure);
        assert_eq!(boundary.failure_category(), FailureCategory::Convergence);
        assert_eq!(boundary.variant_name(), "EstimationError::DominatedCertifiedPlateau");
        let WorkflowError::Fit(failure) = &boundary else {
            panic!("the CTN prefit refusal must stay a typed fit failure: {boundary}");
        };
        assert!(matches!(
            failure.estimation_error(),
            Some(EstimationError::DominatedCertifiedPlateau {
                kind: gam_problem::DominanceRefusalKind::IncumbentUnescapableSaddle,
                ..
            })
        ));
    }

    #[test]
    fn context_and_notes_keep_the_category_and_the_old_text_2937() {
        let failure = FitFailure::from(seeds_refused())
            .context("exact two-block spatial optimization failed")
            .annotated("the automatic Firth/Jeffreys rescue WAS attempted");
        assert_eq!(
            failure.to_string(),
            format!(
                "exact two-block spatial optimization failed: {}; the automatic Firth/Jeffreys \
                 rescue WAS attempted",
                seeds_refused()
            )
        );
        assert_eq!(failure.category(), FailureCategory::StartupSeeds);
        assert_eq!(
            failure.causes(),
            vec![
                "exact two-block spatial optimization failed".to_string(),
                seeds_refused().to_string(),
                "the automatic Firth/Jeffreys rescue WAS attempted".to_string(),
            ]
        );
        assert!(std::error::Error::source(&failure).is_some());
    }

    #[test]
    fn a_spatial_certification_refit_is_categorized_by_the_refit_failure_2937() {
        let refit = WorkflowError::from(FitFailure::from(seeds_refused()));
        let refused = WorkflowError::SpatialUnderresolved {
            term: "s(x)".to_string(),
            current_centers: 8,
            attempted_centers: 16,
            reason: refit.to_string(),
            refit_failure: Some(Box::new(refit)),
        };
        assert_eq!(refused.failure_category(), FailureCategory::StartupSeeds);
        assert_eq!(refused.variant_name(), "EstimationError::StartupSeedsRefused");
        let exhausted = WorkflowError::SpatialUnderresolved {
            term: "s(x)".to_string(),
            current_centers: 8,
            attempted_centers: 8,
            reason: "term EDF remains at its realized basis ceiling".to_string(),
            refit_failure: None,
        };
        assert_eq!(exhausted.failure_category(), FailureCategory::Convergence);
        assert_eq!(exhausted.variant_name(), "WorkflowError::SpatialUnderresolved");
    }

    #[test]
    fn nested_workflow_failures_do_not_stack_wrappers_2937() {
        let config = WorkflowError::InvalidConfig {
            reason: "unknown family".to_string(),
        };
        let lifted = FitFailure::from(config);
        assert_eq!(lifted.category(), FailureCategory::Input);
        assert!(matches!(
            WorkflowError::from(lifted),
            WorkflowError::InvalidConfig { .. }
        ));
        let failure = FitFailure::from(seeds_refused());
        assert!(matches!(
            FitFailure::from(WorkflowError::Fit(failure)),
            FitFailure::Estimation(_)
        ));
    }

    fn uncertified_inner_solve() -> CustomFamilyError {
        CustomFamilyError::InnerSolveNotConverged {
            cycles: 1,
            terminal: None,
            kkt_residual: Some(2.5e-1),
            kkt_tol: Some(1.0e-6),
            theta_dim: 4,
            rho_dim: 2,
            psi_dim: 0,
            cycle_budget: Some(1),
            carrying_block: Some("eta".to_string()),
        }
    }

    fn outer_smoothing_failed(
        last_refusal: Option<CustomFamilyError>,
        search_inner_refusal: Option<CustomFamilyError>,
    ) -> CustomFamilyError {
        CustomFamilyError::OuterSmoothingFailed {
            reason: "outer smoothing optimization failed certified-fit validation".to_string(),
            last_refusal: last_refusal.map(Box::new),
            search_inner_refusal: search_inner_refusal.map(Box::new),
            outer_error: Arc::new(EstimationError::RemlOptimizationFailed("stalled".to_string())),
        }
    }

    fn assert_ended_on_the_uncertified_solve(failure: &FitFailure) {
        assert_eq!(
            failure.variant_name(),
            "CustomFamilyError::FitEndedWithoutCertifiedInnerMode",
            "{failure}"
        );
        assert_eq!(failure.category(), FailureCategory::Convergence, "{failure}");
        let evidence = failure
            .terminal_inner_mode_evidence()
            .unwrap_or_else(|| panic!("the terminal solve's evidence must be readable: {failure}"));
        assert_eq!(evidence.cycles, 1, "{failure}");
        assert_eq!(evidence.cycle_budget, Some(1), "{failure}");
        assert_eq!(evidence.carrying_block, Some("eta"), "{failure}");
    }

    #[test]
    fn the_fit_boundary_names_a_fit_that_ended_on_an_uncertified_inner_solve_2943() {
        // A bare refusal becomes the variant, with nothing in front of it.
        let bare = FitFailure::from(uncertified_inner_solve()).ending_the_fit();
        assert_ended_on_the_uncertified_solve(&bare);
        assert!(matches!(bare, FitFailure::CustomFamily(_)), "{bare}");

        // Context a layer put in front stays in front.
        let folded = FitFailure::from(uncertified_inner_solve())
            .context("CTN fold 1 failed")
            .ending_the_fit();
        assert_ended_on_the_uncertified_solve(&folded);
        assert!(folded.to_string().starts_with("CTN fold 1 failed: "), "{folded}");

        // The search's last inner refusal, read through the outer verdict, which
        // stays in front as context.
        let outer = outer_smoothing_failed(
            Some(uncertified_inner_solve()),
            Some(uncertified_inner_solve()),
        );
        let outer_text = outer.to_string();
        let through_outer = FitFailure::from(outer.clone()).ending_the_fit();
        assert_ended_on_the_uncertified_solve(&through_outer);
        assert!(
            through_outer.to_string().starts_with(&outer_text),
            "{through_outer}"
        );

        // A search whose inner solve refused and whose later trials ran finite:
        // the last evaluation did not refuse, and the whole-search record still
        // names the inner solve that decided the fit.
        let after_finite_trials =
            FitFailure::from(outer_smoothing_failed(None, Some(uncertified_inner_solve())))
                .ending_the_fit();
        assert_ended_on_the_uncertified_solve(&after_finite_trials);

        // The same outer failure arriving as an engine error.
        let through_estimation =
            FitFailure::from(EstimationError::CustomFamily(outer)).ending_the_fit();
        assert_ended_on_the_uncertified_solve(&through_estimation);
    }

    #[test]
    fn the_fit_boundary_leaves_every_other_failure_as_it_was_2943() {
        let no_inner_refusal = FitFailure::from(outer_smoothing_failed(None, None));
        let before = (no_inner_refusal.variant_name(), no_inner_refusal.to_string());
        let after = no_inner_refusal.ending_the_fit();
        assert_eq!((after.variant_name(), after.to_string()), before);

        let seeds = FitFailure::from(seeds_refused()).ending_the_fit();
        assert_eq!(seeds.variant_name(), "EstimationError::StartupSeedsRefused");

        // A variant already minted is not wrapped a second time.
        let minted = FitFailure::from(CustomFamilyError::fit_ended_without_certified_inner_mode(
            uncertified_inner_solve(),
        ));
        let text = minted.to_string();
        let again = minted.ending_the_fit();
        assert_ended_on_the_uncertified_solve(&again);
        assert_eq!(again.to_string(), text);
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

impl From<crate::survival::lognormal_kernel::LognormalKernelError> for WorkflowError {
    fn from(err: crate::survival::lognormal_kernel::LognormalKernelError) -> Self {
        match err {
            crate::survival::lognormal_kernel::LognormalKernelError::InvalidSpec { reason } => {
                Self::InvalidConfig { reason }
            }
        }
    }
}

/// Cross-module cascade: a `FormulaDslError` raised inside `materialize` /
/// `fit_from_formula` (via `parse_formula`, `parse_surv_response`, etc.) flows
/// up with its parser-layer source attached instead of stringifying into a
/// generic workflow configuration bucket.
impl From<gam_terms::inference::formula_dsl::FormulaDslError> for WorkflowError {
    fn from(err: gam_terms::inference::formula_dsl::FormulaDslError) -> Self {
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
impl From<gam_terms::term_builder::TermBuilderError> for WorkflowError {
    fn from(err: gam_terms::term_builder::TermBuilderError) -> Self {
        use gam_terms::term_builder::TermBuilderError;
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
impl From<gam_data::DataError> for WorkflowError {
    fn from(err: gam_data::DataError) -> Self {
        use gam_data::DataError;
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
            DataError::DegenerateColumn { column, problem } => {
                Self::InvalidData { column, problem }
            }
        }
    }
}
