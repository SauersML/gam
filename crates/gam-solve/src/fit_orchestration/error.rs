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
        }
    }
}
