use super::*;
use gam::ErrorCategory;

pub(crate) type CliResult<T> = Result<T, CliError>;

/// A failure the CLI reports on stderr before exiting with its category's
/// [`ErrorCategory::exit_code`].
///
/// Typed library errors keep the category their own type declares, so the exit
/// code names the same category as the Python exception class. The CLI's own
/// refusals (a flag conflict, a path it cannot read or write, a malformed saved
/// model) are all the invocation being wrong, so they are
/// [`ErrorCategory::Formula`], the code the argument parser already uses for a
/// malformed invocation.
#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{message}")]
    Message {
        message: String,
        advice: Option<String>,
        category: ErrorCategory,
    },
    #[error("{reason}")]
    ArgumentInvalid { reason: String },
    #[error("{reason}")]
    IncompatibleConfig { reason: String },
    #[error("{reason}")]
    FileWriteFailed { reason: String },
    #[error("{reason}")]
    Internal { reason: String },
}

impl CliError {
    pub(crate) fn advice(&self) -> Option<&str> {
        match self {
            Self::Message { advice, .. } => advice.as_deref(),
            Self::ArgumentInvalid { .. }
            | Self::IncompatibleConfig { .. }
            | Self::FileWriteFailed { .. }
            | Self::Internal { .. } => None,
        }
    }

    pub(crate) fn error_category(&self) -> ErrorCategory {
        match self {
            Self::Message { category, .. } => *category,
            Self::ArgumentInvalid { .. }
            | Self::IncompatibleConfig { .. }
            | Self::FileWriteFailed { .. } => ErrorCategory::Formula,
            Self::Internal { .. } => ErrorCategory::Internal,
        }
    }

    /// Prefix the message with the CLI step that failed, keeping the category
    /// and advice of the error underneath.
    pub(crate) fn context(self, step: &str) -> Self {
        Self::Message {
            message: format!("{step}: {self}"),
            advice: self.advice().map(str::to_string),
            category: self.error_category(),
        }
    }

    fn typed(message: String, advice: Option<String>, category: ErrorCategory) -> Self {
        Self::Message {
            message,
            advice,
            category,
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        // A bare string carries no typed identity and therefore no advice:
        // remediation is a property of the typed error that produced the
        // failure (`EstimationError::advice` and friends), never something
        // re-derived from the rendered text.
        Self::typed(message, None, ErrorCategory::Formula)
    }
}

impl From<&str> for CliError {
    fn from(message: &str) -> Self {
        Self::from(message.to_string())
    }
}

impl From<CliError> for String {
    fn from(err: CliError) -> Self {
        err.to_string()
    }
}

// Cross-module `?` cascade: typed library errors flow into `CliError` with
// the advice their own type declares, so the `help:` line the CLI prints is
// the same remediation the Python exception carries.

impl From<gam::inference::formula_dsl::FormulaDslError> for CliError {
    fn from(err: gam::inference::formula_dsl::FormulaDslError) -> Self {
        // Every formula-DSL failure is, from the CLI's point of view, an
        // argument-validation failure: the user-supplied formula string did
        // not parse / type-check / use a supported identifier.
        Self::ArgumentInvalid {
            reason: err.to_string(),
        }
    }
}

impl From<gam::data::DataError> for CliError {
    fn from(err: gam::data::DataError) -> Self {
        Self::typed(err.to_string(), err.advice(), err.error_category())
    }
}

impl From<WorkflowError> for CliError {
    fn from(err: WorkflowError) -> Self {
        Self::typed(err.to_string(), err.advice(), err.error_category())
    }
}

impl From<gam::estimate::EstimationError> for CliError {
    fn from(err: gam::estimate::EstimationError) -> Self {
        Self::typed(err.to_string(), err.advice(), err.error_category())
    }
}
