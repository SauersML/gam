use super::*;

pub(crate) trait CliCauseCountResult {
    fn into_cli_result(self) -> Result<usize, String>;
}

impl CliCauseCountResult for usize {
    fn into_cli_result(self) -> Result<usize, String> {
        Ok(self)
    }
}

impl<E: ToString> CliCauseCountResult for Result<usize, E> {
    fn into_cli_result(self) -> Result<usize, String> {
        self.map_err(|err| err.to_string())
    }
}

pub(crate) type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{message}")]
    Message {
        message: String,
        advice: Option<String>,
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
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        classify_cli_error(message)
    }
}

impl From<CliError> for String {
    fn from(err: CliError) -> Self {
        err.to_string()
    }
}

// Cross-module `?` cascade: typed library errors flow into `CliError` directly
// without losing their structured payload via the legacy `.to_string()` boundary.
// Each conversion routes the typed error into the most appropriate `CliError`
// variant. The `reason` text is preserved verbatim so user-visible messages
// stay byte-equivalent to the pre-cascade shape.

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
        // Data-loader failures land in the user-facing argument-validation
        // surface: the path / schema / columns the user pointed us at could
        // not be opened or parsed. The classifier still runs on the rendered
        // text in case it carries hints we want to dress up further.
        classify_cli_error(err.to_string())
    }
}

impl From<WorkflowError> for CliError {
    fn from(err: WorkflowError) -> Self {
        // The workflow layer is the bridge between user-supplied config /
        // formula / data and the solver. Its errors are routed through the
        // shared classifier so error text already carries CLI-friendly
        // wording, hints, and family-specific advice.
        classify_cli_error(err.to_string())
    }
}

impl From<gam::estimate::EstimationError> for CliError {
    fn from(err: gam::estimate::EstimationError) -> Self {
        // EstimationError is the solver's structured failure type. We route
        // it through the shared `classify_cli_error` so its hints and
        // model-overparameterisation breakdown stay user-facing identical
        // to the prior `.to_string()` boundary path.
        classify_cli_error(err.to_string())
    }
}

pub(crate) fn extract_quoted_field(message: &str) -> Option<String> {
    let mut it = message.match_indices('\'');
    let (start_q, _) = it.next()?;
    let start = start_q + '\''.len_utf8();
    let (end_q, _) = it.next()?;
    if end_q > start {
        Some(message[start..end_q].to_string())
    } else {
        None
    }
}

pub(crate) fn classify_invalid_tpsspec(lower: &str) -> Option<String> {
    if !lower.contains("thin-plate spline") {
        return None;
    }
    if lower.contains("requires at least d+1 knots") {
        return Some(
            "Invalid thin-plate model specification. Increase the number of centers/knots for this joint smooth or reduce its covariate dimension."
                .to_string(),
        );
    }
    if lower
        .contains("fewer unique covariate combinations than specified maximum degrees of freedom")
    {
        return Some(
            "Invalid thin-plate model specification. The requested basis is too large for the joint covariate support in this term; reduce the basis size or the joint smooth dimension."
                .to_string(),
        );
    }
    None
}

pub(crate) fn classify_cli_error(message: String) -> CliError {
    let lower = message.to_ascii_lowercase();
    let advice = if let Some(advice) = classify_invalid_tpsspec(&lower) {
        Some(advice)
    } else if lower.contains("separation") || lower.contains("perfectly separated") {
        let culprit = extract_quoted_field(&message);
        Some(match culprit {
            Some(col) => format!(
                "Detected (quasi-)separation likely driven by '{col}'. Try removing or regularizing that term, or switch link via link(type=...)."
            ),
            None => "Detected (quasi-)separation. Try removing the strongest predictor, adding stronger regularization, or switching link via link(type=...).".to_string(),
        })
    } else if lower.contains("rank deficient")
        || lower.contains("singular")
        || lower.contains("ill-conditioned")
        || lower.contains("cholesky")
    {
        let culprit = extract_quoted_field(&message);
        Some(match culprit {
            Some(col) => format!(
                "Matrix conditioning issue likely tied to '{col}'. Check collinearity/constant columns and reduce redundant smooth terms."
            ),
            None => "Matrix conditioning issue detected. Check for collinear/constant predictors and overly complex smooth bases.".to_string(),
        })
    } else if lower.contains("duchon") && lower.contains("2*(p+s)") {
        // A Duchon spline whose power is too low for the radial-kernel
        // derivative a given path needs (e.g. the exact two-block spatial /
        // transformation-normal joint, which differentiates the kernel at the
        // origin). The basis-layer message already states the minimum
        // admissible power; surface that as the actionable advice rather than
        // mistaking the literal "dimension=N" for a data-shape mismatch.
        Some(
            "Duchon smooth is not smooth enough for this fit path. Raise its `power=...` to the minimum stated in the error above, or reduce the joint smooth's dimension."
                .to_string(),
        )
    } else if lower.contains("mismatch")
        || lower.contains("dimension")
        || lower.contains("shape mismatch")
    {
        Some(
            "Shape mismatch detected. Verify the new data has the same columns/types as training and that formula terms match."
                .to_string(),
        )
    } else {
        None
    };
    CliError::Message { message, advice }
}
