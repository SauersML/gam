//! Custom-family error type and its String conversions.

use thiserror::Error;

use crate::{IdentifiabilityAudit, MapUniquenessError};

#[derive(Debug, Clone, Error)]
pub enum CustomFamilyError {
    #[error("custom-family invalid input in {context}: {reason}")]
    InvalidInput {
        context: &'static str,
        reason: String,
    },
    #[error("custom-family optimization error in {context}: {reason}")]
    Optimization {
        context: &'static str,
        reason: String,
    },
    #[error("{reason}")]
    DimensionMismatch { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    ConstraintViolation { reason: String },
    #[error("{reason}")]
    UnsupportedConfiguration { reason: String },
    #[error("{reason}")]
    BasisDecompositionFailed { reason: String },
    /// Pre-fit cross-block identifiability audit refused the fit. The
    /// joint design across `ParameterBlockSpec`s carries a rank
    /// deficiency that the post-`joint_null_rotation` absorption did
    /// not resolve: two or more blocks contribute the same direction,
    /// or a structural >2-way alias was detected without per-pair
    /// attribution. The full `IdentifiabilityAudit` is held so
    /// consumers (logs, structured-error sinks, the seed driver's
    /// classifier) can extract the alias pairs and the summary string
    /// without reparsing.
    #[error("identifiability audit refused the fit: {}", audit.summary)]
    IdentifiabilityFailure { audit: IdentifiabilityAudit },
    /// MAP estimate uniqueness condition `ker(J^T W J) ∩ ker(S) = {0}` is
    /// violated.  A null direction of `J^T W J` carries zero penalty
    /// curvature, so the posterior is flat along that direction and the
    /// MAP is non-unique.  The structured [`MapUniquenessError`] names the
    /// dominant block so the caller can add the missing penalty or remove
    /// the unpenalised direction.
    #[error("MAP estimate non-unique: {}", error)]
    MapUniquenessFailure { error: MapUniquenessError },
}

impl From<String> for CustomFamilyError {
    fn from(value: String) -> Self {
        Self::InvalidInput {
            context: "custom-family string boundary",
            reason: value,
        }
    }
}

impl From<CustomFamilyError> for String {
    fn from(value: CustomFamilyError) -> Self {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input_display_contains_context_and_reason() {
        let err = CustomFamilyError::InvalidInput {
            context: "my_context",
            reason: "something broke".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("my_context"), "message: {msg}");
        assert!(msg.contains("something broke"), "message: {msg}");
    }

    #[test]
    fn optimization_display_contains_context_and_reason() {
        let err = CustomFamilyError::Optimization {
            context: "outer_loop",
            reason: "diverged".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("outer_loop") && msg.contains("diverged"), "message: {msg}");
    }

    #[test]
    fn dimension_mismatch_displays_reason() {
        let err = CustomFamilyError::DimensionMismatch { reason: "3 vs 4".to_string() };
        assert_eq!(err.to_string(), "3 vs 4");
    }

    #[test]
    fn numerical_failure_displays_reason() {
        let err = CustomFamilyError::NumericalFailure { reason: "NaN detected".to_string() };
        assert_eq!(err.to_string(), "NaN detected");
    }

    #[test]
    fn from_string_creates_invalid_input_with_boundary_context() {
        let err = CustomFamilyError::from("string error".to_string());
        assert!(matches!(err, CustomFamilyError::InvalidInput { .. }));
        assert!(err.to_string().contains("string error"));
    }

    #[test]
    fn from_custom_family_error_for_string_uses_display() {
        let err = CustomFamilyError::NumericalFailure { reason: "singular".to_string() };
        let s = String::from(err);
        assert_eq!(s, "singular");
    }
}
