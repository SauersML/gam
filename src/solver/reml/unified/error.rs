use super::*;

pub(crate) fn reml_contract_panic(message: impl Into<String>) -> ! {
    std::panic::panic_any(message.into())
}


// ═══════════════════════════════════════════════════════════════════════════
//  Typed errors for the unified REML/LAML evaluator.
//
//  The evaluator and its helpers historically returned `Result<_, String>`.
//  Internally we now build typed errors at the leaves and convert at the
//  boundary via `From<RemlError> for String`, which is byte-equivalent to
//  the previous `format!(...)` strings so external callers continue to see
//  the same diagnostic text.
// ═══════════════════════════════════════════════════════════════════════════

/// Typed failure categories raised by the unified REML/LAML evaluator and
/// its outer-Hessian / penalty-root helpers.
///
/// Each variant carries a pre-formatted `reason` string so that the
/// `Display` impl is byte-equivalent to the original `format!(...)` text the
/// module emitted before the typed-error migration. External signatures
/// remain `Result<_, String>`; the boundary conversion goes through
/// `From<RemlError> for String`.
#[derive(Debug, Clone)]
pub enum RemlError {
    /// A length / shape disagreement between two views that should match
    /// (penalty coords vs Hessian dim, residual length vs operator dim,
    /// precomputed-correction count vs total, etc.).
    DimensionMismatch { reason: String },
    /// A scalar / vector / matrix entry that must be finite came back NaN
    /// or ±∞ (cost, gradient entry, Hessian entry, cross-trace entry).
    NonFiniteValue { reason: String },
    /// A correction path was invoked against an operator kernel that does
    /// not support it (scalar-only correction on a non-scalar kernel,
    /// callback correction on a non-callback kernel).
    InvalidKernelMode { reason: String },
    /// A caller violated the evaluator contract. These are not numerical
    /// failures; they mean an upstream solver presented an inner state with
    /// insufficient certificates for the requested derivative surface.
    ContractViolation { reason: String },
}


impl std::fmt::Display for RemlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemlError::DimensionMismatch { reason }
            | RemlError::NonFiniteValue { reason }
            | RemlError::InvalidKernelMode { reason }
            | RemlError::ContractViolation { reason } => f.write_str(reason),
        }
    }
}


impl std::error::Error for RemlError {}


impl From<RemlError> for String {
    fn from(err: RemlError) -> String {
        err.to_string()
    }
}
