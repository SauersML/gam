use super::*;


// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

/// Typed errors emitted by the transformation-normal family pipeline.
///
/// Each variant carries a pre-formatted `reason` so `Display` is
/// byte-equivalent to the original `format!(...)` strings the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind (e.g. distinguish a degenerate
/// covariate design from a non-finite intermediate) without parsing text.
///
/// Public/trait boundaries (e.g. `CustomFamily::evaluate`) still return
/// `Result<_, String>`; the `From<TransformationNormalError> for String`
/// impl below provides the shim so every typed error flushes through `?`
/// or `.into()` at the boundary without per-callsite `.map_err`.
#[derive(Debug, Clone)]
pub enum TransformationNormalError {
    /// Shape/length/dimension/contract violations on inputs to a routine
    /// (e.g. response/covariate row mismatch, beta length mismatch,
    /// wrong number of blocks, malformed configuration parameters).
    InvalidInput { reason: String },
    /// A required covariate design or weight configuration cannot support
    /// the routine — empty design, zero total weight, residual variance
    /// not representable, warm-start coefficients all non-finite.
    DesignDegenerate { reason: String },
    /// A numeric intermediate (response transform, derivative,
    /// log-likelihood, weight, offset, gradient component, calibration
    /// quantity) came out non-finite or non-positive where positive
    /// finite is required.
    NonFinite { reason: String },
    /// The fitted monotone transform's derivative dropped to or below
    /// zero, or the response endpoint ordering required by the latent
    /// score (lower < h < upper) was not satisfied at evaluation time.
    MonotonicityViolated { reason: String },
    /// A numerical step that maps through the standard-normal CDF
    /// (endpoint mass, log-difference, PIT probability, derivative
    /// ratio) underflowed or became non-representable at the requested
    /// arguments.
    NumericalFailure { reason: String },
}


impl std::fmt::Display for TransformationNormalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformationNormalError::InvalidInput { reason }
            | TransformationNormalError::DesignDegenerate { reason }
            | TransformationNormalError::NonFinite { reason }
            | TransformationNormalError::MonotonicityViolated { reason }
            | TransformationNormalError::NumericalFailure { reason } => f.write_str(reason),
        }
    }
}


impl std::error::Error for TransformationNormalError {}


impl From<TransformationNormalError> for String {
    /// Shim for the many `Result<_, String>` signatures the module exposes
    /// (notably the `CustomFamily` and joint-Hessian / psi-workspace
    /// trait surfaces). Lets a typed `Err(TransformationNormalError::…)`
    /// flow through `?` or `.into()` without per-callsite stringification.
    fn from(err: TransformationNormalError) -> String {
        err.to_string()
    }
}


impl From<crate::util::block_count::BlockCountMismatch> for TransformationNormalError {
    fn from(err: crate::util::block_count::BlockCountMismatch) -> TransformationNormalError {
        TransformationNormalError::InvalidInput {
            reason: err.message(),
        }
    }
}
