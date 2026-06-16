//! Typed errors emitted by smooth-term helpers.
//!
//! Carved out of the `smooth` module along its first real seam: this is the
//! error vocabulary every smooth-term builder shares, with no dependency on the
//! basis/penalty/optimizer machinery that surrounds it. `Display` reproduces the
//! exact pre-refactor `format!(...)` text byte-for-byte, so string-matching
//! callers (tests, log assertions) keep working unchanged. Public boundaries
//! that interface with `term_builder.rs` / `construction` continue to return
//! `Result<_, String>`; typed `Err(...)` values flow through
//! `From<SmoothError> for String` at those boundaries via `.into()`.

use crate::families::block_layout::block_count::BlockCountMismatch;

/// Typed errors emitted by smooth-term helpers in the `smooth` module.
#[derive(Clone, Debug)]
pub enum SmoothError {
    /// Spec-level configuration error: unfrozen knots/centers, invalid
    /// numeric bounds on coefficient priors, length-scale optimizer options
    /// that violate `min > 0 < max` / `min < max` invariants, etc.
    InvalidConfig { reason: String },
    /// Shape / length disagreement between design columns, penalty blocks,
    /// coefficient ranges, directional-derivative vectors, theta length, and
    /// other bookkeeping invariants that are checked at runtime.
    DimensionMismatch { reason: String },
    /// Out-of-range index into adaptive hyperparameter components, psi
    /// derivative blocks, or non-zero block indices for single-block
    /// custom-family impls that only support `block_idx == 0`.
    InvalidIndex { reason: String },
}

crate::impl_reason_error_boilerplate! {
    SmoothError {
        InvalidConfig,
        DimensionMismatch,
        InvalidIndex,
    }
}

impl From<BlockCountMismatch> for SmoothError {
    fn from(err: BlockCountMismatch) -> SmoothError {
        SmoothError::invalid_index(err.message())
    }
}

impl SmoothError {
    #[inline]
    pub(super) fn invalid_config(reason: impl Into<String>) -> Self {
        SmoothError::InvalidConfig {
            reason: reason.into(),
        }
    }
    #[inline]
    pub(super) fn dimension_mismatch(reason: impl Into<String>) -> Self {
        SmoothError::DimensionMismatch {
            reason: reason.into(),
        }
    }
    #[inline]
    pub(super) fn invalid_index(reason: impl Into<String>) -> Self {
        SmoothError::InvalidIndex {
            reason: reason.into(),
        }
    }
}
