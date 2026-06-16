//! Shared block-count arity guard.
//!
//! Every family / term entry point that consumes a `&[ParameterBlockState]`
//! opens with the identical check: "this family expects exactly `N` blocks;
//! reject any other count". The structure is fixed — only the family name,
//! the expected arity, and the per-module error enum vary. This module holds
//! the single canonical implementation so the guard is not re-typed by hand
//! across modules.
//!
//! The canonical error message is owned by [`BlockCountMismatch::message`].
//! Each module routes its own error enum through `From<BlockCountMismatch>`,
//! so the per-module error identity (and its `Display`/variant) is preserved
//! while the arity-check logic and the message wording live in one place.

/// A block-count arity mismatch: a family that needs exactly `expected`
/// parameter blocks was handed `got` of them.
///
/// This is the neutral carrier produced by [`validate_block_count`]; each
/// module converts it into its own error type via `From<BlockCountMismatch>`.
pub struct BlockCountMismatch {
    /// Human-readable family / term name used as the message prefix.
    pub family: String,
    /// Number of parameter blocks the family requires.
    pub expected: usize,
    /// Number of parameter blocks that were actually supplied.
    pub got: usize,
}

impl BlockCountMismatch {
    /// The canonical arity-mismatch message, e.g.
    /// `"FooFamily expects 2 blocks, got 1"` (plural) or
    /// `"BarFamily expects 1 block, got 0"` (singular when `expected == 1`).
    pub fn message(&self) -> String {
        let unit = if self.expected == 1 {
            "block"
        } else {
            "blocks"
        };
        format!(
            "{} expects {} {unit}, got {}",
            self.family, self.expected, self.got
        )
    }
}

/// Reject any `got` block count that does not exactly equal `expected`.
///
/// On mismatch, builds a [`BlockCountMismatch`] and converts it into the
/// caller's error type `E` (chosen via the turbofish or inferred from the
/// surrounding `?`). On a match, returns `Ok(())`.
///
/// This is the single source of truth for the block-arity guard shared
/// across the family and term modules.
#[inline]
pub fn validate_block_count<E>(
    family: impl Into<String>,
    expected: usize,
    got: usize,
) -> Result<(), E>
where
    E: From<BlockCountMismatch>,
{
    if got != expected {
        return Err(BlockCountMismatch {
            family: family.into(),
            expected,
            got,
        }
        .into());
    }
    Ok(())
}
