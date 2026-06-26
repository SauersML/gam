//! Neutral block-count arity mismatch carrier.
//!
//! Descended from `gam-models`' `block_layout::block_count` so that lower
//! tiers (e.g. `gam-terms`) can route their own error enums through
//! `From<BlockCountMismatch>` without depending on `gam-models`. The
//! `validate_block_count` helper continues to live in `gam-models` and
//! re-exports this type unchanged.

/// A block-count arity mismatch: a family that needs exactly `expected`
/// parameter blocks was handed `got` of them.
///
/// This is the neutral carrier produced by `validate_block_count`; each
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

impl From<BlockCountMismatch> for String {
    fn from(err: BlockCountMismatch) -> String {
        err.message()
    }
}
