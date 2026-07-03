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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_uses_plural_blocks_when_expected_greater_than_one() {
        let m = BlockCountMismatch {
            family: "FooFamily".to_string(),
            expected: 2,
            got: 0,
        };
        let msg = m.message();
        assert!(msg.contains("blocks"), "message: {msg}");
        assert!(msg.contains("FooFamily"), "message: {msg}");
        assert!(msg.contains("2") && msg.contains("0"), "message: {msg}");
    }

    #[test]
    fn message_uses_singular_block_when_expected_is_one() {
        let m = BlockCountMismatch {
            family: "BarFamily".to_string(),
            expected: 1,
            got: 3,
        };
        let msg = m.message();
        assert!(
            msg.contains("block") && !msg.contains("blocks"),
            "message: {msg}"
        );
        assert!(msg.contains("1") && msg.contains("3"), "message: {msg}");
    }

    #[test]
    fn from_block_count_mismatch_for_string_matches_message() {
        let m = BlockCountMismatch {
            family: "Baz".to_string(),
            expected: 2,
            got: 1,
        };
        let expected_msg = m.message();
        let as_string = String::from(BlockCountMismatch {
            family: "Baz".to_string(),
            expected: 2,
            got: 1,
        });
        assert_eq!(as_string, expected_msg);
    }
}
