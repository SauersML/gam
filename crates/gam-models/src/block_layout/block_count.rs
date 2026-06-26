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
///
/// The type itself has descended to `gam-problem` so lower tiers can route
/// their error enums through `From<BlockCountMismatch>` without depending on
/// `gam-models`; it is re-exported here unchanged.
pub use gam_problem::BlockCountMismatch;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_count_is_ok() {
        // Exact match returns Ok(()) and produces no error.
        let result: Result<(), String> = validate_block_count("FooFamily", 2, 2);
        assert!(result.is_ok());

        let zero: Result<(), String> = validate_block_count("EmptyFamily", 0, 0);
        assert!(zero.is_ok());
    }

    #[test]
    fn wrong_count_is_rejected_with_canonical_message() {
        // Plural form when expected != 1.
        let err: String = validate_block_count::<String>("FooFamily", 2, 1).unwrap_err();
        assert_eq!(err, "FooFamily expects 2 blocks, got 1");

        // Too many blocks is rejected just the same.
        let too_many: String = validate_block_count::<String>("FooFamily", 2, 3).unwrap_err();
        assert_eq!(too_many, "FooFamily expects 2 blocks, got 3");
    }

    #[test]
    fn singular_block_wording_when_expected_is_one() {
        // The message switches to the singular "block" when expected == 1.
        let err: String = validate_block_count::<String>("BarFamily", 1, 0).unwrap_err();
        assert_eq!(err, "BarFamily expects 1 block, got 0");
    }

    #[test]
    fn mismatch_carrier_message_matches_helper() {
        // The carrier's message is the single source of truth for the wording.
        let carrier = BlockCountMismatch {
            family: "BazFamily".to_string(),
            expected: 3,
            got: 5,
        };
        assert_eq!(carrier.message(), "BazFamily expects 3 blocks, got 5");
        // And the String conversion forwards to it.
        let converted: String = carrier.into();
        assert_eq!(converted, "BazFamily expects 3 blocks, got 5");
    }
}
