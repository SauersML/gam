use crate::terms::analytic_penalties::NestedPrefixPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for NestedPrefixPenalty {
    const KIND_TAG: &'static str = "nested_prefix";
    const PYTHON_WRAPPER: &'static str = "NestedPrefixPenalty";
    /// The penalty is purely diagonal in the latent target (per-coordinate
    /// L¹ surrogate with per-axis cumulative shell weight), so it slots into
    /// the row-block-diagonal Newton path.
    const ROW_BLOCK_DIAGONAL: bool = true;
}
