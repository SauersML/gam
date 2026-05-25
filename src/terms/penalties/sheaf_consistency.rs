//! [`PenaltyManifest`] registration for the cellular-sheaf consistency
//! penalty. The implementation lives in [`crate::terms::sheaf`].

use crate::terms::sheaf::SheafConsistencyPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for SheafConsistencyPenalty {
    const KIND_TAG: &'static str = "sheaf_consistency";
    const PYTHON_WRAPPER: &'static str = "SheafConsistencyPenalty";
    // The Hessian `L = δᵀ δ` couples every pair of vertices joined by an
    // edge — block-diagonal only for an empty edge set. Default to false so
    // downstream consumers know they must use `hvp`, not row-blocked solvers.
    const ROW_BLOCK_DIAGONAL: bool = false;
}
