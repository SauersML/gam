use crate::terms::analytic_penalties::BlockOrthogonalityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for BlockOrthogonalityPenalty {
    const KIND_TAG: &'static str = "block_orthogonality";
    const PYTHON_WRAPPER: &'static str = "BlockOrthogonalityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

