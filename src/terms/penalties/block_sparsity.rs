use crate::terms::analytic_penalties::BlockSparsityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for BlockSparsityPenalty {
    const KIND_TAG: &'static str = "block_sparsity";
    const PYTHON_WRAPPER: &'static str = "BlockSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

