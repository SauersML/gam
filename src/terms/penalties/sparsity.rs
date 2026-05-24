use crate::terms::analytic_penalties::SparsityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for SparsityPenalty {
    const KIND_TAG: &'static str = "sparsity";
    const PYTHON_WRAPPER: &'static str = "SparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

