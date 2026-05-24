use crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for SoftmaxAssignmentSparsityPenalty {
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";
    const PYTHON_WRAPPER: &'static str = "SoftmaxAssignmentSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

