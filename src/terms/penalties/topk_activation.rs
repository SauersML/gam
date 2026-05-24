use crate::terms::analytic_penalties::TopKActivationPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for TopKActivationPenalty {
    const KIND_TAG: &'static str = "topk_activation";
    const PYTHON_WRAPPER: &'static str = "TopKActivationPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

