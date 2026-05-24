use crate::terms::analytic_penalties::ARDPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for ARDPenalty {
    const KIND_TAG: &'static str = "ard";
    const PYTHON_WRAPPER: &'static str = "ARDPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

