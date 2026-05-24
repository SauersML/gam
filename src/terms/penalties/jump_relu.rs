use crate::terms::analytic_penalties::JumpReLUPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for JumpReLUPenalty {
    const KIND_TAG: &'static str = "jumprelu";
    const PYTHON_WRAPPER: &'static str = "JumpReLUPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

