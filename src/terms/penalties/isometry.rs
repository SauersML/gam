use crate::terms::analytic_penalties::IsometryPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for IsometryPenalty {
    const KIND_TAG: &'static str = "isometry";
    const PYTHON_WRAPPER: &'static str = "IsometryPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
