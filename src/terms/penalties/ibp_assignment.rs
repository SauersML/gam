use crate::terms::analytic_penalties::IBPAssignmentPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for IBPAssignmentPenalty {
    const KIND_TAG: &'static str = "ibp_assignment";
    const PYTHON_WRAPPER: &'static str = "IBPAssignmentPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

