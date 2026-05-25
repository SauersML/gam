use crate::terms::analytic_penalties::MonotonicityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for MonotonicityPenalty {
    const KIND_TAG: &'static str = "monotonicity";
    const PYTHON_WRAPPER: &'static str = "MonotonicityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
