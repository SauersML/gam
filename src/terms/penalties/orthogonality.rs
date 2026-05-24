use crate::terms::analytic_penalties::OrthogonalityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for OrthogonalityPenalty {
    const KIND_TAG: &'static str = "orthogonality";
    const PYTHON_WRAPPER: &'static str = "OrthogonalityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
