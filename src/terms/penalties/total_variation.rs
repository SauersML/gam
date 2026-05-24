use crate::terms::analytic_penalties::TotalVariationPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for TotalVariationPenalty {
    const KIND_TAG: &'static str = "total_variation";
    const PYTHON_WRAPPER: &'static str = "TotalVariationPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
