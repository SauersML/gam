use crate::terms::analytic_penalties::NuclearNormPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for NuclearNormPenalty {
    const KIND_TAG: &'static str = "nuclear_norm";
    const PYTHON_WRAPPER: &'static str = "NuclearNormPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
