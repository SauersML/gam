use crate::terms::analytic_penalties::MechanismSparsityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for MechanismSparsityPenalty {
    const KIND_TAG: &'static str = "mechanism_sparsity";
    const PYTHON_WRAPPER: &'static str = "MechanismSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
