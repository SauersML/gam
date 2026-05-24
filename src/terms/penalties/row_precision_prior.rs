use crate::terms::analytic_penalties::RowPrecisionPriorPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for RowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "AuxConditionalPriorPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}
