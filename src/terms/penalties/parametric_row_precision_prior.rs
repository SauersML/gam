use crate::terms::analytic_penalties::ParametricRowPrecisionPriorPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for ParametricRowPrecisionPriorPenalty {
    const KIND_TAG: &'static str = "parametric_row_precision_prior";
    const PYTHON_WRAPPER: &'static str = "ParametricAuxConditionalPriorPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

