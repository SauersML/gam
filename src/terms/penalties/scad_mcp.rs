use crate::terms::analytic_penalties::ScadMcpPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for ScadMcpPenalty {
    const KIND_TAG: &'static str = "scad_mcp";
    const PYTHON_WRAPPER: &'static str = "ScadMcpPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}

