//! Manifest entry for the SCAD/MCP nonconvex sparsity penalty.
//!
//! The implementation lives in [`ScadMcpPenalty`]. It is an elementwise
//! extension-coordinate sparsity prior that tapers shrinkage for large
//! coefficients so active SAE latent amplitudes are not biased by constant L1
//! pull. The exact curvature is diagonal but can be indefinite.

use crate::terms::analytic_penalties::ScadMcpPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for ScadMcpPenalty {
    const KIND_TAG: &'static str = "scad_mcp";
    const PYTHON_WRAPPER: &'static str = "ScadMcpPenalty";
    /// Coordinate-separable; the diagonal may be negative in the SCAD/MCP taper
    /// region because this is the exact nonconvex Hessian.
    const ROW_BLOCK_DIAGONAL: bool = true;
}
