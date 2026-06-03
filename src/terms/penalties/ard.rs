//! Manifest entry for per-axis ARD latent precision.
//!
//! The implementation lives in [`ARDPenalty`]. It is an extension-coordinate
//! quadratic prior with one REML-selected precision per latent axis; it selects
//! intrinsic dimension after another term has fixed the latent gauge.

use crate::terms::analytic_penalties::ARDPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for ARDPenalty {
    const KIND_TAG: &'static str = "ard";
    const PYTHON_WRAPPER: &'static str = "ARDPenalty";
    /// The target Hessian is diagonal in coordinates, even though the row-major
    /// layout makes each axis strided in memory.
    const ROW_BLOCK_DIAGONAL: bool = true;
}
