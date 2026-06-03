//! Manifest entry for between-block latent orthogonality.
//!
//! The implementation lives in [`BlockOrthogonalityPenalty`]. It is an
//! extension-coordinate penalty
//! `½·w·Σ_{g<h} ||T[:,g]^T T[:,h]||_F^2` that keeps SAE latent axis groups
//! separated while leaving within-block structure free.

use crate::terms::analytic_penalties::BlockOrthogonalityPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for BlockOrthogonalityPenalty {
    const KIND_TAG: &'static str = "block_orthogonality";
    const PYTHON_WRAPPER: &'static str = "BlockOrthogonalityPenalty";
    /// Between-block Gram curvature is dense; use the analytic HVP for full
    /// Newton curvature.
    const ROW_BLOCK_DIAGONAL: bool = false;
}
