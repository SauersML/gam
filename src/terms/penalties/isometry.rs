//! Manifest entry for the latent-coordinate isometry gauge penalty.
//!
//! The implementation lives in [`IsometryPenalty`]. It is an extension-coordinate
//! penalty on the decoder pullback metric,
//! `½·μ·Σ_n ||J_n^T W_n J_n - g_ref,n||_F^2`, used to pin SAE latent chart
//! units before axis/rank selection penalties are interpreted.

use crate::terms::analytic_penalties::IsometryPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for IsometryPenalty {
    const KIND_TAG: &'static str = "isometry";
    const PYTHON_WRAPPER: &'static str = "IsometryPenalty";
    /// Metric residual curvature couples latent coordinates through decoder
    /// jets; use HVP/Gauss-Newton majorizer paths.
    const ROW_BLOCK_DIAGONAL: bool = false;
}
