//! Manifest entry for the smoothed nuclear-norm analytic penalty.
//!
//! The implementation lives in [`NuclearNormPenalty`]. It is an
//! extension-coordinate spectral penalty
//! `w·Σ_i(sqrt(σ_i^2 + ε^2) - ε)` for decoder/latent embedding-rank selection
//! in SAE wiring. Its curvature is dense and should be queried through the
//! analytic HVP.

use crate::terms::analytic_penalties::NuclearNormPenalty;

use super::PenaltyManifest;

impl PenaltyManifest for NuclearNormPenalty {
    const KIND_TAG: &'static str = "nuclear_norm";
    const PYTHON_WRAPPER: &'static str = "NuclearNormPenalty";
    /// Spectral penalty with dense matrix-function curvature.
    const ROW_BLOCK_DIAGONAL: bool = false;
}
