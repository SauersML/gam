//! Manifest entry for the SAE decoder column-space incoherence penalty.
//!
//! The implementation lives in [`DecoderIncoherencePenalty`]. It is a β-tier,
//! co-activation-masked cross-atom penalty,
//! `½·w·Σ_{j<k} W[j,k]·||B_j^T B_k||_F^2`, used as the SAE separability lever
//! for decoder atoms that co-fire on the same observations.

use crate::terms::analytic_penalties::DecoderIncoherencePenalty;

use super::PenaltyManifest;

impl PenaltyManifest for DecoderIncoherencePenalty {
    const KIND_TAG: &'static str = "decoder_incoherence";
    const PYTHON_WRAPPER: &'static str = "DecoderIncoherencePenalty";
    /// Dense across atom decoder blocks; uses a Gauss-Newton HVP rather than a
    /// row-block diagonal curvature shortcut.
    const ROW_BLOCK_DIAGONAL: bool = false;
}
