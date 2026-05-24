//! Phase-specific GPU backend placeholder.
//!
//! The public marker keeps the HAL surface explicit while the default build and
//! unsupported CUDA configurations use CPU fallback through dispatch policy.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendPhaseMarker;
