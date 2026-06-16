//! Family identifiability machinery: the family-agnostic block compiler, its
//! GPU acceleration paths, and the per-family row-Hessian implementations
//! (Bernoulli, survival marginal-slope).
//!
//! See issue #1140 — this consolidates the previously scattered
//! `identifiability_*` modules under one coherent home.

pub mod compiler;
pub(crate) mod gpu;
#[cfg(target_os = "linux")]
pub(crate) mod gpu_kernel;
pub mod bernoulli;
