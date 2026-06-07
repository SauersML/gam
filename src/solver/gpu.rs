//! Solver GPU kernels and dispatch wiring.
//!
//! Process-wide GPU policy and direct device selection live in [`crate::gpu`].

pub mod arrow_schur_gpu;
pub mod pirls_dispatch;
pub mod pirls_dispatch_wire;
pub mod pirls_gpu;
pub mod reml_gpu;
pub mod reml_outer;
