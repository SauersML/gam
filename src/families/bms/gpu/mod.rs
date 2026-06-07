//! Bernoulli marginal-slope GPU kernels.
//!
//! These modules contain BMS-specific row math. Generic CUDA runtime, memory,
//! policy, and diagnostic plumbing stays under [`crate::gpu`].

pub mod flex;
pub mod row;
