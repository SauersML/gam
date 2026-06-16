//! Identifiability: one coherent home for every identifiability concern in the
//! crate (issue #1140).
//!
//! Previously the identifiability logic was smeared across four directories
//! under four different vocabulary words. It now lives here, under one
//! consistent vocabulary:
//!
//! - [`kernel`] — the low-level rank / null-space kernels.
//! - [`precondition`] — cheap pre-fit precondition checks.
//! - [`audit`] — the joint cross-block identifiability audit.
//! - [`canonical`] — canonicalization of specs for identifiability.
//! - [`families`] — the family-agnostic block compiler, its GPU paths, and the
//!   per-family row-Hessian implementations.
//! - [`marginal_slope`] — survival marginal-slope identifiability.

pub mod audit;
pub mod canonical;
pub mod families;
pub mod kernel;
pub mod marginal_slope;
pub mod precondition;
