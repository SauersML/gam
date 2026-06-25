//! Domain-specific GPU compute kernels.
//!
//! These modules implement the heavy numerical device kernels (arrow-Schur
//! solves, Polya-Gamma sampling, cubic B-spline / cubic-cell moments, sigma
//! cubature, REML trace estimation, per-row Hessian ops, device-resident SAE,
//! and the PIRLS row kernel). They are layered on top of the hardware
//! abstraction in the parent [`crate::gpu`] module (runtime, memory, driver,
//! linalg, error, …) and are kept separate from that infrastructure boundary.

pub mod arrow_schur;
pub mod arrow_schur_nvrtc;
pub mod cubic_bspline_moments;
pub mod cubic_cell;
pub mod pirls_row;
pub mod polya_gamma;
pub mod reml_trace;
pub mod row_hessian_ops;
pub mod sae_resident;
pub mod sae_rowjet;
pub(crate) mod sigma_cubature;
pub mod survival_rowjet;
