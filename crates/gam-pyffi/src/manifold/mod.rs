//! Manifold descriptor `#[pyclass]`es.
//!
//! The Python-visible manifold config classes (`EuclideanManifold`,
//! `SphereManifold`, …). Re-exported at the crate root so the `#[pymodule]`
//! registration resolves `manifold_pyclasses::…` unchanged.
//!
//! The geometry / manifold-and-posterior FFI *entrypoint* fragments
//! (`geometry_ffi.rs`, `manifold_and_posterior_ffi.rs`) also live in this
//! directory for navigability, but they are `include!`d textually at the crate
//! root (they are not submodules) — `geometry_ffi.rs` carries the
//! `#[pymodule] _rust` registration.

pub(crate) mod manifold_pyclasses;
pub(crate) mod manifold_sae_payload;
