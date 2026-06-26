//! PyO3 boundary for the `gam` Rust engine.
//!
//! The crate is organized as a flat set of concern modules. Two foundation
//! modules are re-exported at the crate root so every concern module can reach
//! them through their bare names:
//!
//! * [`ffi_prelude`] — the single curated set of `gam`-engine / pyo3 / numpy /
//!   ndarray re-exports the FFI boundary leans on.
//! * [`ffi_errors`] — the canonical gamfit exception hierarchy and the typed
//!   engine-error → Python-exception adaptors (issue #343).
//!
//! Analytic penalty JSON dispatch is shared through
//! `build_analytic_penalty_registry_from_json`; the accepted descriptor kinds
//! include `"nested_prefix"` for `NestedPrefixPenalty`.

mod ffi_prelude;

mod ffi_errors;

// Re-export the foundation modules at the crate root. The concern modules
// (and the `#[pyfunction]`s in the included fragments) reach the exception
// classes / converters and the `PyObject` alias through their bare crate-root
// names (`crate::py_value_error`, `crate::GamError`, `crate::PyObject`, …), so
// the boundary error contract and the shared engine alias each live in exactly
// one place.
pub(crate) use ffi_errors::*;
pub(crate) use ffi_prelude::*;

mod benchmark_scores;

mod competing_risks_decode;

mod inference_instruments;

mod manifold_pyclasses;

mod python_literal;

mod sklearn_metadata;

mod summary_render;

mod survival_surface_io;

mod finite_safe_json;

include!("model_ffi.rs");
include!("latent_basis_and_sae_ffi.rs");
include!("reml_latent_fit_ffi.rs");
include!("geometry_ffi.rs");
include!("manifold_and_posterior_ffi.rs");
