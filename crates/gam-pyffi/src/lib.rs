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

// Concern modules are grouped into a submodule tree on disk (issue #1521
// navigability carve): `ffi/` (foundation prelude/errors + literal/JSON
// helpers), `inference/` (inference instruments + benchmark scoring),
// `io/` (summary render, competing-risks decode, survival-surface I/O),
// `sklearn/` (sklearn-compat metadata), and `manifold/` (manifold
// descriptor `#[pyclass]`es alongside the geometry/posterior entrypoint
// fragments). The five large engine-entrypoint files are `include!`d at the
// crate root (textually flat), so they share one namespace and the
// `#[pymodule] _rust` registration in `manifold/geometry_ffi.rs` lives at the
// crate root.
mod ffi;

mod inference;

mod io;

mod sklearn;

mod manifold;

// Re-export the foundation modules at the crate root. The concern modules
// (and the `#[pyfunction]`s in the included fragments) reach the exception
// classes / converters and the `PyObject` alias through their bare crate-root
// names (`crate::py_value_error`, `crate::GamError`, `crate::PyObject`, …), so
// the boundary error contract and the shared engine alias each live in exactly
// one place.
pub(crate) use ffi::ffi_errors::*;
pub(crate) use ffi::ffi_prelude::*;

// Re-export every concern module at the crate root under its historical flat
// name. The `include!`-fragment entrypoint code is textually inlined at the
// crate root and reaches these modules by bare name (`benchmark_scores::…`,
// `crate::finite_safe_json::…`, `inference_instruments::register`, …); the flat
// re-export keeps every such path resolving while the source files live in the
// grouped subdirectories above. The Rust module paths now also exist in their
// canonical grouped form (`crate::ffi::finite_safe_json`, …); the Python
// surface is unchanged.
pub(crate) use ffi::{ffi_errors, ffi_prelude, finite_safe_json, python_literal};
pub(crate) use inference::{benchmark_scores, inference_instruments};
pub(crate) use io::{competing_risks_decode, summary_render, survival_surface_io};
pub(crate) use manifold::manifold_pyclasses;
pub(crate) use sklearn::sklearn_metadata;

include!("model/model_ffi.rs");
include!("latent/latent_basis_and_sae_ffi.rs");
include!("latent/reml_latent_fit_ffi.rs");
include!("manifold/geometry_ffi.rs");
include!("manifold/manifold_and_posterior_ffi.rs");
