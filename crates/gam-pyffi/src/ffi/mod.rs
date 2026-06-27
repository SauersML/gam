//! FFI foundation: the shared engine/pyo3 import prelude, the typed
//! engine-error → Python-exception boundary, and the self-contained
//! Python-literal / non-finite-safe-JSON transport helpers.
//!
//! These are re-exported at the crate root (see `lib.rs`) under their historical
//! flat names so the `include!`-fragment entrypoint code keeps reaching them
//! through `crate::py_value_error`, `crate::GamError`, `crate::PyObject`,
//! `crate::finite_safe_json::…`, etc. The Rust module *paths* gained the `ffi::`
//! prefix; the Python surface is unchanged.

pub(crate) mod ffi_prelude;

pub(crate) mod ffi_errors;

pub(crate) mod python_literal;

pub(crate) mod finite_safe_json;
