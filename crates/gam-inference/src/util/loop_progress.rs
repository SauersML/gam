//! Low-overhead progress ticker for long parallel loops.
//!
//! This util is single-sourced in the `gam-runtime` crate (the runtime layer
//! shared by the sibling crates that consume it). This module re-exports it so
//! existing `crate::util::loop_progress::*` paths keep resolving without a
//! second, drift-prone copy of the definitions.
pub use gam_runtime::loop_progress::*;
