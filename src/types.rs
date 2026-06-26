//! Shared REML/LAML contract types.
//!
//! These types are single-sourced in the `gam-problem` crate. This module
//! re-exports them so existing `crate::types::*` paths keep resolving without
//! maintaining a second, drift-prone copy of the definitions.
pub use gam_problem::types::*;
