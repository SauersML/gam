//! `gam-core` — foundational shared types for the gam workspace.
//!
//! This is the leaf crate of the issue #1521 workspace split. The split makes
//! the high-churn code (families, terms, solver) a *downstream* compile unit so
//! a change to a family or term no longer invalidates the whole 770k-LOC crate:
//! Rust only recompiles a crate when it or an upstream dependency changes, so
//! pushing the foundational, near-static types into this leaf keeps everything
//! above it cached across the common edit.
//!
//! Window 1 (this crate) owns the two fully leaf-clean modules — `types` and
//! `resource` — which between them are referenced by ~130 files in `gam` and
//! depend on nothing upstream once their two stray edges were inlined
//! (`LikelihoodSpec::supports_firth`'s Fisher-weight-jet classifier and a dead
//! `PeeledHull` re-export). The `gam` crate re-exports both modules
//! (`pub use gam_core::{types, resource};`) so every existing `crate::types::*`
//! / `crate::resource::*` path resolves unchanged — no tree-wide rewrite.
//!
//! Later windows move `model_types` (after its 17 upward edges to
//! terms/linalg/families/solver are broken) and then peel off `linalg`, `gpu`,
//! `solver`, `terms`, `families`, and `inference` as their own crates.

pub mod resource;
pub mod types;
