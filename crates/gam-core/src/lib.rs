//! `gam-core` — foundational shared types for the gam workspace.
//!
//! Leaf crate of the issue #1521 workspace split: the high-churn code
//! (families, terms, solver) becomes a downstream compile unit so a change to a
//! family or term no longer recompiles these near-static types. Window 1 owns
//! the two fully leaf-clean modules `types` and `resource`, which `gam`
//! re-exports (`pub use gam_core::{resource, types};`) so every existing
//! `crate::types::*` / `crate::resource::*` path resolves unchanged.
//!
//! Every item these modules expose to `gam` is `pub` (not `pub(crate)`): across
//! a crate boundary `pub(crate)` is private to `gam-core`, so the shared floor
//! `MIN_WEIGHT`, the diagnostic helper, and all referenced types are public.

pub mod resource;
pub mod types;
