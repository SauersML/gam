//! Survival family, consolidated into one folder.
//!
//! Each submodule owns one concern; the family's public surface is preserved
//! at this module root via glob re-exports (each item keeps its own
//! `pub`/`pub(crate)` visibility), so `crate::families::survival::Foo` resolves
//! exactly as before the consolidation.
//!
//! - [`base`]              — the core survival family (formerly `survival.rs`).

pub mod base;

pub use base::*;
