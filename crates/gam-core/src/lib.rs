//! `gam-core` — foundational shared types for the gam workspace.
//!
//! This is the leaf crate of the issue #1521 workspace split. The split makes
//! the high-churn code (families, terms, solver) a *downstream* compile unit so
//! a change to a family or term no longer invalidates the whole 770k-LOC crate:
//! Rust only recompiles a crate when it or an upstream dependency changes, so
//! pushing the foundational, near-static types into this leaf keeps everything
//! above it cached across the common edit.
//!
//! Planned contents (moved in the serializing window, held until post-#932):
//! the contents of `gam::types`, `gam::resource`, `gam::model_types`, and the
//! shared error enums that today force `linalg`/`gpu`/`solver`/`terms` to all
//! reference each other through `crate::`. Moving them here breaks the
//! `families ↔ solver ↔ terms` reference cycles by giving every layer a common
//! upstream to name instead of naming each other.
//!
//! Until that window, this crate is intentionally minimal so the workspace
//! skeleton lands additively without colliding with in-flight edits to the
//! monolith.

/// Marker for the gam-core scaffold (issue #1521). Replaced by the real shared
/// types when the maintainer declares the migration window. Present so the
/// crate exposes a public item and compiles as a non-empty library under the
/// workspace's `warnings = "deny"` lint.
pub const WORKSPACE_SPLIT_PHASE: &str = "1-skeleton";
