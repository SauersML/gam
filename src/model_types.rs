//! # Shared model-estimation contract types
//!
//! Lower-layer types that both the `families` layer (which constructs penalty
//! and dispersion specifications and propagates estimation errors) and the
//! `solver` layer (which consumes them) need to name. Hosting them here breaks
//! the `families → solver::estimate` back-edge that #1135 tracks: families now
//! import these from `crate::model_types` instead of reaching *up* into
//! `crate::solver::estimate`.
//!
//! ## Layering
//! These types now live in the `gam-solve` crate
//! ([`gam_solve::model_types`]); this module re-exports them so existing
//! `crate::model_types::*` / `gam::model_types::*` paths keep resolving after
//! the #1521 descent. They depend only on lower or sibling layers (`linalg`,
//! `terms`, `gam-problem`' error types) — never *up* into the model crate.

pub use gam_solve::model_types::*;
