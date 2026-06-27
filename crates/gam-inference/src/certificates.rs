//! The unified certificate contract (task #16).
//!
//! The pure-data certificate contract types (the verdict ladder, the claim /
//! evidence value model, the [`Certificate`] trait, and the
//! [`CertificateLedger`]) were contracted down into the neutral `gam-problem`
//! crate (task #1521) so that lower-tier crates such as `gam-solve` can name
//! them without an upward dependency on this monolith. They are re-exported
//! here unchanged so every existing `crate::certificates::*` path
//! continues to resolve to the same types.
pub use gam_problem::topology_certificates::*;
