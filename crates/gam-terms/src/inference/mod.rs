//! Basis/structure-level inference instruments that descended into `gam-terms`
//! during the #1521 crate carve.
//!
//! These modules depend only on `gam-terms` and crates below it (gam-linalg,
//! gam-spec, gam-math, gam-problem). They were hoisted out of the monolith's
//! `gam::inference::*` namespace; the root crate keeps the old paths valid via
//! re-exports.

pub mod higher_order;
pub mod lawley;
pub mod smooth_test;
pub mod structure_evidence;
