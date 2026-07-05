//! SAE-level inference instruments that descended into `gam-sae` during the
//! #1521 crate carve (top of the DAG).
//!
//! These modules consume the SAE manifold term (`crate::manifold`,
//! `crate::chart_canonicalization`) plus solver/terms/problem items reached as
//! `gam_solve::*`, `gam_terms::*`, and `gam_problem::*`. They were hoisted out
//! of the monolith's `gam::inference::*` namespace; the root crate keeps the
//! old `gam::inference::{atom_lens, steering, ...}` paths valid via re-exports.

pub mod atom_geometry;
pub mod atom_lens;
pub mod checkpoint_dynamics;
pub mod harvest;
pub mod intervention_shard;
pub mod layer_transport;
pub mod probe_runner;
pub mod riesz;
pub mod steering;
pub mod transport_class;
