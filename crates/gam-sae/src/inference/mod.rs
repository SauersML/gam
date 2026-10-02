//! SAE-level inference instruments that descended into `gam-sae` during the
//! #1521 crate carve (top of the DAG).
//!
//! These modules consume the SAE manifold term (`crate::manifold`,
//! `crate::chart_canonicalization`) plus solver/terms/problem items reached as
//! `gam_solve::*`, `gam_terms::*`, and `gam_problem::*`. They were hoisted out
//! of the monolith's `gam::inference::*` namespace and are reached as
//! `gam::terms::sae::inference::*`: gam-inference does not depend on gam-sae,
//! so an SAE edit never rebuilds gam-inference, gam-predict, or gam-models' tests.

pub mod atlas_holonomy;
pub mod atlas_nerve;
pub mod atom_lens;
pub mod atom_shape_race;
pub mod checkpoint_dynamics;
pub mod contracts;
pub mod cross_model_transport;
pub use gam_response::intervention_shard;
pub mod layer_transport;
pub mod sparse_audit;
pub mod steering;
pub mod transport_class;

#[cfg(test)]
mod tests_dose_units_2249;

#[cfg(test)]
mod tests_dose_calibration_2249;

// #2263 item 3 — requested-vs-realized chart DISPLACEMENT. The two modules
// above pin requested-vs-realized nats; realized position had no reader at all.
#[cfg(test)]
mod tests_displacement_2263;

// #2234 / #2263 item 3 — requested-vs-realized INTRINSIC (arc-length)
// displacement. The module above pins the chart-coordinate round trip in the
// chart's own parameter; this one asks whether that parameter is the intrinsic
// unit the steering surface documents.
#[cfg(test)]
mod tests_unit_speed_steering_2234;
