//! `EstimationError` moved to the lower-layer [`crate::model_types`] module
//! (issue #1135) so the `families` layer can name it without importing *up*
//! into `solver::estimate`. Re-exported here for source compatibility.

pub use crate::model_types::EstimationError;

//
// This uses the joint model architecture where the base predictor and
// flexible link are fitted together in one optimization with REML.
//
// The model is: η = g(Xβ) where g is a learned flexible link function.
// Domain-specific training orchestration is handled by caller adapters.
// The gam engine exposes matrix/family-based external-design APIs for supported
// GLM-style families: fit_gam / optimize_external_design.
