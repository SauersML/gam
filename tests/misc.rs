//! Grouped integration-test crate root for misc tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "misc/additive_models_new_smooths.rs"]
mod additive_models_new_smooths;
#[path = "misc/boundary_formula_integration.rs"]
mod boundary_formula_integration;
#[path = "misc/ctn_crossfit_fold_knots_859.rs"]
mod ctn_crossfit_fold_knots_859;
#[path = "misc/formula_dsl_factor_smooth_aliases.rs"]
mod formula_dsl_factor_smooth_aliases;
#[path = "misc/formula_dsl_new_smooth_aliases.rs"]
mod formula_dsl_new_smooth_aliases;
#[path = "misc/full_conformal_predict_route_quality.rs"]
mod full_conformal_predict_route_quality;
#[path = "misc/residual_cascade_auto_route_quality.rs"]
mod residual_cascade_auto_route_quality;
#[path = "misc/residual_cascade_certification.rs"]
mod residual_cascade_certification;
#[path = "misc/residual_cascade_workflow_detection.rs"]
mod residual_cascade_workflow_detection;
#[path = "misc/sas_mixture_system.rs"]
mod sas_mixture_system;
#[path = "misc/synthbug_repro.rs"]
mod synthbug_repro;
