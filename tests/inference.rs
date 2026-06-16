//! Grouped integration-test crate root for inference tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "inference/alo_tests.rs"]
mod alo_tests;
#[path = "inference/bms_audit_nonzero_logslope_baseline_370.rs"]
mod bms_audit_nonzero_logslope_baseline_370;
#[path = "inference/bms_probit_confound_orthogonalization_cure.rs"]
mod bms_probit_confound_orthogonalization_cure;
#[path = "inference/coefficient_groups.rs"]
mod coefficient_groups;
#[path = "inference/coefficient_groups_hierarchical_hard.rs"]
mod coefficient_groups_hierarchical_hard;
#[path = "inference/coefficient_label_by_block_name_rejects_duplicate_block_names.rs"]
mod coefficient_label_by_block_name_rejects_duplicate_block_names;
#[path = "inference/conformal_coverage_quality.rs"]
mod conformal_coverage_quality;
#[path = "inference/functionals_average_derivative.rs"]
mod functionals_average_derivative;
#[path = "inference/marginal_slope_neyman_orthogonal_reference.rs"]
mod marginal_slope_neyman_orthogonal_reference;
#[path = "inference/margslope_flex_large_scale_repro.rs"]
mod margslope_flex_large_scale_repro;
#[path = "inference/margslope_inner_pirls_scaling.rs"]
mod margslope_inner_pirls_scaling;
#[path = "inference/margslope_smallcondition_smoke.rs"]
mod margslope_smallcondition_smoke;
#[path = "inference/multi_z_marginal_slope.rs"]
mod multi_z_marginal_slope;
#[path = "inference/nuts_leapfrog_identity_posterior_recovery.rs"]
mod nuts_leapfrog_identity_posterior_recovery;
#[path = "inference/random_effect_recovers_group_means.rs"]
mod random_effect_recovers_group_means;
#[path = "inference/riesz_functionals_contract.rs"]
mod riesz_functionals_contract;
#[path = "inference/subsample_outer_integration.rs"]
mod subsample_outer_integration;
#[path = "inference/uncertainty_integration.rs"]
mod uncertainty_integration;
