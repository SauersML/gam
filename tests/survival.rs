//! Grouped integration-test crate root for survival tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "survival/competing_risks_survival.rs"]
mod competing_risks_survival;
#[path = "survival/frailty_scale_audit_plumbing.rs"]
mod frailty_scale_audit_plumbing;
#[path = "survival/integration_large_scale_survival_marginal_slope.rs"]
mod integration_large_scale_survival_marginal_slope;
#[path = "survival/pathological_tied_survival_times_no_divzero.rs"]
mod pathological_tied_survival_times_no_divzero;
#[path = "survival/surv_two_arg_shorthand_issue_156.rs"]
mod surv_two_arg_shorthand_issue_156;
#[path = "survival/survival_bug_hunt_regressions.rs"]
mod survival_bug_hunt_regressions;
#[path = "survival/survival_fit_from_parts_rejects_mismatched_lambda_lengths.rs"]
mod survival_fit_from_parts_rejects_mismatched_lambda_lengths;
#[path = "survival/survival_location_scale_constraint_projection_regression.rs"]
mod survival_location_scale_constraint_projection_regression;
#[path = "survival/survival_location_scale_small_repro.rs"]
mod survival_location_scale_small_repro;
#[path = "survival/survival_loglogistic_aft_recovers_covariate_1110.rs"]
mod survival_loglogistic_aft_recovers_covariate_1110;
#[path = "survival/survival_marginal_slope_1040_convergence.rs"]
mod survival_marginal_slope_1040_convergence;
#[path = "survival/survival_marginal_slope_jacobian_hyperbolic_correction.rs"]
mod survival_marginal_slope_jacobian_hyperbolic_correction;
#[path = "survival/survival_marginal_slope_kappa_probe_penalty_width_788.rs"]
mod survival_marginal_slope_kappa_probe_penalty_width_788;
#[path = "survival/survival_marginal_slope_large_scale_repro.rs"]
mod survival_marginal_slope_large_scale_repro;
#[path = "survival/survival_marginal_slope_neyman_orthogonal_reference.rs"]
mod survival_marginal_slope_neyman_orthogonal_reference;
#[path = "survival/survival_marginal_slope_outer_gradient_fd_1040.rs"]
mod survival_marginal_slope_outer_gradient_fd_1040;
#[path = "survival/survival_marginal_slope_stall.rs"]
mod survival_marginal_slope_stall;
#[path = "survival/survival_marginal_slope_vm_exact_integration.rs"]
mod survival_marginal_slope_vm_exact_integration;
#[path = "survival/survival_multi_z_covariance_autoderiv_hard.rs"]
mod survival_multi_z_covariance_autoderiv_hard;
#[path = "survival/survival_multi_z_fit_hard.rs"]
mod survival_multi_z_fit_hard;
#[path = "survival/survival_multi_z_marginal_slope.rs"]
mod survival_multi_z_marginal_slope;
#[path = "survival/survival_multi_z_margpreserve_hard.rs"]
mod survival_multi_z_margpreserve_hard;
#[path = "survival/survival_multi_z_neglog_hard.rs"]
mod survival_multi_z_neglog_hard;
#[path = "survival/survival_multi_z_reduction_hard.rs"]
mod survival_multi_z_reduction_hard;
#[path = "survival/survival_optimizer_api.rs"]
mod survival_optimizer_api;
#[path = "survival/survival_regression.rs"]
mod survival_regression;
#[path = "survival/survival_right_censored_shorthand_posterior_sample.rs"]
mod survival_right_censored_shorthand_posterior_sample;
