//! Grouped integration-test crate root for autodiff tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[macro_use]
#[path = "common/mod.rs"]
mod common;

#[path = "autodiff/analytic_ard_penalty_logdet_derivatives_match_finite_difference.rs"]
mod analytic_ard_penalty_logdet_derivatives_match_finite_difference;
#[path = "autodiff/analytic_penalty_nested_prefix_dispatch_missing.rs"]
mod analytic_penalty_nested_prefix_dispatch_missing;
#[path = "autodiff/autodiff_binomial_location_scale_exact.rs"]
mod autodiff_binomial_location_scale_exact;
#[path = "autodiff/autodiff_crosscheck.rs"]
mod autodiff_crosscheck;
#[path = "autodiff/autodiff_crosscheck_extended.rs"]
mod autodiff_crosscheck_extended;
#[path = "autodiff/autodiff_custom_family_joint_laml.rs"]
mod autodiff_custom_family_joint_laml;
#[path = "autodiff/autodiff_custom_family_pseudo_laplace.rs"]
mod autodiff_custom_family_pseudo_laplace;
#[path = "autodiff/autodiff_gaussian_location_scale_exact.rs"]
mod autodiff_gaussian_location_scale_exact;
#[path = "autodiff/autodiff_sas_hypergradient_localization.rs"]
mod autodiff_sas_hypergradient_localization;
#[path = "autodiff/block_orthogonality_hvp_correctness.rs"]
mod block_orthogonality_hvp_correctness;
#[path = "autodiff/channel_hessian_beta_dependent.rs"]
mod channel_hessian_beta_dependent;
#[path = "autodiff/channel_hessian_matches_fd.rs"]
mod channel_hessian_matches_fd;
#[path = "autodiff/contract_gradient_gates.rs"]
mod contract_gradient_gates;
#[path = "autodiff/diag_1255_binomial_probit_outer.rs"]
mod diag_1255_binomial_probit_outer;
#[path = "autodiff/cubic_cell_kernel_transformed_link_jet_fd_mismatch.rs"]
mod cubic_cell_kernel_transformed_link_jet_fd_mismatch;
#[path = "autodiff/derivative_consistency_fd.rs"]
mod derivative_consistency_fd;
#[path = "autodiff/exact_oracle_tests.rs"]
mod exact_oracle_tests;
#[path = "autodiff/external_gradient_trend.rs"]
mod external_gradient_trend;
#[path = "autodiff/gradient_decompose_test.rs"]
mod gradient_decompose_test;
#[path = "autodiff/gradient_isolation.rs"]
mod gradient_isolation;
#[path = "autodiff/ground_truth_gradient.rs"]
mod ground_truth_gradient;
#[path = "autodiff/lbfgs_secant_pair_curvature_positive.rs"]
mod lbfgs_secant_pair_curvature_positive;
#[path = "autodiff/no_production_finite_differences.rs"]
mod no_production_finite_differences;
#[path = "autodiff/objective_gradient_consistency_universal.rs"]
mod objective_gradient_consistency_universal;
#[path = "autodiff/survival_laml_erfc_oracle_931.rs"]
mod survival_laml_erfc_oracle_931;
#[path = "autodiff/zero_gradient_proposes_zero_step.rs"]
mod zero_gradient_proposes_zero_step;
