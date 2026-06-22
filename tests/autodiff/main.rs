//! Grouped integration-test crate root for autodiff tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[macro_use]
mod common;

mod analytic_ard_penalty_logdet_derivatives_match_finite_difference;
mod analytic_penalty_nested_prefix_dispatch_missing;
mod autodiff_binomial_location_scale_exact;
mod autodiff_crosscheck;
mod autodiff_crosscheck_extended;
mod autodiff_custom_family_joint_laml;
mod autodiff_custom_family_pseudo_laplace;
mod autodiff_gaussian_location_scale_exact;
mod autodiff_sas_hypergradient_localization;
mod block_orthogonality_hvp_correctness;
mod channel_hessian_beta_dependent;
mod channel_hessian_matches_fd;
mod contract_gradient_gates;
mod cubic_cell_kernel_transformed_link_jet_fd_mismatch;
mod derivative_consistency_fd;
mod exact_oracle_tests;
mod external_gradient_trend;
mod gradient_decompose_test;
mod gradient_isolation;
mod ground_truth_gradient;
mod lbfgs_secant_pair_curvature_positive;
mod no_production_finite_differences;
mod objective_gradient_consistency_universal;
mod survival_laml_erfc_oracle_931;
mod zero_gradient_proposes_zero_step;
