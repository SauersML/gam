//! Grouped integration-test crate root for glm tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod beta_generative_phi_drives_draw_variance;
mod cross_cutting_family_scalar_passthrough;
mod family_scalars_contract_uniform;
mod firth_behavior;
mod firth_general_pc_hyperprior_default;
mod gamlss_smooth_noise_mean_recovery_365;
mod gamma_log_coefficient_se_coverage;
mod gamma_precision_hyperpriors;
mod gaussian_closed_form_reml;
mod gaussian_fit_saves_covariance_for_uncertainty;
mod gaussian_fixed_dispersion_is_ignored_in_pirls_geometry_and_deviance;
mod gaussian_sparse_xtwx_cache_correctness;
mod gaussian_xtwx_cache_correctness;
mod magic_family_auto_poisson_count;
mod multinomial_reml_deviance_reuse_issue_348;
mod probit_integration;
