//! Grouped integration-test crate root for glm tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "glm/beta_generative_phi_drives_draw_variance.rs"]
mod beta_generative_phi_drives_draw_variance;
#[path = "glm/cross_cutting_family_scalar_passthrough.rs"]
mod cross_cutting_family_scalar_passthrough;
#[path = "glm/family_scalars_contract_uniform.rs"]
mod family_scalars_contract_uniform;
#[path = "glm/firth_behavior.rs"]
mod firth_behavior;
#[path = "glm/firth_general_pc_hyperprior_default.rs"]
mod firth_general_pc_hyperprior_default;
#[path = "glm/gamlss_smooth_noise_mean_recovery_365.rs"]
mod gamlss_smooth_noise_mean_recovery_365;
#[path = "glm/gamma_log_coefficient_se_coverage.rs"]
mod gamma_log_coefficient_se_coverage;
#[path = "glm/gamma_precision_hyperpriors.rs"]
mod gamma_precision_hyperpriors;
#[path = "glm/gaussian_closed_form_reml.rs"]
mod gaussian_closed_form_reml;
#[path = "glm/gaussian_fit_saves_covariance_for_uncertainty.rs"]
mod gaussian_fit_saves_covariance_for_uncertainty;
#[path = "glm/gaussian_fixed_dispersion_is_ignored_in_pirls_geometry_and_deviance.rs"]
mod gaussian_fixed_dispersion_is_ignored_in_pirls_geometry_and_deviance;
#[path = "glm/gaussian_sparse_xtwx_cache_correctness.rs"]
mod gaussian_sparse_xtwx_cache_correctness;
#[path = "glm/gaussian_xtwx_cache_correctness.rs"]
mod gaussian_xtwx_cache_correctness;
#[path = "glm/magic_family_auto_poisson_count.rs"]
mod magic_family_auto_poisson_count;
#[path = "glm/multinomial_reml_deviance_reuse_issue_348.rs"]
mod multinomial_reml_deviance_reuse_issue_348;
#[path = "glm/probit_integration.rs"]
mod probit_integration;
