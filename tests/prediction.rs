//! Grouped integration-test crate root for prediction tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "prediction/apply_family_inverse_link_variants_match_documented_mu.rs"]
mod apply_family_inverse_link_variants_match_documented_mu;
#[path = "prediction/inverse_link_beta_logistic_jet_matches_finite_difference.rs"]
mod inverse_link_beta_logistic_jet_matches_finite_difference;
#[path = "prediction/inverse_link_latent_cloglog_jet_matches_finite_difference.rs"]
mod inverse_link_latent_cloglog_jet_matches_finite_difference;
#[path = "prediction/inverse_link_mixture_jet_matches_finite_difference.rs"]
mod inverse_link_mixture_jet_matches_finite_difference;
#[path = "prediction/inverse_link_sas_jet_matches_finite_difference.rs"]
mod inverse_link_sas_jet_matches_finite_difference;
#[path = "prediction/inverse_link_standard_jet_matches_finite_difference.rs"]
mod inverse_link_standard_jet_matches_finite_difference;
#[path = "prediction/latent_cloglog_inverse_link_cdf_monotone_bounds.rs"]
mod latent_cloglog_inverse_link_cdf_monotone_bounds;
#[path = "prediction/mixture_link_cauchit_loglog_state_rejection.rs"]
mod mixture_link_cauchit_loglog_state_rejection;
#[path = "prediction/predict_at_training_matches_fitted.rs"]
mod predict_at_training_matches_fitted;
#[path = "prediction/predict_at_training_matches_fitted_basis_sweep.rs"]
mod predict_at_training_matches_fitted_basis_sweep;
#[path = "prediction/predict_byfactor_recovers_truth_on_grid.rs"]
mod predict_byfactor_recovers_truth_on_grid;
#[path = "prediction/predict_dispersion_location_scale_not_binomial_1064.rs"]
mod predict_dispersion_location_scale_not_binomial_1064;
#[path = "prediction/predict_linear_interaction_recovers_truth_on_new_grid.rs"]
mod predict_linear_interaction_recovers_truth_on_new_grid;
#[path = "prediction/predict_linear_term_extrapolation_se_grows.rs"]
mod predict_linear_term_extrapolation_se_grows;
#[path = "prediction/predict_new_row_eta_uses_training_column_order.rs"]
mod predict_new_row_eta_uses_training_column_order;
#[path = "prediction/predict_on_cpu_only_host_does_not_panic_with_cudarc.rs"]
mod predict_on_cpu_only_host_does_not_panic_with_cudarc;
#[path = "prediction/predict_outside_train_range_bounded.rs"]
mod predict_outside_train_range_bounded;
#[path = "prediction/predict_parametric_survival_linear_covariate.rs"]
mod predict_parametric_survival_linear_covariate;
#[path = "prediction/predict_tensor_te_recovers_truth_on_grid.rs"]
mod predict_tensor_te_recovers_truth_on_grid;
#[path = "prediction/predict_uncertainty_interval_coverage.rs"]
mod predict_uncertainty_interval_coverage;
#[path = "prediction/sas_inverse_link_cdf_monotone_bounds.rs"]
mod sas_inverse_link_cdf_monotone_bounds;
