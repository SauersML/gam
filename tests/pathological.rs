//! Grouped integration-test crate root for pathological tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "pathological/bimodal_two_bumps_recovery.rs"]
mod bimodal_two_bumps_recovery;
#[path = "pathological/constant_response_smooth_baseline.rs"]
mod constant_response_smooth_baseline;
#[path = "pathological/duplicate_x_rows_safety.rs"]
mod duplicate_x_rows_safety;
#[path = "pathological/extrapolation_does_not_explode.rs"]
mod extrapolation_does_not_explode;
#[path = "pathological/feature_scale_stress.rs"]
mod feature_scale_stress;
#[path = "pathological/fuzz_fit_invariants.rs"]
mod fuzz_fit_invariants;
#[path = "pathological/low_noise_recovers_truth_tightly.rs"]
mod low_noise_recovers_truth_tightly;
#[path = "pathological/many_centers_does_not_degrade.rs"]
mod many_centers_does_not_degrade;
#[path = "pathological/narrow_gaussian_bump_quality.rs"]
mod narrow_gaussian_bump_quality;
#[path = "pathological/new_smooths_nan_predict_rejection.rs"]
mod new_smooths_nan_predict_rejection;
#[path = "pathological/new_smooths_small_n_stability.rs"]
mod new_smooths_small_n_stability;
#[path = "pathological/pathological_all_zero_weights_rejected.rs"]
mod pathological_all_zero_weights_rejected;
#[path = "pathological/pathological_empty_smooth_collection_behavior.rs"]
mod pathological_empty_smooth_collection_behavior;
#[path = "pathological/pathological_extreme_rho_reml_finite.rs"]
mod pathological_extreme_rho_reml_finite;
#[path = "pathological/pathological_n0_p0_rejected.rs"]
mod pathological_n0_p0_rejected;
#[path = "pathological/pathological_nan_inputs_rejected.rs"]
mod pathological_nan_inputs_rejected;
#[path = "pathological/pathological_rank_deficient_design_diagnostic.rs"]
mod pathological_rank_deficient_design_diagnostic;
#[path = "pathological/pathological_single_row_graceful_error.rs"]
mod pathological_single_row_graceful_error;
#[path = "pathological/polynomial_truth_smooth_recovery.rs"]
mod polynomial_truth_smooth_recovery;
#[path = "pathological/robust_clean_fit_invariance.rs"]
mod robust_clean_fit_invariance;
#[path = "pathological/robust_clean_fit_invariance_families.rs"]
mod robust_clean_fit_invariance_families;
#[path = "pathological/robust_never_fail_stress.rs"]
mod robust_never_fail_stress;
#[path = "pathological/small_n_smooth_safety.rs"]
mod small_n_smooth_safety;
#[path = "pathological/standard_gam_scaling.rs"]
mod standard_gam_scaling;
#[path = "pathological/step_function_recovery.rs"]
mod step_function_recovery;
#[path = "pathological/symmetric_truth_predict_symmetric.rs"]
mod symmetric_truth_predict_symmetric;
