//! Grouped integration-test crate root for pathological tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod bimodal_two_bumps_recovery;
mod constant_response_smooth_baseline;
mod duplicate_x_rows_safety;
mod extrapolation_does_not_explode;
mod feature_scale_stress;
mod fuzz_fit_invariants;
mod low_noise_recovers_truth_tightly;
mod many_centers_does_not_degrade;
mod narrow_gaussian_bump_quality;
mod new_smooths_nan_predict_rejection;
mod new_smooths_small_n_stability;
mod pathological_all_zero_weights_rejected;
mod pathological_empty_smooth_collection_behavior;
mod pathological_extreme_rho_reml_finite;
mod pathological_n0_p0_rejected;
mod pathological_nan_inputs_rejected;
mod pathological_rank_deficient_design_diagnostic;
mod pathological_single_row_graceful_error;
mod polynomial_truth_smooth_recovery;
mod robust_clean_fit_invariance;
mod robust_clean_fit_invariance_families;
mod robust_never_fail_stress;
mod small_n_smooth_safety;
mod standard_gam_scaling;
mod step_function_recovery;
mod symmetric_truth_predict_symmetric;
