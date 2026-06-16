//! Grouped integration-test crate root for manifolds tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "manifolds/cylinder_fit_profile.rs"]
mod cylinder_fit_profile;
#[path = "manifolds/cylinder_periodic.rs"]
mod cylinder_periodic;
#[path = "manifolds/cylinder_tensor_seam_continuity.rs"]
mod cylinder_tensor_seam_continuity;
#[path = "manifolds/cylinder_torus_advanced_periodicity.rs"]
mod cylinder_torus_advanced_periodicity;
#[path = "manifolds/cylinder_torus_quality_batch_d.rs"]
mod cylinder_torus_quality_batch_d;
#[path = "manifolds/equivariant_lie_atom.rs"]
mod equivariant_lie_atom;
#[path = "manifolds/geodesic_small_step_matches_exp_to_second_order.rs"]
mod geodesic_small_step_matches_exp_to_second_order;
#[path = "manifolds/layer_transport_functorial.rs"]
mod layer_transport_functorial;
#[path = "manifolds/poincare_conformal_dirichlet_penalty.rs"]
mod poincare_conformal_dirichlet_penalty;
#[path = "manifolds/poincare_tangent_decode_backward_fd.rs"]
mod poincare_tangent_decode_backward_fd;
#[path = "manifolds/product_retraction_matches_componentwise.rs"]
mod product_retraction_matches_componentwise;
#[path = "manifolds/retraction_round_trip_sphere_log_exp_consistency.rs"]
mod retraction_round_trip_sphere_log_exp_consistency;
#[path = "manifolds/spd_retraction_preserves_positive_definiteness.rs"]
mod spd_retraction_preserves_positive_definiteness;
#[path = "manifolds/sphere_antipodal_concentration.rs"]
mod sphere_antipodal_concentration;
#[path = "manifolds/sphere_basis_jet.rs"]
mod sphere_basis_jet;
#[path = "manifolds/sphere_binomial_logit_fit.rs"]
mod sphere_binomial_logit_fit;
#[path = "manifolds/sphere_binomial_pseudo_m4.rs"]
mod sphere_binomial_pseudo_m4;
#[path = "manifolds/sphere_both_kernels_fit_smooth_truth.rs"]
mod sphere_both_kernels_fit_smooth_truth;
#[path = "manifolds/sphere_center_data_coincidence.rs"]
mod sphere_center_data_coincidence;
#[path = "manifolds/sphere_constant_truth_flat_fit.rs"]
mod sphere_constant_truth_flat_fit;
#[path = "manifolds/sphere_degenerate_data_robust.rs"]
mod sphere_degenerate_data_robust;
#[path = "manifolds/sphere_exp_map_vjp_matches_finite_difference.rs"]
mod sphere_exp_map_vjp_matches_finite_difference;
#[path = "manifolds/sphere_harmonic_default_degree.rs"]
mod sphere_harmonic_default_degree;
#[path = "manifolds/sphere_harmonic_large_L_stability.rs"]
mod sphere_harmonic_large_l_stability;
#[path = "manifolds/sphere_high_freq_truth_robust.rs"]
mod sphere_high_freq_truth_robust;
#[path = "manifolds/sphere_integration.rs"]
mod sphere_integration;
#[path = "manifolds/sphere_invalid_lat_handling.rs"]
mod sphere_invalid_lat_handling;
#[path = "manifolds/sphere_low_snr_robust.rs"]
mod sphere_low_snr_robust;
#[path = "manifolds/sphere_m4_lambda_diagnostic.rs"]
mod sphere_m4_lambda_diagnostic;
#[path = "manifolds/sphere_methods_agree_on_smooth_truth.rs"]
mod sphere_methods_agree_on_smooth_truth;
#[path = "manifolds/sphere_methods_diagnose.rs"]
mod sphere_methods_diagnose;
#[path = "manifolds/sphere_near_pole_predict_stability.rs"]
mod sphere_near_pole_predict_stability;
#[path = "manifolds/sphere_options_batch_g.rs"]
mod sphere_options_batch_g;
#[path = "manifolds/sphere_overresourced_small_n.rs"]
mod sphere_overresourced_small_n;
#[path = "manifolds/sphere_penalty_order_sweep.rs"]
mod sphere_penalty_order_sweep;
#[path = "manifolds/sphere_pole_and_seam_predict.rs"]
mod sphere_pole_and_seam_predict;
#[path = "manifolds/sphere_pole_continuity.rs"]
mod sphere_pole_continuity;
#[path = "manifolds/sphere_quality_batch_a.rs"]
mod sphere_quality_batch_a;
#[path = "manifolds/sphere_response_robustness.rs"]
mod sphere_response_robustness;
#[path = "manifolds/sphere_seam_zero_to_two_pi.rs"]
mod sphere_seam_zero_to_two_pi;
#[path = "manifolds/sphere_single_hemisphere_data_does_not_blow_up.rs"]
mod sphere_single_hemisphere_data_does_not_blow_up;
#[path = "manifolds/sphere_single_hemisphere_extrapolation.rs"]
mod sphere_single_hemisphere_extrapolation;
#[path = "manifolds/sphere_small_k_corner_cases.rs"]
mod sphere_small_k_corner_cases;
#[path = "manifolds/sphere_top_pole_fit_quality.rs"]
mod sphere_top_pole_fit_quality;
#[path = "manifolds/sphere_uncertainty_intervals.rs"]
mod sphere_uncertainty_intervals;
#[path = "manifolds/sphere_wahba_kernels_are_distinct.rs"]
mod sphere_wahba_kernels_are_distinct;
#[path = "manifolds/sphere_with_bc_option_rejected.rs"]
mod sphere_with_bc_option_rejected;
#[path = "manifolds/spherical_spline.rs"]
mod spherical_spline;
#[path = "manifolds/spherical_wahba_spectrum.rs"]
mod spherical_wahba_spectrum;
#[path = "manifolds/stiefel_exp_map_vjp.rs"]
mod stiefel_exp_map_vjp;
#[path = "manifolds/torus_wraps_angles_to_principal_interval.rs"]
mod torus_wraps_angles_to_principal_interval;
#[path = "manifolds/wahba_kernel_spectral_truth.rs"]
mod wahba_kernel_spectral_truth;
