//! Grouped integration-test crate root for basis_smooth tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "basis_smooth/aniso_integration.rs"]
mod aniso_integration;
#[path = "basis_smooth/aniso_population_calibration.rs"]
mod aniso_population_calibration;
#[path = "basis_smooth/barrier_term_diverges_at_constraint_boundary.rs"]
mod barrier_term_diverges_at_constraint_boundary;
#[path = "basis_smooth/basis_anisotropic_axis_derivative_independence.rs"]
mod basis_anisotropic_axis_derivative_independence;
#[path = "basis_smooth/basis_derivative_symmetry.rs"]
mod basis_derivative_symmetry;
#[path = "basis_smooth/basis_duchon_anisotropic_dominant_axis_factorization.rs"]
mod basis_duchon_anisotropic_dominant_axis_factorization;
#[path = "basis_smooth/basis_duchon_mixed_periodicity_auto_matches_manual.rs"]
mod basis_duchon_mixed_periodicity_auto_matches_manual;
#[path = "basis_smooth/basis_log_kappa_derivative_boundary_behavior.rs"]
mod basis_log_kappa_derivative_boundary_behavior;
#[path = "basis_smooth/basis_matern_double_penalty_log_kappa_derivative_fd.rs"]
mod basis_matern_double_penalty_log_kappa_derivative_fd;
#[path = "basis_smooth/basis_matern_log_kappa_first_derivative_fd.rs"]
mod basis_matern_log_kappa_first_derivative_fd;
#[path = "basis_smooth/basis_matern_log_kappa_penalty_derivative_fd.rs"]
mod basis_matern_log_kappa_penalty_derivative_fd;
#[path = "basis_smooth/basis_matern_log_kappa_second_derivative_fd.rs"]
mod basis_matern_log_kappa_second_derivative_fd;
#[path = "basis_smooth/basis_workspace_and_nonworkspace_match.rs"]
mod basis_workspace_and_nonworkspace_match;
#[path = "basis_smooth/bc_alias_equivalence.rs"]
mod bc_alias_equivalence;
#[path = "basis_smooth/bc_anchored_hermite_stability.rs"]
mod bc_anchored_hermite_stability;
#[path = "basis_smooth/bc_anchored_large_k_robust.rs"]
mod bc_anchored_large_k_robust;
#[path = "basis_smooth/bc_anchored_variants_predict_works.rs"]
mod bc_anchored_variants_predict_works;
#[path = "basis_smooth/bc_clamped_predict_shape_bug.rs"]
mod bc_clamped_predict_shape_bug;
#[path = "basis_smooth/bc_fit_quality_sanity.rs"]
mod bc_fit_quality_sanity;
#[path = "basis_smooth/bc_minimum_k_stability.rs"]
mod bc_minimum_k_stability;
#[path = "basis_smooth/bc_periodic_combination_rejected.rs"]
mod bc_periodic_combination_rejected;
#[path = "basis_smooth/bc_predict_dimension_invariants.rs"]
mod bc_predict_dimension_invariants;
#[path = "basis_smooth/bspline_derivative_fd_oracle.rs"]
mod bspline_derivative_fd_oracle;
#[path = "basis_smooth/bspline_derivative_identity_bug.rs"]
mod bspline_derivative_identity_bug;
#[path = "basis_smooth/bspline_knot_options_formula.rs"]
mod bspline_knot_options_formula;
#[path = "basis_smooth/bspline_partition_unity_degree_1.rs"]
mod bspline_partition_unity_degree_1;
#[path = "basis_smooth/bspline_partition_unity_degree_2.rs"]
mod bspline_partition_unity_degree_2;
#[path = "basis_smooth/bspline_partition_unity_degree_3.rs"]
mod bspline_partition_unity_degree_3;
#[path = "basis_smooth/bspline_partition_unity_degree_4.rs"]
mod bspline_partition_unity_degree_4;
#[path = "basis_smooth/bspline_partition_unity_degree_5.rs"]
mod bspline_partition_unity_degree_5;
#[path = "basis_smooth/cyclic_bspline_first_derivative_periodicity_breaks.rs"]
mod cyclic_bspline_first_derivative_periodicity_breaks;
#[path = "basis_smooth/cyclic_bspline_second_derivative_periodicity_breaks.rs"]
mod cyclic_bspline_second_derivative_periodicity_breaks;
#[path = "basis_smooth/cyclic_duchon_torus_containment.rs"]
mod cyclic_duchon_torus_containment;
#[path = "basis_smooth/difference_smooth_formula.rs"]
mod difference_smooth_formula;
#[path = "basis_smooth/duchon_ard_quality.rs"]
mod duchon_ard_quality;
#[path = "basis_smooth/duchon_basis_build_d_scaling.rs"]
mod duchon_basis_build_d_scaling;
#[path = "basis_smooth/duchon_collocation_conditioning.rs"]
mod duchon_collocation_conditioning;
#[path = "basis_smooth/duchon_default_cubic_resolution.rs"]
mod duchon_default_cubic_resolution;
#[path = "basis_smooth/duchon_dimension_scaling_probe.rs"]
mod duchon_dimension_scaling_probe;
#[path = "basis_smooth/duchon_hilbert_scale.rs"]
mod duchon_hilbert_scale;
#[path = "basis_smooth/duchon_integration.rs"]
mod duchon_integration;
#[path = "basis_smooth/duchon_kernel_accuracy_bench.rs"]
mod duchon_kernel_accuracy_bench;
#[path = "basis_smooth/duchon_kernel_s0_vs_cubic.rs"]
mod duchon_kernel_s0_vs_cubic;
#[path = "basis_smooth/duchon_order_nullspace.rs"]
mod duchon_order_nullspace;
#[path = "basis_smooth/duchon_per_axis_relevance.rs"]
mod duchon_per_axis_relevance;
#[path = "basis_smooth/duchon_scale_and_memory.rs"]
mod duchon_scale_and_memory;
#[path = "basis_smooth/duchon_sin8_quality.rs"]
mod duchon_sin8_quality;
#[path = "basis_smooth/duchon_structural_seminorms.rs"]
mod duchon_structural_seminorms;
#[path = "basis_smooth/effective_jacobian_at_bms_marginal_slope.rs"]
mod effective_jacobian_at_bms_marginal_slope;
#[path = "basis_smooth/effective_jacobian_at_gamlss.rs"]
mod effective_jacobian_at_gamlss;
#[path = "basis_smooth/effective_jacobian_at_survival_location_scale.rs"]
mod effective_jacobian_at_survival_location_scale;
#[path = "basis_smooth/effective_jacobian_at_survival_marginal_slope.rs"]
mod effective_jacobian_at_survival_marginal_slope;
#[path = "basis_smooth/effective_jacobian_at_timewiggle.rs"]
mod effective_jacobian_at_timewiggle;
#[path = "basis_smooth/factor_smooth_formula.rs"]
mod factor_smooth_formula;
#[path = "basis_smooth/factor_smooths_formula.rs"]
mod factor_smooths_formula;
#[path = "basis_smooth/matern_2d_iso_kappa_outer_gradient_fd.rs"]
mod matern_2d_iso_kappa_outer_gradient_fd;
#[path = "basis_smooth/matern_all_nu_sweep_diagnose.rs"]
mod matern_all_nu_sweep_diagnose;
#[path = "basis_smooth/matern_extreme_length_scale.rs"]
mod matern_extreme_length_scale;
#[path = "basis_smooth/matern_formula_integration.rs"]
mod matern_formula_integration;
#[path = "basis_smooth/matern_high_frequency_init.rs"]
mod matern_high_frequency_init;
#[path = "basis_smooth/matern_integration.rs"]
mod matern_integration;
#[path = "basis_smooth/matern_length_scale_sensitivity.rs"]
mod matern_length_scale_sensitivity;
#[path = "basis_smooth/matern_nu_sweep_easy_truth.rs"]
mod matern_nu_sweep_easy_truth;
#[path = "basis_smooth/matern_quality_batch_e.rs"]
mod matern_quality_batch_e;
#[path = "basis_smooth/mspline_ispline_scalar_identity_bug.rs"]
mod mspline_ispline_scalar_identity_bug;
#[path = "basis_smooth/penalized_complexity_prior.rs"]
mod penalized_complexity_prior;
#[path = "basis_smooth/penalty_joint_nullspace_check.rs"]
mod penalty_joint_nullspace_check;
#[path = "basis_smooth/penalty_prior_mean_hard.rs"]
mod penalty_prior_mean_hard;
#[path = "basis_smooth/periodic_1d_degree_sweep.rs"]
mod periodic_1d_degree_sweep;
#[path = "basis_smooth/periodic_1d_higher_order_continuity.rs"]
mod periodic_1d_higher_order_continuity;
#[path = "basis_smooth/periodic_1d_origin_offset.rs"]
mod periodic_1d_origin_offset;
#[path = "basis_smooth/periodic_1d_partial_data_coverage.rs"]
mod periodic_1d_partial_data_coverage;
#[path = "basis_smooth/periodic_1d_seam_discontinuity_bug.rs"]
mod periodic_1d_seam_discontinuity_bug;
#[path = "basis_smooth/periodic_bspline_wrap_derivative_continuity_bug.rs"]
mod periodic_bspline_wrap_derivative_continuity_bug;
#[path = "basis_smooth/periodic_curve.rs"]
mod periodic_curve;
#[path = "basis_smooth/periodic_default_period_behavior.rs"]
mod periodic_default_period_behavior;
#[path = "basis_smooth/periodic_duchon_seam_continuity.rs"]
mod periodic_duchon_seam_continuity;
#[path = "basis_smooth/periodic_formula_integration.rs"]
mod periodic_formula_integration;
#[path = "basis_smooth/periodic_quality_batch_c.rs"]
mod periodic_quality_batch_c;
#[path = "basis_smooth/periodic_spline_1d.rs"]
mod periodic_spline_1d;
#[path = "basis_smooth/precision_gamma_hyperprior.rs"]
mod precision_gamma_hyperprior;
#[path = "basis_smooth/reml_iter_reduction_correctness.rs"]
mod reml_iter_reduction_correctness;
#[path = "basis_smooth/reml_scale_invariance.rs"]
mod reml_scale_invariance;
#[path = "basis_smooth/reml_trace_hutchinson_validation.rs"]
mod reml_trace_hutchinson_validation;
#[path = "basis_smooth/ridge_2d_smooth_quality.rs"]
mod ridge_2d_smooth_quality;
#[path = "basis_smooth/ridge_ledger_invariants.rs"]
mod ridge_ledger_invariants;
#[path = "basis_smooth/smooth_rejects_constant_input.rs"]
mod smooth_rejects_constant_input;
#[path = "basis_smooth/smooth_term_lr_bartlett_calibration.rs"]
mod smooth_term_lr_bartlett_calibration;
#[path = "basis_smooth/smooth_term_lr_size_calibration.rs"]
mod smooth_term_lr_size_calibration;
#[path = "basis_smooth/spline_k_sweep_easy_truth.rs"]
mod spline_k_sweep_easy_truth;
#[path = "basis_smooth/spline_scan_exact_oracle.rs"]
mod spline_scan_exact_oracle;
#[path = "basis_smooth/spline_scan_workflow_equivalence.rs"]
mod spline_scan_workflow_equivalence;
#[path = "basis_smooth/t2_tensor_penalty_decomposition.rs"]
mod t2_tensor_penalty_decomposition;
#[path = "basis_smooth/te_k_consistency_easy_truth.rs"]
mod te_k_consistency_easy_truth;
#[path = "basis_smooth/te_tensor_2d_hifreq_quality.rs"]
mod te_tensor_2d_hifreq_quality;
#[path = "basis_smooth/tensor_3d_smooth_robust.rs"]
mod tensor_3d_smooth_robust;
#[path = "basis_smooth/tensor_clamped_margin.rs"]
mod tensor_clamped_margin;
#[path = "basis_smooth/thin_plate_integration.rs"]
mod thin_plate_integration;
#[path = "basis_smooth/ti_tensor_interaction_smooth.rs"]
mod ti_tensor_interaction_smooth;
#[path = "basis_smooth/weighted_blockwise_penalty_sum_rejects_negative_weights.rs"]
mod weighted_blockwise_penalty_sum_rejects_negative_weights;
#[path = "basis_smooth/wiggle_penalty_orders_do_not_drop_requested_orders.rs"]
mod wiggle_penalty_orders_do_not_drop_requested_orders;
