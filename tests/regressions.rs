//! Grouped integration-test crate root for regression / bug-hunt tests (issue #1146).
//!
//! The formerly-top-level regression, repro, issue, and bug-hunt crates are
//! included here as modules so they link as ONE binary instead of one linker
//! invocation each. Add new regression-family tests as a module here.

#[path = "regressions/arrow_schur_bug_hunt.rs"]
mod arrow_schur_bug_hunt;
#[path = "regressions/bernoulli_marginal_slope_bug_hunt_1.rs"]
mod bernoulli_marginal_slope_bug_hunt_1;
#[path = "regressions/build_rs_ban_gates_present.rs"]
mod build_rs_ban_gates_present;
#[path = "regressions/bug_hunt_1089_small_n_gaussian_double_penalty_terminates.rs"]
mod bug_hunt_1089_small_n_gaussian_double_penalty_terminates;
#[path = "regressions/bug_hunt_874_cyclic_bs_cc_outer_loop_terminates.rs"]
mod bug_hunt_874_cyclic_bs_cc_outer_loop_terminates;
#[path = "regressions/bug_hunt_979_gaussian_duchon_inference_nuts_not_quadratic.rs"]
mod bug_hunt_979_gaussian_duchon_inference_nuts_not_quadratic;
#[path = "regressions/bug_hunt_979_margslope_duchon_slowdown.rs"]
mod bug_hunt_979_margslope_duchon_slowdown;
#[path = "regressions/bug_hunt_979_margslope_matern_logslope_slowdown.rs"]
mod bug_hunt_979_margslope_matern_logslope_slowdown;
#[path = "regressions/bug_hunt_affine_anchor_moment_deep_tail_precision.rs"]
mod bug_hunt_affine_anchor_moment_deep_tail_precision;
#[path = "regressions/bug_hunt_apply_inverse_link_cloglog_deep_tail_precision.rs"]
mod bug_hunt_apply_inverse_link_cloglog_deep_tail_precision;
#[path = "regressions/bug_hunt_apply_inverse_link_probit_deep_tail_precision.rs"]
mod bug_hunt_apply_inverse_link_probit_deep_tail_precision;
#[path = "regressions/bug_hunt_bc_clamped_startup_kkt_abort.rs"]
mod bug_hunt_bc_clamped_startup_kkt_abort;
#[path = "regressions/bug_hunt_beta_generative_noise_ignores_estimated_phi.rs"]
mod bug_hunt_beta_generative_noise_ignores_estimated_phi;
#[path = "regressions/bug_hunt_beta_observation_interval_ignores_estimated_phi.rs"]
mod bug_hunt_beta_observation_interval_ignores_estimated_phi;
#[path = "regressions/bug_hunt_beta_phi_frozen_at_null_predictor.rs"]
mod bug_hunt_beta_phi_frozen_at_null_predictor;
#[path = "regressions/bug_hunt_beta_phi_reported_at_fitted_eta.rs"]
mod bug_hunt_beta_phi_reported_at_fitted_eta;
#[path = "regressions/bug_hunt_beta_phi_smooth_plus_parametric.rs"]
mod bug_hunt_beta_phi_smooth_plus_parametric;
#[path = "regressions/bug_hunt_beta_regression_reml_nonfinite.rs"]
mod bug_hunt_beta_regression_reml_nonfinite;
#[path = "regressions/bug_hunt_bounded_term_identifiability_audit_zero_column.rs"]
mod bug_hunt_bounded_term_identifiability_audit_zero_column;
#[path = "regressions/bug_hunt_by_factor_smooth_column_not_loaded_from_cli.rs"]
mod bug_hunt_by_factor_smooth_column_not_loaded_from_cli;
#[path = "regressions/bug_hunt_cloglog_integrated_large_sigma.rs"]
mod bug_hunt_cloglog_integrated_large_sigma;
#[path = "regressions/bug_hunt_cloglog_inverse_link_deep_tail_clamp_flattens.rs"]
mod bug_hunt_cloglog_inverse_link_deep_tail_clamp_flattens;
#[path = "regressions/bug_hunt_cloglog_survival_large_sigma_asymptotic_biased_low.rs"]
mod bug_hunt_cloglog_survival_large_sigma_asymptotic_biased_low;
#[path = "regressions/bug_hunt_concave_shape_smooth_depends_on_warm_start_store.rs"]
mod bug_hunt_concave_shape_smooth_depends_on_warm_start_store;
#[path = "regressions/bug_hunt_conformal_held_out_calibration_fold_size_mismatch.rs"]
mod bug_hunt_conformal_held_out_calibration_fold_size_mismatch;
#[path = "regressions/bug_hunt_constrained_linear_active_bound_panics.rs"]
mod bug_hunt_constrained_linear_active_bound_panics;
#[path = "regressions/bug_hunt_corrected_covariance_response_scale_not_equivariant.rs"]
mod bug_hunt_corrected_covariance_response_scale_not_equivariant;
#[path = "regressions/bug_hunt_cyclic_period_option.rs"]
mod bug_hunt_cyclic_period_option;
#[path = "regressions/bug_hunt_debug_assert_ban_gate_marginal_slope.rs"]
mod bug_hunt_debug_assert_ban_gate_marginal_slope;
#[path = "regressions/bug_hunt_decoder_incoherence_hvp_is_gauss_newton_not_exact.rs"]
mod bug_hunt_decoder_incoherence_hvp_is_gauss_newton_not_exact;
#[path = "regressions/bug_hunt_diagnose_drops_response_column.rs"]
mod bug_hunt_diagnose_drops_response_column;
#[path = "regressions/bug_hunt_dispersion_location_scale_generate_predict_variance_agreement.rs"]
mod bug_hunt_dispersion_location_scale_generate_predict_variance_agreement;
#[path = "regressions/bug_hunt_dispersion_location_scale_observation_interval_symmetric_1346.rs"]
mod bug_hunt_dispersion_location_scale_observation_interval_symmetric_1346;
#[path = "regressions/bug_hunt_double_penalty_inflates_edf_instead_of_shrinking.rs"]
mod bug_hunt_double_penalty_inflates_edf_instead_of_shrinking;
#[path = "regressions/bug_hunt_double_penalty_shrinks_irrelevant_covariate_1266.rs"]
mod bug_hunt_double_penalty_shrinks_irrelevant_covariate_1266;
#[path = "regressions/bug_hunt_estimate_external_family_and_links.rs"]
mod bug_hunt_estimate_external_family_and_links;
#[path = "regressions/bug_hunt_evidence_gaussian_reml_topology.rs"]
mod bug_hunt_evidence_gaussian_reml_topology;
#[path = "regressions/bug_hunt_factor_smooth_degree_shrink_predict_replay.rs"]
mod bug_hunt_factor_smooth_degree_shrink_predict_replay;
#[path = "regressions/bug_hunt_family_link_paren_poisson_gamma_gaussian_unknown.rs"]
mod bug_hunt_family_link_paren_poisson_gamma_gaussian_unknown;
#[path = "regressions/bug_hunt_family_suite.rs"]
mod bug_hunt_family_suite;
#[path = "regressions/bug_hunt_fingerprinter_type_tag_does_not_disambiguate.rs"]
mod bug_hunt_fingerprinter_type_tag_does_not_disambiguate;
#[path = "regressions/bug_hunt_flexible_link_linkwiggle_joint_solve_aborts.rs"]
mod bug_hunt_flexible_link_linkwiggle_joint_solve_aborts;
#[path = "regressions/bug_hunt_gamlss_2_3.rs"]
mod bug_hunt_gamlss_2_3;
#[path = "regressions/bug_hunt_gamma_dispersion_location_scale_predictable_1119.rs"]
mod bug_hunt_gamma_dispersion_location_scale_predictable_1119;
#[path = "regressions/bug_hunt_gamma_dispersion_shape_locked_underestimate.rs"]
mod bug_hunt_gamma_dispersion_shape_locked_underestimate;
#[path = "regressions/bug_hunt_gamma_location_scale_generate_ignores_precision_channel.rs"]
mod bug_hunt_gamma_location_scale_generate_ignores_precision_channel;
#[path = "regressions/bug_hunt_gamma_observation_interval_symmetric_skew_miscoverage.rs"]
mod bug_hunt_gamma_observation_interval_symmetric_skew_miscoverage;
#[path = "regressions/bug_hunt_gamma_parametric_reml_nonfinite_cost.rs"]
mod bug_hunt_gamma_parametric_reml_nonfinite_cost;
#[path = "regressions/bug_hunt_gamma_quantile_small_shape_lower_tail.rs"]
mod bug_hunt_gamma_quantile_small_shape_lower_tail;
#[path = "regressions/bug_hunt_gamma_smooth_reml_startup_rejects_all_seeds.rs"]
mod bug_hunt_gamma_smooth_reml_startup_rejects_all_seeds;
#[path = "regressions/bug_hunt_gaussian_reml_lambda_invariant_to_response_rescale.rs"]
mod bug_hunt_gaussian_reml_lambda_invariant_to_response_rescale;
#[path = "regressions/bug_hunt_gaussian_reml_weight_rescaling_changes_fit.rs"]
mod bug_hunt_gaussian_reml_weight_rescaling_changes_fit;
#[path = "regressions/bug_hunt_gaussian_sample_refit_dispersion_collapse.rs"]
mod bug_hunt_gaussian_sample_refit_dispersion_collapse;
#[path = "regressions/bug_hunt_gaussian_smooth_high_leverage_alo_hessian_abort.rs"]
mod bug_hunt_gaussian_smooth_high_leverage_alo_hessian_abort;
#[path = "regressions/bug_hunt_gaussian_smooth_not_invariant_to_small_response_rescale.rs"]
mod bug_hunt_gaussian_smooth_not_invariant_to_small_response_rescale;
#[path = "regressions/bug_hunt_gaussian_smooth_shape_not_response_shift_invariant.rs"]
mod bug_hunt_gaussian_smooth_shape_not_response_shift_invariant;
#[path = "regressions/bug_hunt_gpu_pirls_row_module_path_break.rs"]
mod bug_hunt_gpu_pirls_row_module_path_break;
#[path = "regressions/bug_hunt_group_random_effect_sparse_exact_smoothing_correction.rs"]
mod bug_hunt_group_random_effect_sparse_exact_smoothing_correction;
#[path = "regressions/bug_hunt_ibp_assignment_hvp_drops_offdiagonal_coupling.rs"]
mod bug_hunt_ibp_assignment_hvp_drops_offdiagonal_coupling;
#[path = "regressions/bug_hunt_inference_model.rs"]
mod bug_hunt_inference_model;
#[path = "regressions/bug_hunt_input_loc_matern.rs"]
mod bug_hunt_input_loc_matern;
#[path = "regressions/bug_hunt_jumprelu_psd_majorizer_underestimates_curvature.rs"]
mod bug_hunt_jumprelu_psd_majorizer_underestimates_curvature;
#[path = "regressions/bug_hunt_latent_cloglog_integrated_slope_collapses_large_sigma.rs"]
mod bug_hunt_latent_cloglog_integrated_slope_collapses_large_sigma;
#[path = "regressions/bug_hunt_latent_sae_gated.rs"]
mod bug_hunt_latent_sae_gated;
#[path = "regressions/bug_hunt_linalg_low_rank_utils.rs"]
mod bug_hunt_linalg_low_rank_utils;
#[path = "regressions/bug_hunt_linear_box_constraint_violated_by_internal_scaling.rs"]
mod bug_hunt_linear_box_constraint_violated_by_internal_scaling;
#[path = "regressions/bug_hunt_location_scale_noise_floor_not_response_scale_equivariant.rs"]
mod bug_hunt_location_scale_noise_floor_not_response_scale_equivariant;
#[path = "regressions/bug_hunt_logit_integrated_deriv_small_sigma.rs"]
mod bug_hunt_logit_integrated_deriv_small_sigma;
#[path = "regressions/bug_hunt_logit_integrated_mean_large_sigma.rs"]
mod bug_hunt_logit_integrated_mean_large_sigma;
#[path = "regressions/bug_hunt_main_does_not_compile_at_head.rs"]
mod bug_hunt_main_does_not_compile_at_head;
#[path = "regressions/bug_hunt_matern_2d_default_collapse_1357.rs"]
mod bug_hunt_matern_2d_default_collapse_1357;
#[path = "regressions/bug_hunt_matern_2d_kappa_nonconvergence.rs"]
mod bug_hunt_matern_2d_kappa_nonconvergence;
#[path = "regressions/bug_hunt_matern_aniso_input_loc.rs"]
mod bug_hunt_matern_aniso_input_loc;
#[path = "regressions/bug_hunt_matern_zero_aniso_overridden_by_geometry.rs"]
mod bug_hunt_matern_zero_aniso_overridden_by_geometry;
#[path = "regressions/bug_hunt_matrix_2_2.rs"]
mod bug_hunt_matrix_2_2;
#[path = "regressions/bug_hunt_measure_jet_formula_fit_aborts_at_tight_outer_tol.rs"]
mod bug_hunt_measure_jet_formula_fit_aborts_at_tight_outer_tol;
#[path = "regressions/bug_hunt_monotone_shape_binding_constraint.rs"]
mod bug_hunt_monotone_shape_binding_constraint;
#[path = "regressions/bug_hunt_monotone_shape_smooth_aborts_fit.rs"]
mod bug_hunt_monotone_shape_smooth_aborts_fit;
#[path = "regressions/bug_hunt_monotonicity_hvp_extra_eps_factor.rs"]
mod bug_hunt_monotonicity_hvp_extra_eps_factor;
#[path = "regressions/bug_hunt_multinomial_blocks_collapse_to_zero_width.rs"]
mod bug_hunt_multinomial_blocks_collapse_to_zero_width;
#[path = "regressions/bug_hunt_multinomial_predict_requires_response_column.rs"]
mod bug_hunt_multinomial_predict_requires_response_column;
#[path = "regressions/bug_hunt_negative_binomial_fixed_theta_ignored.rs"]
mod bug_hunt_negative_binomial_fixed_theta_ignored;
#[path = "regressions/bug_hunt_negative_binomial_generative_noise_ignores_estimated_theta.rs"]
mod bug_hunt_negative_binomial_generative_noise_ignores_estimated_theta;
#[path = "regressions/bug_hunt_negative_binomial_theta_frozen_at_one.rs"]
mod bug_hunt_negative_binomial_theta_frozen_at_one;
#[path = "regressions/bug_hunt_nonnegative_constraint_kkt_abort_with_free_term.rs"]
mod bug_hunt_nonnegative_constraint_kkt_abort_with_free_term;
#[path = "regressions/bug_hunt_nonnegative_constraint_kkt_scale_invariant.rs"]
mod bug_hunt_nonnegative_constraint_kkt_scale_invariant;
#[path = "regressions/bug_hunt_numeric_by_smooth_column_not_loaded_from_cli.rs"]
mod bug_hunt_numeric_by_smooth_column_not_loaded_from_cli;
#[path = "regressions/bug_hunt_outer_strategy_persistent_warm_start.rs"]
mod bug_hunt_outer_strategy_persistent_warm_start;
#[path = "regressions/bug_hunt_overlapping_smooth_predict_design_mismatch.rs"]
mod bug_hunt_overlapping_smooth_predict_design_mismatch;
#[path = "regressions/bug_hunt_parametric_survival_predict_design_columns.rs"]
mod bug_hunt_parametric_survival_predict_design_columns;
#[path = "regressions/bug_hunt_poincare_distance_nearby_points_cancellation.rs"]
mod bug_hunt_poincare_distance_nearby_points_cancellation;
#[path = "regressions/bug_hunt_poisson_observation_interval_below_support.rs"]
mod bug_hunt_poisson_observation_interval_below_support;
#[path = "regressions/bug_hunt_predict_2_2.rs"]
mod bug_hunt_predict_2_2;
#[path = "regressions/bug_hunt_predict_ignores_unrelated_cli_column.rs"]
mod bug_hunt_predict_ignores_unrelated_cli_column;
#[path = "regressions/bug_hunt_predict_linear_term_clamped_to_training_range.rs"]
mod bug_hunt_predict_linear_term_clamped_to_training_range;
#[path = "regressions/bug_hunt_psis_gpd_moments_khat_saturates_heavy_tail.rs"]
mod bug_hunt_psis_gpd_moments_khat_saturates_heavy_tail;
#[path = "regressions/bug_hunt_quadrature_alo_smooth_test.rs"]
mod bug_hunt_quadrature_alo_smooth_test;
#[path = "regressions/bug_hunt_reduced_aft_location_scale_predict_surface.rs"]
mod bug_hunt_reduced_aft_location_scale_predict_surface;
#[path = "regressions/bug_hunt_reml_unified_2_3.rs"]
mod bug_hunt_reml_unified_2_3;
#[path = "regressions/bug_hunt_sample_generative_polya_gamma.rs"]
mod bug_hunt_sample_generative_polya_gamma;
#[path = "regressions/bug_hunt_scad_mcp_majorizer_not_psd.rs"]
mod bug_hunt_scad_mcp_majorizer_not_psd;
#[path = "regressions/bug_hunt_shape_constrained_alo_seed_validation_aborts_1191.rs"]
mod bug_hunt_shape_constrained_alo_seed_validation_aborts_1191;
#[path = "regressions/bug_hunt_sheaf_hessian_diag_wrong_on_self_loop_edge.rs"]
mod bug_hunt_sheaf_hessian_diag_wrong_on_self_loop_edge;
#[path = "regressions/bug_hunt_smooth_extrapolation_axes_whole_class.rs"]
mod bug_hunt_smooth_extrapolation_axes_whole_class;
#[path = "regressions/bug_hunt_smooth_not_covariate_scale_invariant.rs"]
mod bug_hunt_smooth_not_covariate_scale_invariant;
#[path = "regressions/bug_hunt_smooth_term_predict_flat_clamped_outside_training_range.rs"]
mod bug_hunt_smooth_term_predict_flat_clamped_outside_training_range;
#[path = "regressions/bug_hunt_softmax_entropy_majorizer_not_psd.rs"]
mod bug_hunt_softmax_entropy_majorizer_not_psd;
#[path = "regressions/bug_hunt_solver_links_and_topology.rs"]
mod bug_hunt_solver_links_and_topology;
#[path = "regressions/bug_hunt_sphere_gpu_macro_unscoped_breaks_build.rs"]
mod bug_hunt_sphere_gpu_macro_unscoped_breaks_build;
#[path = "regressions/bug_hunt_sphere_log_map_nearby_geodesic_cancellation.rs"]
mod bug_hunt_sphere_log_map_nearby_geodesic_cancellation;
#[path = "regressions/bug_hunt_sphere_pole_not_single_valued.rs"]
mod bug_hunt_sphere_pole_not_single_valued;
#[path = "regressions/bug_hunt_standard_normal_quantile_upper_tail_precision.rs"]
mod bug_hunt_standard_normal_quantile_upper_tail_precision;
#[path = "regressions/bug_hunt_subsample_k_cap_and_audit.rs"]
mod bug_hunt_subsample_k_cap_and_audit;
#[path = "regressions/bug_hunt_summary_penalty_cursor_skips_unpenalized_re.rs"]
mod bug_hunt_summary_penalty_cursor_skips_unpenalized_re;
#[path = "regressions/bug_hunt_survival_location_scale_exact_newton_workspace_state_reuse.rs"]
mod bug_hunt_survival_location_scale_exact_newton_workspace_state_reuse;
#[path = "regressions/bug_hunt_survival_location_scale_smooth_block_gradient_mismatch.rs"]
mod bug_hunt_survival_location_scale_smooth_block_gradient_mismatch;
#[path = "regressions/bug_hunt_survival_surface_independent_of_predict_exit.rs"]
mod bug_hunt_survival_surface_independent_of_predict_exit;
#[path = "regressions/bug_hunt_tensor_periodic_margin_predict_offbyone.rs"]
mod bug_hunt_tensor_periodic_margin_predict_offbyone;
#[path = "regressions/bug_hunt_terms_supporting.rs"]
mod bug_hunt_terms_supporting;
#[path = "regressions/bug_hunt_thinplate_formula_fit_knot_collision.rs"]
mod bug_hunt_thinplate_formula_fit_knot_collision;
#[path = "regressions/bug_hunt_torch_dispatch_penalties.rs"]
mod bug_hunt_torch_dispatch_penalties;
#[path = "regressions/bug_hunt_transformation_survival_python_path_aborts_on_outer_trial.rs"]
mod bug_hunt_transformation_survival_python_path_aborts_on_outer_trial;
#[path = "regressions/bug_hunt_tweedie_dispersion_frozen_at_one.rs"]
mod bug_hunt_tweedie_dispersion_frozen_at_one;
#[path = "regressions/bug_hunt_tweedie_dispersion_observation_interval.rs"]
mod bug_hunt_tweedie_dispersion_observation_interval;
#[path = "regressions/bug_hunt_univariate_smooth_small_k_rejected_not_degree_reduced.rs"]
mod bug_hunt_univariate_smooth_small_k_rejected_not_degree_reduced;
#[path = "regressions/bug_hunt_uv_lock_gamfit_version_stale.rs"]
mod bug_hunt_uv_lock_gamfit_version_stale;
#[path = "regressions/bug_hunt_weibull_by_factor_per_group_baseline.rs"]
mod bug_hunt_weibull_by_factor_per_group_baseline;
#[path = "regressions/bug_hunt_weibull_saved_baseline_scale_matches_anchor.rs"]
mod bug_hunt_weibull_saved_baseline_scale_matches_anchor;
#[path = "regressions/bug_hunt_weibull_survival_predict_not_degenerate.rs"]
mod bug_hunt_weibull_survival_predict_not_degenerate;
#[path = "regressions/bug_hunt_wps_corrected_edf_below_conditional.rs"]
mod bug_hunt_wps_corrected_edf_below_conditional;
#[path = "regressions/bug_hunt_wps_weighted_gram_invariants.rs"]
mod bug_hunt_wps_weighted_gram_invariants;
#[path = "regressions/bug_hunt_zero_weight_rows_not_equivalent_to_absent_rows.rs"]
mod bug_hunt_zero_weight_rows_not_equivalent_to_absent_rows;
#[path = "regressions/cli_ffi_fit_parity_1196.rs"]
mod cli_ffi_fit_parity_1196;
#[path = "regressions/cubic_cell_kernel_bug_hunt.rs"]
mod cubic_cell_kernel_bug_hunt;
#[path = "regressions/duchon_collocation_symmetry_psd_regression.rs"]
mod duchon_collocation_symmetry_psd_regression;
#[path = "regressions/factor_intercept_issue_157.rs"]
mod factor_intercept_issue_157;
#[path = "regressions/faer_ndarray_bug_hunt.rs"]
mod faer_ndarray_bug_hunt;
#[path = "regressions/formula_dsl_operator_family_issue_219.rs"]
mod formula_dsl_operator_family_issue_219;
#[path = "regressions/gamlss_binomial_location_scale_bug_hunt_3_of_3.rs"]
mod gamlss_binomial_location_scale_bug_hunt_3_of_3;
#[path = "regressions/gamlss_joint_derivatives_fd_bug_hunt.rs"]
mod gamlss_joint_derivatives_fd_bug_hunt;
#[path = "regressions/geometry_bug_hunt.rs"]
mod geometry_bug_hunt;
#[path = "regressions/gpu_bug_hunt_runtime_driver_diagnostics_memory.rs"]
mod gpu_bug_hunt_runtime_driver_diagnostics_memory;
#[path = "regressions/inference_bug_hunt.rs"]
mod inference_bug_hunt;
#[path = "regressions/issue_225_periodic_kernel.rs"]
mod issue_225_periodic_kernel;
#[path = "regressions/issue_239_poincare_self_distance.rs"]
mod issue_239_poincare_self_distance;
#[path = "regressions/issue_531_pure_duchon_probit_repro.rs"]
mod issue_531_pure_duchon_probit_repro;
#[path = "regressions/issue_532_wahba_sphere_intercept_collision.rs"]
mod issue_532_wahba_sphere_intercept_collision;
#[path = "regressions/issue_700_sz_factor_smooth_fits_and_predicts.rs"]
mod issue_700_sz_factor_smooth_fits_and_predicts;
#[path = "regressions/issue_704_by_factor_spatial_freeze_replay.rs"]
mod issue_704_by_factor_spatial_freeze_replay;
#[path = "regressions/issue_750_duchon_hybrid_length_scale_default.rs"]
mod issue_750_duchon_hybrid_length_scale_default;
#[path = "regressions/large_scale_convergence_regression.rs"]
mod large_scale_convergence_regression;
#[path = "regressions/large_scale_dense_regression_guard.rs"]
mod large_scale_dense_regression_guard;
#[path = "regressions/latent_cache_inner_bug_hunt.rs"]
mod latent_cache_inner_bug_hunt;
#[path = "regressions/measure_jet_aniso_isotropic_fallback_regression.rs"]
mod measure_jet_aniso_isotropic_fallback_regression;
#[path = "regressions/new_smooths_misc_regressions.rs"]
mod new_smooths_misc_regressions;
#[path = "regressions/no_rigor_regression.rs"]
mod no_rigor_regression;
#[path = "regressions/owed_work_issues_resolved.rs"]
mod owed_work_issues_resolved;
#[path = "regressions/regression_affine_anchor_moment_both_tails.rs"]
mod regression_affine_anchor_moment_both_tails;
#[path = "regressions/regression_apply_inverse_link_probit_independent_reference.rs"]
mod regression_apply_inverse_link_probit_independent_reference;
#[path = "regressions/regression_decoder_incoherence_majorizer_is_psd_gauss_newton.rs"]
mod regression_decoder_incoherence_majorizer_is_psd_gauss_newton;
#[path = "regressions/regression_ibp_assignment_hvp_block_structure.rs"]
mod regression_ibp_assignment_hvp_block_structure;
#[path = "regressions/regression_scale_basis_parsimony_501.rs"]
mod regression_scale_basis_parsimony_501;
#[path = "regressions/reml_laml_rho_derivatives_fd_bug_hunt.rs"]
mod reml_laml_rho_derivatives_fd_bug_hunt;
#[path = "regressions/reml_runtime_bug_hunt_ift_dispatch.rs"]
mod reml_runtime_bug_hunt_ift_dispatch;
#[path = "regressions/repro_1066_iso_kappa_2d_binomial.rs"]
mod repro_1066_iso_kappa_2d_binomial;
#[path = "regressions/repro_gamlss_linesearch_failure.rs"]
mod repro_gamlss_linesearch_failure;
#[path = "regressions/smooth_fit_term_collection_bug_hunt_2_3.rs"]
mod smooth_fit_term_collection_bug_hunt_2_3;
#[path = "regressions/solver_gpu_pirls_bug_hunt.rs"]
mod solver_gpu_pirls_bug_hunt;
#[path = "regressions/sparse_exact_bug_hunt.rs"]
mod sparse_exact_bug_hunt;
#[path = "regressions/te_formula_dispatch_issue_154.rs"]
mod te_formula_dispatch_issue_154;
#[path = "regressions/thin_plate_workspace_equivalence_regression.rs"]
mod thin_plate_workspace_equivalence_regression;
#[path = "regressions/tps_smallk_basis_regression.rs"]
mod tps_smallk_basis_regression;
#[path = "regressions/warm_start_quality_regression.rs"]
mod warm_start_quality_regression;
#[path = "regressions/warm_start_store_regressions.rs"]
mod warm_start_store_regressions;
#[path = "regressions/wiggle_option_validation_issue_377_373.rs"]
mod wiggle_option_validation_issue_377_373;
#[path = "regressions/workflow_bug_hunt.rs"]
mod workflow_bug_hunt;
#[path = "regressions/bug_hunt_1379_univariate_matern_gp_degenerate_range_penalty.rs"]
mod bug_hunt_1379_univariate_matern_gp_degenerate_range_penalty;
