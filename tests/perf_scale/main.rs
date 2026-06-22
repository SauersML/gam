//! Grouped integration-test crate root for perf_scale tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod glm_frozen_w_tensor_n_independence;
mod grid_spline_2d_exact_oracle;
mod grid_spline_2d_streaming_bench;
mod joint_newton_isotropic_tr_starvation;
mod large_scale_accuracy_sweep;
mod large_scale_ctn_bootstrap_repro;
mod large_scale_margslope_repro;
mod large_scale_perf_benchmark;
mod large_scale_reml_stress;
mod pair_surface_grid_consumer;
mod pairwise_reduce_reproducibility;
mod parallelism_pool_size_fit_invariance_1045;
mod perf_kappa_loop_n_scaling;
mod perf_rho_outer_loop_n_scaling;
mod pirls_beta_dispersion_bug;
mod pirls_gradient_finite_difference_cross_family;
mod power_law_analyzer;
mod power_law_common;
mod resource_policy_auto_strict;
mod rho_posterior_escalation_tiers;
mod rho_posterior_tier0_real_fit;
mod rho_posterior_tier1_sae_coverage;
mod row_measure_enrichment;
mod row_metric_contract;
mod row_metric_loud_vs_loadbearing;
mod sparse_data_smooth_fit;
mod sparse_dense_imbalance_diagnose;
mod sparse_design_to_csr_arc_returns_some_for_well_formed_matrix;
mod sparse_tensor_design_correctness;
mod trust_region_first_step_respects_max_radius;
mod trust_region_step_is_radius_bounded;
