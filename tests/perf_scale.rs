//! Grouped integration-test crate root for perf_scale tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "perf_scale/grid_spline_2d_exact_oracle.rs"]
mod grid_spline_2d_exact_oracle;
#[path = "perf_scale/grid_spline_2d_streaming_bench.rs"]
mod grid_spline_2d_streaming_bench;
#[path = "perf_scale/joint_newton_isotropic_tr_starvation.rs"]
mod joint_newton_isotropic_tr_starvation;
#[path = "perf_scale/large_scale_accuracy_sweep.rs"]
mod large_scale_accuracy_sweep;
#[path = "perf_scale/large_scale_ctn_bootstrap_repro.rs"]
mod large_scale_ctn_bootstrap_repro;
#[path = "perf_scale/large_scale_margslope_repro.rs"]
mod large_scale_margslope_repro;
#[path = "perf_scale/large_scale_perf_benchmark.rs"]
mod large_scale_perf_benchmark;
#[path = "perf_scale/large_scale_reml_stress.rs"]
mod large_scale_reml_stress;
#[path = "perf_scale/pair_surface_grid_consumer.rs"]
mod pair_surface_grid_consumer;
#[path = "perf_scale/pairwise_reduce_reproducibility.rs"]
mod pairwise_reduce_reproducibility;
#[path = "perf_scale/parallelism_pool_size_fit_invariance_1045.rs"]
mod parallelism_pool_size_fit_invariance_1045;
#[path = "perf_scale/perf_kappa_loop_n_scaling.rs"]
mod perf_kappa_loop_n_scaling;
#[path = "perf_scale/perf_rho_outer_loop_n_scaling.rs"]
mod perf_rho_outer_loop_n_scaling;
#[path = "perf_scale/pirls_beta_dispersion_bug.rs"]
mod pirls_beta_dispersion_bug;
#[path = "perf_scale/pirls_gradient_finite_difference_cross_family.rs"]
mod pirls_gradient_finite_difference_cross_family;
#[path = "perf_scale/power_law_analyzer.rs"]
mod power_law_analyzer;
#[path = "perf_scale/power_law_common.rs"]
mod power_law_common;
#[path = "perf_scale/resource_policy_auto_strict.rs"]
mod resource_policy_auto_strict;
#[path = "perf_scale/rho_posterior_escalation_tiers.rs"]
mod rho_posterior_escalation_tiers;
#[path = "perf_scale/rho_posterior_tier0_real_fit.rs"]
mod rho_posterior_tier0_real_fit;
#[path = "perf_scale/rho_posterior_tier1_sae_coverage.rs"]
mod rho_posterior_tier1_sae_coverage;
#[path = "perf_scale/row_measure_enrichment.rs"]
mod row_measure_enrichment;
#[path = "perf_scale/row_metric_contract.rs"]
mod row_metric_contract;
#[path = "perf_scale/row_metric_loud_vs_loadbearing.rs"]
mod row_metric_loud_vs_loadbearing;
#[path = "perf_scale/sparse_data_smooth_fit.rs"]
mod sparse_data_smooth_fit;
#[path = "perf_scale/sparse_dense_imbalance_diagnose.rs"]
mod sparse_dense_imbalance_diagnose;
#[path = "perf_scale/sparse_design_to_csr_arc_returns_some_for_well_formed_matrix.rs"]
mod sparse_design_to_csr_arc_returns_some_for_well_formed_matrix;
#[path = "perf_scale/sparse_tensor_design_correctness.rs"]
mod sparse_tensor_design_correctness;
#[path = "perf_scale/trust_region_first_step_respects_max_radius.rs"]
mod trust_region_first_step_respects_max_radius;
#[path = "perf_scale/trust_region_step_is_radius_bounded.rs"]
mod trust_region_step_is_radius_bounded;
