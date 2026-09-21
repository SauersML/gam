# MASTER_FAILURES

- Compile failures: **4**
- Workspace tests run: **NOT MEASURED** (the archive population was never listed)
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (8 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 100** (LOWER BOUND, not a count: Python API tests (job `cancelled`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `success`, matrix `success`, the build job published no archive test listing, so the population that should have run is unknown)
- gam-pyffi unit tests: **MEASURED** (job `failure`)
- Python API tests: **NOT MEASURED** (job `cancelled`)
- Python populations (slow + torch): **MEASURED** (job `failure`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python API tests (job `cancelled`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the Python surface was NOT measured — the Python job reported `cancelled`. The Python counter above is not a result.

> NOTE: the runtime surface was NOT measured — the workspace test population was not certified: the build job published no archive test listing, so the population that should have run is unknown; a shard reported ARCHIVE_MISSING. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.

## Compile failures

- `crates/gam-models/src/bms/mod.rs:4448:1` — [E0428] the name `anchor_law_2926_tests` is defined multiple times
- `crates/gam-models/src/inference/model.rs:7929:10` — [E0277] `saved_summary::SummaryPayload` doesn't implement `std::fmt::Debug`
- `?` — could not compile `gam-models` (lib test) due to 2 previous errors
- `?` — command `/home/runner/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin/cargo '--color=always' test --no-run --message-format json-render-diagnostics --workspace --exclude gam-pyffi --all-features --config 'profile.test.debug=0'` exited with code 101

## Runtime test failures

- **FAIL** `gam-pyffi` :: `batch_tests::circle_latent_recovers_circle_not_collapse`
- **FAIL** `gam-pyffi` :: `ffi::ffi_errors::saved_model_error_dispatch_tests::a_refused_saved_model_raises_its_category_with_its_variant_3008`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262`
- **FAIL** `gam-pyffi` :: `tests::blocks_negative_reml_score_backward_sign_matches_profile_perturbations`
- **FAIL** `gam-pyffi` :: `tests::position_batched_duchon_forward_matches_prebuilt_design`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_fit_is_output_rotation_equivariant`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_lambda_edf_are_shared_per_smooth`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_sigma2_pools_and_counts_unpenalized_columns`

## Python test failures

_Lower bound: 100 recorded before the run stopped. Unmeasured: Python API tests (job `cancelled`)._

- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_in_sample_r2_per_seed[0]`
- **FAIL** `python::tests/test_pygam_audit_predict_interval_hygiene` :: `test_gauss_small_estimated_scale_interval_coverage_is_nominal`
- **ABORT** `python::tests/test_examples_compose_tiers` :: `test_no_alternation_flag_is_respected`
- **FAIL** `python::tests/test_sae_atom_inference_roundtrip` :: `test_atom_inference_reports_functionals_and_no_e_value`
- **FAIL** `python::tests/test_basis_check_pvalue_calibration` :: `test_basis_check_is_uniform_under_an_adequate_basis[poisson-low-count]`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_in_sample_r2_per_seed[1]`
- **FAIL** `python::tests/test_sae_atom_inference_roundtrip` :: `test_atom_inference_reports_survive_json_roundtrip`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_mean_r2_across_seeds`
- **FAIL** `python::tests/test_basis_check_pvalue_calibration` :: `test_basis_check_is_uniform_under_an_adequate_basis[binomial-low-rate]`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_oos_r2`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_in_sample_r2_per_seed[2]`
- **ABORT** `python::tests/test_examples_compose_tiers` :: `test_compose_adds_reconstruction_over_linear_tier`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_oos_uses_fit_time_hyperparameters`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_sphere_atom_on_sphere_data`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_in_sample_r2_per_seed[7]`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_curved_circle_atom_in_sample_r2_per_seed[13]`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_euclidean_atom_dim_succeeds[2]`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_sae_manifold_oos_reconstruction_idempotence`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_duchon_2d_does_not_violate_collocation`
- **ABORT** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_duchon_1d_builds_primary_penalty`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[1-torus]`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[1-sphere]`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[2-circle]`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[2-euclidean]`
- **FAIL** `python::tests/test_sae_manifold_accuracy_oos` :: `test_near_duplicate_training_input_takes_oos_path_not_cache`
- **ABORT** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[2-sphere]`
- **FAIL** `python::tests/test_sae_manifold_oos_noise_sweep` :: `test_periodic_atom_oos_r2_noise_sweep[0.2]`
- **FAIL** `python::tests/test_sae_manifold_ordered_beta_bernoulli_prior_saturation` :: `test_penalized_quasi_laplace_resolves_true_k_under_prior_saturation`
- **ABORT** `python::tests/test_sae_manifold_ordered_beta_bernoulli_prior_saturation` :: `test_ordered_beta_bernoulli_assignments_decay_not_truncate_under_saturation`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[2-torus]`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_euclidean_atom_is_not_thin_plate`
- **ABORT** `python::tests/test_sae_manifold_euclidean_k4_oos_issue_1132` :: `test_euclidean_k4_fit_and_oos_reconstruct_issue_1132`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_summary_topology_matches_atom_basis[sphere-sphere-2]`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_summary_topology_matches_atom_basis[torus-torus-2]`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_payload_round_trip_preserves_topology[sphere-sphere-2]`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_payload_round_trip_preserves_topology[torus-torus-2]`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_summary_topology_matches_atom_basis[periodic-circle-1]`
- **FAIL** `python::tests/test_sae_oos_returns_converged_latents_issue_1229` :: `test_oos_returned_assignments_reconstruct_the_returned_fitted`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_geometry_plans_and_topology_are_internally_consistent`
- **FAIL** `python::tests/test_sae_oos_returns_converged_latents_issue_1229` :: `test_oos_returned_routing_is_the_softmax_of_the_returned_logits`
- **FAIL** `python::tests/test_sae_manifold_topology_stale` :: `test_linear_topology_is_distinct_from_euclidean_quadratic_patch`
- **FAIL** `python::tests/test_smooth_lr_pvalue_calibration` :: `test_smooth_lr_is_sized_and_keeps_its_power[gaussian_n60]`
- **ABORT** `python::tests/test_smooth_lr_pvalue_calibration` :: `test_smooth_lr_is_sized_and_keeps_its_power[binomial_n400]`
- **FAIL** `python::tests/test_survival_marginal_slope_large_scale_hard` :: `test_survival_marginal_slope_large_scale_startup_seeds_do_not_all_reject`
- **FAIL** `python::tests/test_sample_smoothing_corrected_coverage` :: `test_sample_mean_intervals_cover_like_predict[binomial]`
- **ABORT** `python::tests/test_examples_sae_supervised` :: `test_sae_supervised_end_to_end_returns_uniform_result`
- **FAIL** `python::tests/test_sample_smoothing_corrected_coverage` :: `test_sample_mean_intervals_cover_like_predict[poisson]`
- **ABORT** `python::tests/test_smooth_lr_pvalue_calibration` :: `test_smooth_lr_is_sized_and_keeps_its_power[poisson_n200]`
- **ABORT** `python::tests/test_sae_manifold_decoder_incoherence_crossgram` :: `test_decoder_incoherence_reduces_recovered_cross_atom_decoder_cross_gram`
- **FAIL** `python::tests/test_sae_manifold_dim_matrix` :: `test_sae_manifold_fits_each_topology_dimension_pair[1-euclidean]`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_multi_atom_duchon_mix`
- **FAIL** `python::tests/test_sae_manifold_multi_topology` :: `test_single_atom_recovers_each_supported_topology[duchon]`
- **FAIL** `python::tests/test_sae_manifold_capacity` :: `test_penalized_quasi_laplace_picks_k1_on_one_harmonic_data`
- **FAIL** `python::tests/test_sae_manifold_oos_noise_sweep` :: `test_periodic_atom_oos_r2_noise_sweep[0.05]`
- **FAIL** `python::tests/test_sae_manifold_oos_noise_sweep` :: `test_periodic_atom_oos_r2_noise_sweep[0.1]`
- **FAIL** `python::tests/test_sae_manifold_oos_noise_sweep` :: `test_periodic_atom_oos_r2_noise_sweep[0.01]`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_duchon_atom_dim_succeeds[3]`
- **ABORT** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_duchon_atom_dim_succeeds[1]`
- **ABORT** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_euclidean_atom_dim_succeeds[1]`
- **FAIL** `python::tests/test_sae_manifold_duchon_euclidean_issue` :: `test_sae_manifold_fit_duchon_atom_dim_succeeds[2]`
- **FAIL** `python::tests/test_sae_manifold_row_loss_weights_1062` :: `test_uniform_weights_match_unweighted`
- **FAIL** `python::tests/test_sae_manifold_sphere_pole_stability` :: `test_sphere_atom_stable_at_poles`
- **FAIL** `python::tests/test_sae_manifold_multi_topology` :: `test_single_atom_recovers_each_supported_topology[sphere]`
- **FAIL** `python::tests/test_sae_manifold_multi_topology` :: `test_single_atom_recovers_each_supported_topology[euclidean]`
- **FAIL** `python::tests/test_sae_manifold_multi_topology` :: `test_single_atom_recovers_each_supported_topology[periodic]`
- **ABORT** `python::tests/test_examples_sae_supervised` :: `test_sae_supervised_oos_predict_runs_the_frozen_decoder_encoder`
- **FAIL** `python::tests/test_sae_manifold_ordered_beta_bernoulli_prior_saturation` :: `test_reml_keeps_k1_winner_under_saturation`
- **ABORT** `python::tests/test_examples_sae_supervised` :: `test_sae_supervised_predicts_on_training_X`
- **FAIL** `python::tests/test_sae_manifold_row_loss_weights_1062` :: `test_row_weights_pull_fit_toward_upweighted_rows`
- **FAIL** `python::tests/test_sae_manifold_curved_beats_linear` :: `test_curved_atom_beats_linear_shards_on_one_harmonic`
- **FAIL** `python::tests/torch/test_basis_evaluators_match_rust` :: `test_basis_evaluators_match_rust`
- **FAIL** `python::tests/torch/test_gated_sae_decoder_atoms_match_rust` :: `test_gated_sae_decoder_atoms_match_rust`
- **FAIL** `python::tests/torch/test_gated_sae_decoder_heaviside_contract` :: `test_python_decode_matches_heaviside_reference`
- **FAIL** `python::tests/torch/test_gated_sae_decoder_heaviside_contract` :: `test_python_rust_parity_random_inputs`
- **FAIL** `python::tests/torch/test_gated_sae_decoder_heaviside_contract` :: `test_python_rust_parity_float32_inputs`
- **FAIL** `python::tests/torch/test_gradcheck` :: `test_gaussian_reml_fit_blocks_gradcheck`
- **FAIL** `python::tests/torch/test_harvest` :: `test_full_factorization_is_explicit_exact_without_fake_tail`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiable_factor_fit_default_auto_weights_issue_790`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiable_factor_fit_smoke`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiable_factor_fit_certifies_stationarity`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiable_factor_fit_free_block_has_unit_second_moment`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiable_factor_fit_budget_does_not_move_certified_point`
- **FAIL** `python::tests/torch/test_identifiable_factor_fit` :: `test_identifiability_check_flags_constant_aux`
- **FAIL** `python::tests/torch/test_numpy_parity` :: `test_duchon_basis_parity`
- **FAIL** `python::tests/torch/test_reml_backward_matches_rust_gradient` :: `test_reml_backward_matches_rust_gradient`
- **FAIL** `python::tests/torch/test_reml_blocks_backward` :: `test_public_blocks_gradcheck`
- **FAIL** `python::tests/torch/test_reml_blocks_backward` :: `test_blocks_function_gradcheck`
- **FAIL** `python::tests/torch/test_reml_ill_conditioned_backward` :: `test_backward_does_not_raise_when_lambda_saturates`
- **FAIL** `python::tests/torch/test_sae_e2e` :: `test_encoder_receives_gradient_through_reml`
- **FAIL** `python::tests/torch/test_shared_scale_orthogonal_reml` :: `test_shared_scale_orthogonal_matches_dense_joint_when_cross_gram_zero`
- **FAIL** `python::tests/test_python_api` :: `test_sklearn_classifier_roundtrip`
- **FAIL** `python::tests/test_python_api` :: `test_survival_marginal_slope_weibull_n3000_returns_under_60s`
- **FAIL** `python::tests/bench_large_scale_runner_test/LargeScaleRunnerTests` :: `test_large_scale_preflight_accepts_production_marginal_slope_width`
- **FAIL** `python::tests/bench_large_scale_runner_test/LargeScaleRunnerTests` :: `test_marginal_slope_formula_supports_linkwiggle_and_scorewarp`
- **FAIL** `python::tests/bench_large_scale_runner_test/MarkerContractTests` :: `test_every_parsed_marker_family_still_has_a_live_emission_site`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[2.0]`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[-2.0]`
- **FAIL** `python::tests/test_ci_no_orphan_python_tests_1512` :: `test_every_collectible_test_file_is_reached_by_some_ci_step`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_survival_location_scale_regressor_prediction_does_not_saturate`
- **FAIL** `python::tests/test_survival_save_load_roundtrip` :: `test_survival_location_scale_save_load_predict_roundtrips`

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

