# MASTER_FAILURES

- Compile failures: **3**
- Workspace tests run: **NOT MEASURED** (the archive population was never listed)
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (10 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 12** (LOWER BOUND, not a count: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `success`, matrix `success`, the build job published no archive test listing, so the population that should have run is unknown)
- gam-pyffi unit tests: **MEASURED** (job `failure`)
- Python API tests: **NOT MEASURED** (job `cancelled`)
- Python populations (slow + torch): **NOT MEASURED** (job `cancelled`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the Python surface was NOT measured — the Python job reported `cancelled`. The Python counter above is not a result.

> NOTE: the runtime surface was NOT measured — the workspace test population was not certified: the build job published no archive test listing, so the population that should have run is unknown; a shard reported ARCHIVE_MISSING. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.

## Compile failures

- `crates/gam-models/src/inference/model_payload_builders.rs:3039:34` — [E0505] cannot move out of `conditional` because it is borrowed
- `?` — could not compile `gam-models` (lib test) due to 1 previous error
- `?` — command `/home/runner/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin/cargo '--color=always' test --no-run --message-format json-render-diagnostics --workspace --exclude gam-pyffi --all-features --config 'profile.test.debug=0'` exited with code 101

## Runtime test failures

- **FAIL** `gam-pyffi` :: `batch_tests::circle_latent_recovers_circle_not_collapse`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262`
- **FAIL** `gam-pyffi` :: `isometry_decoder_jet_facade_tests::facade_isometry_hvp_with_the_third_decoder_jet_is_the_exact_hessian_2933`
- **FAIL** `gam-pyffi` :: `tests::batched_state_round_trip_matches_refit`
- **FAIL** `gam-pyffi` :: `tests::blocks_negative_reml_score_backward_sign_matches_profile_perturbations`
- **FAIL** `gam-pyffi` :: `tests::manifold_sae_structured_metric_without_behavior_shard_is_loadable`
- **FAIL** `gam-pyffi` :: `tests::position_batched_duchon_forward_matches_prebuilt_design`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_fit_is_output_rotation_equivariant`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_lambda_edf_are_shared_per_smooth`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_sigma2_pools_and_counts_unpenalized_columns`

## Python test failures

_Lower bound: 12 recorded before the run stopped. Unmeasured: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`)._

- **FAIL** `python::tests/test_python_api` :: `test_sklearn_classifier_roundtrip`
- **FAIL** `python::tests/test_python_api` :: `test_survival_marginal_slope_weibull_n3000_returns_under_60s`
- **FAIL** `python::tests/test_python_api` :: `test_gaussian_reml_fit_all_shape_constraints_do_not_panic[convex]`
- **FAIL** `python::tests/bench_large_scale_runner_test/LargeScaleRunnerTests` :: `test_large_scale_preflight_accepts_production_marginal_slope_width`
- **FAIL** `python::tests/bench_large_scale_runner_test/LargeScaleRunnerTests` :: `test_marginal_slope_formula_supports_linkwiggle_and_scorewarp`
- **FAIL** `python::tests/bench_large_scale_runner_test/MarkerContractTests` :: `test_every_parsed_marker_family_still_has_a_live_emission_site`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[2.0]`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[-2.0]`
- **FAIL** `python::tests/test_ci_no_orphan_python_tests_1512` :: `test_every_collectible_test_file_is_reached_by_some_ci_step`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_survival_transformation_is_reachable_from_fit`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_survival_at_accepts_array_like_times`
- **FAIL** `python::tests/test_survival_save_load_roundtrip` :: `test_survival_transformation_save_load_predict_roundtrips`

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

