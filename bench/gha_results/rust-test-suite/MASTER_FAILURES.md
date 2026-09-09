# MASTER_FAILURES

- Compile failures: **0**
- Workspace tests run: **NOT MEASURED** (9676 runnable in the archive listing)
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (5 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 0** (LOWER BOUND, not a count: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `success`, matrix `failure`, workspace shard set differs from 1..10: missing=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], extra=[])
- gam-pyffi unit tests: **MEASURED** (job `failure`)
- Python API tests: **NOT MEASURED** (job `cancelled`)
- Python populations (slow + torch): **NOT MEASURED** (job `cancelled`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the Python surface was NOT measured — the Python job reported `cancelled`. The Python counter above is not a result.

> NOTE: the runtime surface was NOT measured — only 0 of 10 planned shard logs were collected; no workspace shard log was collected at all; the workspace test population was not certified: workspace shard set differs from 1..10: missing=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], extra=[]. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.
>
> The archive is missing and NO compile error was captured either, so this run reports nothing at all about the workspace — neither that it builds nor that it passes. Read the build-logs artifact.

## Compile failures

_None._

## Runtime test failures

- **FAIL** `gam-pyffi` :: `batch_tests::circle_latent_recovers_circle_not_collapse`
- **FAIL** `gam-pyffi` :: `batch_tests::gaussian_reml_batched_matches_single_fit_on_ragged_offsets`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::ring_of_clusters_owns_discrete_cyclic_verdict_2262`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_fit_is_output_rotation_equivariant`

## Python test failures

_Lower bound: 0 recorded before the run stopped. Unmeasured: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`)._

_Not measured — see the note above._

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

