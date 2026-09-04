# MASTER_FAILURES

- Compile failures: **0**
- Workspace tests run: **NOT MEASURED** (required floor: 558)
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (0 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 0** (LOWER BOUND, not a count: Python API tests (job `failure`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `failure`, matrix `skipped`, workspace shard set differs from 1..10: missing=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], extra=[])
- gam-pyffi unit tests: **MEASURED** (job `success`)
- Python API tests: **NOT MEASURED** (job `failure`)
- Python populations (slow + torch): **MEASURED** (job `failure`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python API tests (job `failure`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the Python surface was NOT measured — the Python job went red without recording a single failing test — a step before pytest (wheel build, CLI integration script) most likely died; read the job log. The Python counter above is not a result.

> NOTE: the runtime surface was NOT measured — the archive build reported `failure`; the shard matrix reported `skipped`; only 0 of 10 planned shard logs were collected; no workspace shard log was collected at all; the workspace test population was not certified: workspace shard set differs from 1..10: missing=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], extra=[]; a shard reported ARCHIVE_MISSING. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.
>
> The archive is missing and NO compile error was captured either, so this run reports nothing at all about the workspace — neither that it builds nor that it passes. Read the build-logs artifact.

## Compile failures

_None._

## Runtime test failures

_None._

## Python test failures

_Lower bound: 0 recorded before the run stopped. Unmeasured: Python API tests (job `failure`)._

_Not measured — see the note above._

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

