# Bernoulli marginal-slope FLEX biobank-shape harness

This note records the local feedback loop for optimizing the biobank-scale
bernoulli marginal-slope FLEX hang in `bench/biobank_scale/biobank_scale.yml`.
The goal is to reproduce the joint-Newton cycle-0 matrix-free Hv shape without
waiting for the scheduled 320K-row biobank lane.

## Ignored repro test

Run the 50K-row cycle-0 reproducer:

```bash
cargo test --release --test margslope_flex_biobank_repro -- --ignored --nocapture margslope_flex_biobank_repro_cycle0
```

Useful environment knobs:

- `GAM_MARGSLOPE_REPRO_N` (default `50000`) changes synthetic row count.
- `GAM_MARGSLOPE_REPRO_BOUND_SECS` (default `300`) controls the loose wall-time assertion.
- `GAM_MARGSLOPE_REPRO_PERSISTENT=1` adds a second run capped at five inner cycles.
- Profile with `samply`, `cargo flamegraph`, or macOS `sample`; `--nocapture`
  preserves the solver's per-cycle phase logs.

The synthetic fixture uses binomial/probit bernoulli marginal-slope, 16 PC
columns, a Duchon smooth for the PCs, an age smooth, score-warp FLEX, and
linkwiggle FLEX.  The fit is capped with `inner_max_cycles=1` and
`outer_max_iter=1`, so elapsed time is a local proxy for the production
`[PIRLS/blockwise joint-Newton] cycle 0/100` wall time.

## Beta-equivalence smoke

Run the reusable verification helper against two deterministic fits of the same
synthetic problem:

```bash
cargo test --release --test margslope_flex_biobank_repro -- --ignored --nocapture margslope_flex_beta_equivalence_smoke
```

Useful environment knobs:

- `GAM_MARGSLOPE_EQUIV_N` (default `2000`) changes smoke row count.
- `GAM_MARGSLOPE_EQUIV_INNER_CYCLES` (default `1`) changes the cycle cap.
- `GAM_MARGSLOPE_EQUIV_REL_TOL` (default `1e-10`) controls the relative beta tolerance.

The helper lives in `tests/test_support/margslope_flex_equivalence.rs` and emits
a clear PASS/FAIL message.  On failure it reports beta length, maximum absolute
and relative differences, and the worst-disagreeing beta index with both values.
Future optimization tasks should reuse this helper before claiming model
preservation.

## Opt-in Criterion benchmark

Run the Criterion wrapper only when intentionally collecting timing distributions:

```bash
GAM_RUN_MARGSLOPE_FLEX_BIOBANK_BENCH=1 GAM_MARGSLOPE_BENCH_N=50000 cargo bench --bench margslope_flex_biobank_hv
```

Without `GAM_RUN_MARGSLOPE_FLEX_BIOBANK_BENCH=1`, the benchmark records a tiny
skip/no-op so ordinary `cargo bench` does not accidentally launch a multi-minute
workload.
