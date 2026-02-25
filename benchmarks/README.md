# GAM Bench

This folder is the engine-level benchmark harness for `gam`.

It measures:
- Fit time
- Predict time
- AUC
- Brier score
- EDF (from fitted engine result)

## What it runs

- Rust engine contender: `gam` via `examples/bench_fit.rs`
- Optional external contenders (hook points): R `mgcv`, Python `pygam`

The parity scripts already moved into `/Users/user/gam/tests/parity`:
- `mgcv.py`
- `run_mgcv.R`
- `run_pygam.py`
- `compare.py`
- `bench.py`

## Quick start

```bash
cd /Users/user/gam
python3 benchmarks/run_suite.py
```

Output:
- `benchmarks/results.json`

## Scenario config

Edit:
- `/Users/user/gam/benchmarks/scenarios.json`

Default scenarios:
- `small_dense` (`n=1000, p=10`)
- `medium` (`n=50000, p=50`)
- `pathological_ill_conditioned` (`n=50000, p=80`)

## External contender hooks

To include placeholders for R/Python contenders:

```bash
python3 benchmarks/run_suite.py --with-external
```

The runner intentionally marks external entries as `skipped` until you wire local commands for your environment.

