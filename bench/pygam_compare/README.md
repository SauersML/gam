# gamfit vs pyGAM benchmark

This is the permanent, reproducible form of the ad-hoc harnesses from the pyGAM
audit (`bench/pygam_audit` on the audit branch; speed.md §2 "Harness" and the
accuracy harness). It compares speed, memory and held-out accuracy of `gamfit`
against pyGAM in two configurations:

| lib        | what it is |
|------------|------------|
| `gamfit`   | `gamfit.fit(data, "y ~ s(x0) + ...", family=...)`. Smoothing is selected by REML/LAML. |
| `pygam`    | pyGAM's default `.fit`, with a fixed lambda of 0.6 per term and no smoothing selection. |
| `pygam_gs` | pyGAM `.gridsearch(progress=False)` over lambda. This is the fair comparator: it is pyGAM *with* smoothing selection. |

The grid covers the families `gaussian`, `binomial` and `poisson` and the
designs `p1`, `p5` and `p20` (additive, `eta = sum_j sin(2 pi x_j + j)/sqrt(p)`)
plus `te` (`sin(2 pi x0) cos(2 pi x1)` fitted with `te(x0, x1)`). The data
generators are the audit's own, so the numbers stay comparable with speed.md.

## Setup

pyGAM is a **bench-only** dependency. It is never a runtime dependency of
gamfit and is never vendored.

```bash
maturin develop --release                          # or pip install a built wheel
pip install -r bench/pygam_compare/requirements.txt
```

## Running a plan

```bash
python -m bench.pygam_compare.run quick --out /tmp/pygam_quick
python -m bench.pygam_compare.report /tmp/pygam_quick --out /tmp/pygam_quick/report.md
```

`run` writes three files, including `report.md`, so the second command is only
needed to merge several runs. The outputs are:

- `records.jsonl`: one object per rep, including reps that failed or never ran;
- `meta.json`: host, nproc, RAM, library versions, git sha, the pinned thread
  env and the safety-net values;
- `report.md`: the rendered report.

These are the plans (see `plans.py`):

| plan        | cells | reps |
|-------------|-------|------|
| `smoke`     | n=300, all families × `p1` (the CI smoke test) | 1 |
| `quick`     | n=1e3, all families × all designs (the committed baseline) | 3 |
| `small_n`   | n ∈ {50, 200, 500}, all families × {`p1`, `p3`, `p5`} (fixed per-fit overhead) | 3 |
| `n1e4_core` | n=1e4, all families × {`p1`, `p5`, `te`} | 3 |
| `n1e5_core` | n=1e5, all families × {`p1`, `p5`, `te`} | 2 |
| `n1e6_memory` | n=1e6, {gaussian, poisson} × {`p1`, `p5`}: peak RSS and user/sys CPU | 1 |
| `full`      | n ∈ {1e3, 1e4, 1e5}, all families × all designs | 3 |
| `threads`   | gamfit only: n ∈ {1e4, 1e5, 1e6} × {gaussian, binomial} × {`p5`, `p20`, `te`} × threads {1, 2, 4, 8, auto} | 2 |
| `oversubscribe` | gamfit only: gaussian n=2e4 `te` and n=1e5 `p5`, alone and as one process per CPU at once, threads {1, auto} | 2 |

The last two measure parallelism rather than compare libraries. A cell's
`threads` sets every pool variable listed under **Threads** below (`auto`
unsets them all, so each pool sizes itself to the host); `concurrency` K runs K
identical processes at once, which is what `joblib` or `n_jobs=-1` does, and
records the batch wall time. The report then adds a thread-scaling table
(speedup over one thread) and a process fan-out table (throughput of the batch
against the same process run alone).

Overrides: `--reps`, `--timeout`, `--memcap-mb` and `--only-libs gamfit,pygam_gs`.

To regenerate the docs page from committed baselines:

```bash
python -m bench.pygam_compare.report bench/pygam_compare/baseline/quick --out docs/benchmarks.md
```

## What one rep measures

Every rep is a fresh subprocess (`worker.py`) whose working directory is a
scratch dir, so the installed wheel is imported instead of the source tree's
`./gamfit`. Within a cell the libraries are interleaved rep by rep, so drift in
host load hits all of them alike.

**Threads.** `RAYON_NUM_THREADS`, `MATMUL_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`NUMEXPR_NUM_THREADS` are all set to 1. The comparison is single-core against single-core. Only the
`threads` and `oversubscribe` cells change this.

**Time.** Each phase is timed as both wall time (`perf_counter`) and process
CPU time (`process_time`). The phases are import, one cold fit, a warm refit
of the same data in the same process (the per-fit cost once imports and lazy
initialisation are paid), point predict on n fresh rows, and a 95% interval
predict. CPU time is the primary metric, because wall time on a shared box also
measures the neighbours. The 1-minute load average is recorded at the start and
end of each rep.

**Memory.** The worker's own peak RSS comes from `ru_maxrss`. The driver also
polls the process-tree RSS and thread count with psutil every 50 ms.

**Accuracy.** Accuracy is measured on n held-out rows drawn with `seed + 1000`:

| metric | definition |
|--------|------------|
| `rmse_mu` | RMSE of the point prediction against the true mean |
| `deviance` | mean held-out unit deviance |
| `logscore` | mean negative log predictive density |
| `coverage` | fraction of true means inside the library's 95% interval for the mean |

For `logscore`, binomial and Poisson are scored at the predicted mean. Gaussian
uses the predictive sd implied by each library's own 95% prediction interval:
gamfit `observation_interval=True`, pyGAM `prediction_intervals`.

For `coverage`, gamfit's interval is `posterior_mean_lower/upper` and pyGAM's is
`confidence_intervals`.

## Verdicts

Every comparison is gamfit against a comparator, and lower is better for every
metric. Every loss is printed, and none is hidden or skipped.

- **Speed and memory.** The report takes the ratio of medians over the ok reps.
  A ratio above 1.00 is marked **LOSS**.
- **Accuracy.** Differences are paired by seed over the reps where both
  libraries are ok. A difference larger than 2 SE is a **LOSS** or a WIN.
  Anything smaller is "worse n.s." or "better n.s.". With a single seed there is
  no SE, so the sign alone decides, and the verdict says so.
- **Status.** If gamfit has fewer ok reps than the comparator in a cell, that is
  **LOSS(status)**. If the comparator reports a metric and gamfit does not, that
  is **LOSS(missing)**.

## Predictive-interval coverage

`conformal_coverage.py` is a separate scenario set. It measures how often each
method's 90% predictive interval (`alpha = 0.1`) covers a fresh response,
comparing pyGAM `prediction_intervals` with three gamfit routes: the posterior
observation interval, exact full conformal (`training_data=`) and split
conformal (`calibration=`). It covers six DGPs (correct, misspecified mean,
heteroscedastic, heavy tails, binomial, Poisson) at n ∈ {30, 100, 1000}.

```bash
python -m bench.pygam_compare.conformal_coverage --reps 1000
```

It writes `bench/pygam_audit/conformal_coverage.md`, the committed table.
The module docstring defines the DGPs, the width measure and the **nominal**
band, `0.9 - 2 MCSE <= coverage <= 0.9 + 1/(n_cal + 1) + 2 MCSE`. The report
ends with the cells where pyGAM misses that band and gamfit full conformal's
verdict in each.

## Safety net, not a budget

Each plan has a per-rep timeout, and there is a per-rep process-tree memory
cap, which defaults to half of total RAM. Both are a **harness safety net**:
they only stop one runaway rep from stalling the whole plan. Neither is a
solver budget, and nothing inside gamfit ever sees them.

When a rep hits a limit:

- The rep is recorded with status `timeout` or `memcap`.
- Its remaining reps are recorded as `not_run_after_<status>`, and so is every
  larger n of the same (lib, family, design).
- The report counts all of it as a loss for that library.

A rep that crashes or raises is recorded as `crash` or `error`, with the stderr
tail and the traceback of the failed phase.

## CI

- `bench/pygam_compare/test_pygam_compare_smoke.py` runs the `smoke` plan end to
  end and pins the verdict rules. It runs in `python-contracts.yml` (bench step).
- `.github/workflows/pygam-compare.yml` is optional. It runs on manual dispatch
  (with a `plan` input, default `quick`) and weekly, never per PR. It uploads
  `records.jsonl`, `meta.json` and `report.md` as an artifact and writes the
  report to the job summary.

## Baseline

`baseline/quick/` holds the committed output of the `quick` plan: the records,
the meta and the report. Regenerate it with

```bash
python -m bench.pygam_compare.run quick --out bench/pygam_compare/baseline/quick
```

and commit all three files together, with the pyGAM pin from
`requirements.txt` unchanged or bumped in the same commit.
