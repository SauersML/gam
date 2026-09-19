# p-value calibration harness

This harness checks whether gamfit's p-values hold their nominal size. It
draws a seeded grid of datasets, fits each one, and reads the p-value of one
tested term. It does this under the term's null hypothesis and under a matched
local alternative, and fits pyGAM to the same data wherever pyGAM has a
counterpart. [p-values and their calibration](../../docs/pvalues.md) explains
what each p-value tests and shows the table this harness generates.

A cell is `(family, n, null)`:

| axis | values |
|------|--------|
| family | `gaussian`, `binomial`, `poisson`, `gamma`, `negbin` |
| n | 60, 200, 1000, 5000 |
| null | `smooth`, `linear`, `factor`, `ti`, `re` and `concurvity` (below) |

Each null structure is one formula plus one tested term. Every dataset has a
real nuisance smooth `sin(2 pi x1)`, and the tested term `t` is scaled to unit
variance.

| null | formula | tested term |
|------|---------|-------------|
| `smooth` | `y ~ s(x1) + s(x2)` | `s(x2)`, `t = sqrt 2 cos(2 pi x2)` |
| `linear` | `y ~ s(x1) + x2` | the `x2` coefficient |
| `factor` | `y ~ s(x1) + g` | the 3-level factor `g` |
| `ti` | `y ~ s(x1) + s(x2) + ti(x1, x2)` | `ti(x1, x2)`, beside real main effects |
| `re` | `y ~ s(x1) + group(g)` | the random intercept, one level per 10 rows |
| `concurvity` | `y ~ s(x1) + s(x2)` | `s(x2)`, with latent correlation 0.9 between `x1` and `x2` |

## Setup

This harness uses the pyGAM pin of `bench/pygam_compare`. pyGAM is a
**bench-only** dependency.

```bash
maturin develop --release                          # or pip install a built wheel
pip install -r bench/pygam_compare/requirements.txt
```

## Running a plan

```bash
python -m bench.pvalue_calibration.run quick --out /tmp/pv_quick
python -m bench.pvalue_calibration.report /tmp/pv_quick --out /tmp/pv_quick/report.md
```

`run` writes three files, including `report.md`, so the second command is only
needed to merge several runs. The outputs are:

- `records.jsonl`: one object per (cell, seed), including reps whose chunk timed
  out, exceeded the memory cap or crashed;
- `meta.json`: host, nproc, RAM, library versions, git sha, the pinned thread
  env, the safety-net values and one entry per invocation;
- `report.md`: the calibration table, followed by every unusable rep and why.

These are the plans (see `plans.py`):

| plan | cells | reps |
|------|-------|------|
| `ci` | gaussian and poisson `smooth`, gaussian `linear`, at n=200, gamfit only (the CI regression test) | 200 |
| `quick` | n=200, every family × every null except `ti`, plus gaussian `ti` (the committed baseline) | 100 |
| `nightly` | the full grid: every family × every n × every null | 500 |

`quick` runs one `ti` cell instead of five for a cost reason.
`smooth_significance` on an `s + s + ti` model takes minutes per fit, about ten
times any other cell, and each rep calls it twice.

Overrides: `--reps`, `--jobs`, `--timeout`, `--memcap-mb` and
`--only-libs gamfit`.

**Resumable.** Re-running into the same `--out` skips every (cell, seed)
already in `records.jsonl`. An interrupted run picks up where it stopped, and a
larger `--reps` runs only the new seeds.

**Parallel.** `--jobs` worker processes run at once, one per CPU by default.
Each worker runs one chunk of seeds of one cell.

**Memory-bounded.** Each worker's process tree is capped at `--memcap-mb`. By
default the cap is half of total RAM, split across the jobs. The driver streams
records to disk as they arrive.

## What one rep measures

The worker subprocess runner is `bench/pygam_compare`'s `police`. It gives the
same pinned one-thread env (`RAYON_NUM_THREADS=1` and the BLAS equivalents),
the same process-tree RSS and wall-time polling, and the same scratch working
directory, so the installed wheel is imported instead of the source tree's
`./gamfit`.

A rep draws two datasets from its seed. One is under the null (`delta = 0`),
and the other is under the matched alternative, drawn from an independent
stream. Every library in the plan is fitted to both.

The alternative is `delta = sqrt(9 / (n I))`, where `I` is the per-row Fisher
information for the linear predictor at the baseline mean. A one-degree-of-freedom
test of the unit-variance term then has noncentrality 9 (a 3-sigma effect) at
every family and n, so power compares like with like.

These are the surfaces a rep reads:

| surface | source |
|---------|--------|
| `gamfit.wald` | `summary().smooth_terms[...]["p_value"]` |
| `gamfit.lr` | `smooth_significance(data)[...]["p_value_corrected"]` |
| `gamfit.coef` | `summary().coefficients`. The row is located through `model.term_blocks`, and the p-value is the two-sided normal tail of `estimate / std_error` (gamfit reports no coefficient p-value). |
| `pygam.wald` | pyGAM's `statistics_["p_values"]` at its default fixed lambda |
| `pygam_gs.wald` | the same, after `gridsearch` |

pyGAM has no `ti`, no random effect, no negative binomial and no LR test, so
those surfaces are simply not expected. When a gamfit surface is expected but
gives no p-value (a `None`, or no row for the term), the rep records it under
`missing` with the reason. The report counts that rep as unusable for the
surface and lists it. It is never dropped.

## Verdicts

For each (cell, surface), the report gives:

- size at 0.10, 0.05 and 0.01 with its MCSE, `sqrt(size (1 − size) / R)`;
- the Kolmogorov–Smirnov distance of the null p-values from uniform;
- power at 0.05 under the matched alternative.

A p-value is calibrated when it is Uniform(0, 1) under its null. A
conservative p-value fails that just as an anti-conservative one does, so
every check is two-sided. With `m` the number of checks in the report (two
size checks per level and one KS check, per row), a row is:

- **ANTI-CONSERVATIVE** at `a` when its rejection count exceeds
  `reject_bound(R, a, m)`, the `1 − 10⁻³ / m` quantile of Binomial(R, a);
- **CONSERVATIVE** at `a` when its rejection count falls below
  `reject_floor(R, a, m)`, the `10⁻³ / m` quantile of the same law;
- **NOT UNIFORM** when the KS test of its null p-values against Uniform(0, 1),
  over the whole range, has p-value at or below `10⁻³ / m`. This catches
  shapes the three levels miss, such as a point mass at 1.

A calibrated p-value trips any check with family-wise probability at most
10⁻³. The tolerances come from each statistic's own sampling law, so they
are not hand-picked, and they scale with R. At R = 100 and `a` = 0.01 a
calibrated p-value rejects zero times with probability 0.37, so the lower
size check cannot fire there; the KS check covers that level.

A rep with no p-value (a fit that raised, a missing row, or a seed the safety
net killed) takes whichever value is worst for each check: a rejection in the
upper size check, a non-rejection in the lower one, and for KS all such reps
at 0 or all at 1, whichever gives the larger distance. Such a rep may have
been anything, and a size taken only over the reps that succeeded is biased
whenever failing correlates with extreme data. A row therefore passes only if
it passes in that worst case, while its size columns are over the usable reps.
A row whose null reps all lack a p-value is **NO P-VALUE**.

## Safety net, not a budget

Each plan has a per-chunk timeout, and each worker has a process-tree memory
cap. Both are a **harness safety net**: they only stop one runaway chunk from
stalling the whole plan. Neither is a solver budget, and nothing inside gamfit
ever sees them.

Each worker prints each rep as soon as it finishes, so a chunk the safety net
kills still keeps its finished reps. The driver then starts a continuation
chunk at the rep the worker died on, so the reps queued behind a runaway one
are run rather than blamed for it. A rep is charged only when it dies as the
first rep of its worker, with the whole budget to itself. It is then recorded
with status `timeout`, `memcap` or `crash` and the stderr tail, and the report
counts it as a rejection.

## CI

`test_pvalue_calibration_smoke.py` runs in `python-contracts.yml` (the bench
step). It contains:

- `test_ci_plan_is_calibrated`, which runs the `ci` plan end to end. It fails
  when any gamfit surface is flagged anti-conservative, conservative or not
  uniform (with every rep without a p-value placed worst for each check), when
  a rep that fitted lacks an expected surface, or when a row lacks a power;
- tests that pin the harness's own rules on hand-built records or a stand-in
  worker: resume, safety-net records (only a rep that had the budget to itself
  is charged, and the seeds after it run), seeding, both verdict bounds, that
  a point mass at 1, a size below nominal and a non-uniform shape above the
  levels are each flagged, and unusable-rep accounting;
- a test that `docs/pvalues.md`'s table equals the one generated from the
  committed baseline.

## Baseline

`baseline/quick/` holds the committed output of the `quick` plan. Regenerate
it and the docs table with:

```bash
python -m bench.pvalue_calibration.run quick --out bench/pvalue_calibration/baseline/quick
python -m bench.pvalue_calibration.report bench/pvalue_calibration/baseline/quick --docs docs/pvalues.md
```

Commit the records, the meta, the report and the docs page together.
