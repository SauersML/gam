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

The binomial sweep adds three binomial variants. `binomial_p10` and
`binomial_p01` shift the intercept so the prevalence is 0.1 and 0.01 (slope 1.5
on the same `eta`). `binomial_trials` draws 1 to 20 trials per row and fits the
observed proportion with the trial counts as prior weights, through `weights=`
in both libraries; its deviance and log score are trial-weighted.

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
| `postfit`   | n ∈ {1e3, 1e5} × n_predict ∈ {1e2, 1e4, 1e6}, all families × {`p5`, `p20`, `te`}, `gamfit` and `pygam_gs` only, with the post-fit phases | 1 |
| `gaussian_small` | n ∈ {1e2, 1e3, 1e4}, gaussian × {`p1`, `p5`, `p20`, `te`, `te+s`, `by`} (the nightly Gaussian regression cells) | 3 |
| `gaussian_1e5` | n=1e5, gaussian × {`p1`, `p5`, `p20`, `te`, `te+s`, `by`} | 3 |
| `gaussian_1e6` | n=1e6, gaussian × {`p1`, `p5`, `p20`, `te`, `te+s`, `by`}: wall, CPU and peak RSS at the largest scale | 3 |
| `binomial_small` | n ∈ {1e2, 1e3}, {`binomial`, `binomial_p10`, `binomial_p01`, `binomial_trials`} × all designs | 3 |
| `binomial_1e4` | n=1e4, the four binomial variants × all designs | 2 |
| `binomial_1e5` | n=1e5, the four binomial variants × all designs | 1 |
| `positive_small` | n ∈ {1e2, 1e3}, positive-response families × all designs | 3 |
| `positive_1e4` | n=1e4, positive-response families × all designs | 2 |
| `positive_1e5` | n=1e5, positive-response families × {`p1`, `p5`, `te`} | 1 |
| `count_small` | n ∈ {1e2, 1e3}, count families × all designs | 3 |
| `count_1e4` | n=1e4, count families × all designs | 2 |
| `count_1e5` | n=1e5, count families × {`p1`, `p5`, `te`} | 1 |
| `fuzz_families` | gamfit only: n ∈ {50, 500, 5000}, every family/link label × every support-edge regime (convergence fuzz, below) | 3 |
| `fuzz_families_quick` | gamfit only: n ∈ {50, 500}, every family/link label × {base, edge, zeros, lowdisp} (the 0-failure regression test) | 1 |
| `threads`   | gamfit only: n ∈ {1e4, 1e5, 1e6} × {gaussian, binomial} × {`p5`, `p20`, `te`} × threads {1, 2, 4, 8, auto} | 2 |
| `oversubscribe` | gamfit only: gaussian n=2e4 `te` and n=1e5 `p5`, alone and as one process per CPU at once, threads {1, auto} | 2 |
| `fuzz_terms` | gamfit only: 120 seeded term-structure cases × n ∈ {50, 500, 5000} × all families (1080 fits) | 1 |
| `fuzz_terms_quick` | gamfit only: the fixed cases in `FUZZ_QUICK_CASES`, which cover every term kind, × n ∈ {50, 500} × all families (a 0-failure regression test) | 1 |

The positive-response families are Gamma on the log link (`gamma_log`, shape 3),
heavy right skew with responses near zero (`gamma_skew`, shape 0.5), Gamma on the
inverse link (`gamma_inverse`), the inverse Gaussian (`inverse_gaussian`),
log-normal data fitted as a Gaussian on log y (`lognormal_gaussian`) and as a
Gamma on y (`lognormal_gamma`), and scaled-t noise with 3 degrees of freedom
(`student_t`). pyGAM is the comparator for the Gamma and log-normal families.
It has no scaled-t family, and its inverse Gaussian stores sqrt(phi) as its
scale, so `inverse_gaussian` and `student_t` run gamfit alone and report
absolute numbers.

The count plans are the count-family sweep. They use their own families:
`poisson_lo`, `poisson_mid` and `poisson_hi` (Poisson with mean level 0.3, 5
and 500), `poisson_exposure` (Poisson with a log-exposure offset), `negbin`
(negative binomial, theta estimated) and `tweedie` (power fixed at 1.5, phi
estimated). pyGAM has no negative binomial or Tweedie family, so those cells
run gamfit alone and report absolute time and the certification rate (the
`k/n ok` count in the Status table; a fit that returns has certified). For
`poisson_exposure`, pyGAM is fitted on the rate `y/E` with weights `E`, which
is the exposure likelihood; `PoissonGAM.gridsearch` in pyGAM 0.12.0 passes the
weights on as the exposure a second time.

The `fuzz_families` plans are a convergence fuzz, not a comparison
(`fuzz_families.py`). A cell is one family/link label, n and an `ff-<regime>`
design. The labels cover every response family in gamfit's family registry
with every link its legality table admits: Gamma (log, inverse), the inverse
Gaussian (canonical `1/μ²`, log), the negative binomial, Tweedie at p = 1.2, 1.5 and
1.8, beta, scaled t, Poisson, binomial with trials (every binomial link) and the
Gaussian on its inverse link. gamfit has no quasi families. The regime places
the data at an edge of the family's support: responses at the boundary, a mean
spanning orders of magnitude, a region with no events, near-degenerate or
extreme dispersion, no signal, or extreme units (the module docstring lists
them). Every fit is `y ~ s(x0) + s(x1)`.

`failure_cause` classifies each rep. A rep fails when it hung (hit the safety
net), crashed, raised, did not certify its optimum, predicted a non-finite point
or interval, or reported a scale more than `SCALE_Z_MAX` = 10 standard errors
from the truth. The standard error is that of the oracle Pearson estimate at the
true mean, recomputed from the rep's seeded draw. The scale is not judged at
`lowdisp`, where the basis's approximation error is as large as the noise, so a
correct fit's scale carries it. A binomial draw with no events at all has a
constant response, and a typed refusal of it is correct, not a failure. The
triage table (causes by count, with an example rep for each) prints with:

```bash
python -m pygam_compare.fuzz_families RUN_DIR [RUN_DIR ...]
```

`test_fuzz_families_quick.py` runs the quick plan and requires zero failures;
`test_fuzz_families_fixtures.py` pins the cells whose root causes were fixed.

The `threads` and `oversubscribe` plans measure parallelism rather than compare
libraries. A cell's `threads` sets every pool variable listed under **Threads**
below (`auto` unsets them all, so each pool sizes itself to the host);
`concurrency` K runs K identical processes at once, which is what `joblib` or
`n_jobs=-1` does, and records the batch wall time. The report then adds a
thread-scaling table (speedup over one thread) and a process fan-out table
(throughput of the batch against the same process run alone).

The workflow `.github/workflows/pygam-compare.yml` runs `quick` weekly and
`gaussian_small` nightly; any plan can be dispatched by name.

Overrides: `--reps`, `--timeout`, `--memcap-mb`, `--only-libs gamfit,pygam_gs`,
`--designs d1,d2` (every n and family of just those designs, e.g. to re-run
the designs a generator change touched) and `--shard I/K`, which runs every
K-th design starting at design I, so K shards started side by side cover the
plan between them. `--lib-path DIR` puts a pinned library build first on every
worker's `PYTHONPATH` (workers never inherit the caller's), so a before/after
comparison measures the build it names; every gamfit record carries `lib_file`.

### Convergence fuzz over term structure

The `fuzz_terms*` plans draw their formulas from `fuzz_terms.py`. A case number
fixes the term structure: tensor products with two or three margins, `ti`,
factor and numeric `by=` smooths (including empty and singleton levels), fixed
factors with rare levels, random intercepts with 5 to 2000 levels, cyclic,
2-D isotropic, shape-constrained and concurvity terms. The seed draws the data.
To triage one or more run directories by failure cause, term kind, family and n:

```bash
python -m pygam_compare.fuzz_terms RUN_DIR [RUN_DIR ...]
```

A rep counts as a failure if it raised, hung, did not certify its optimum or
predicted a non-finite value. For each cause the table names one example rep as
`FAMILY N DESIGN SEED`, so `python bench/pygam_compare/worker.py gamfit FAMILY N
DESIGN SEED` reruns it in isolation.

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
initialisation are paid), point predict on n fresh rows (or `n_predict` rows
when the cell sets it), and a 95% interval predict. Plans with `postfit` also
time a term's partial dependence on a 200-point grid, the summary, a save/load
round trip (`save_bytes` records the file size), 100 posterior coefficient
draws and gamfit's smooth significance. CPU time is the primary metric, because
wall time on a shared box also measures the neighbours. The 1-minute load
average is recorded at the start and end of each rep.

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
- **Coverage.** Calibration is judged on each library's mean coverage over the
  paired seeds, `|mean cov - 0.95|`, not on per-seed `|cov - 0.95|`. One fit's
  intervals move together with that fit's error, so even an exactly calibrated
  interval scatters per seed, and a per-seed score would rank an interval that
  over-covers every seed above it. Over-coverage is miscalibration, exactly
  like under-coverage. Seed `s` contributes
  `side_g (g_s - 0.95) - side_c (c_s - 0.95)`, with each side fixed at
  `sign(mean - 0.95)`. These terms average to the mean-level difference, and
  their spread gives its SE for the same 2 SE rule.
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

It writes the table to `bench/pygam_audit/conformal_coverage.md`.
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
- `bench/pygam_compare/test_fuzz_terms_quick.py` runs the `fuzz_terms_quick`
  plan and requires every fit to be clean. It also checks that the quick cases
  cover every term kind.
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
