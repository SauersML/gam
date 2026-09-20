# Convergence fuzzer for additive default fits

A seeded fuzzer that asks one question of the engine: **does the default fit
of an ordinary additive model certify?** It draws random additive
data-generating processes, fits `y ~ s(x0) + ... + s(x{p-1})` with every
default left alone, and records every way a fit can fall short.

```sh
cd bench
python -m convergence_fuzz.run full  --out /tmp/fuzz/after  --jobs 3
python -m convergence_fuzz.run quick --out /tmp/fuzz/quick
python -m convergence_fuzz.triage /tmp/fuzz/after                  # report by cause
python -m convergence_fuzz.triage /tmp/fuzz/before /tmp/fuzz/after # before/after
python -m pytest convergence_fuzz/test_quick.py                    # the regression gate
```

The installed `gamfit` wheel is what gets fuzzed (`maturin develop --release`
first); each worker runs with only `bench/` on `PYTHONPATH`, so the source
tree's `./gamfit` is never imported by accident.

## The DGP space (`dgp.py`)

A *case* is an integer; `numpy.random.default_rng((ROOT_SEED, case))` fixes
everything about its DGP except the family and `n`:

| axis | values |
|---|---|
| covariates `p` | 1 .. 8 |
| covariate distribution | uniform, skewed (log-normal), heavy-tailed (Student t, df 1-3), clustered (2-5 tight clusters), discrete/tied (2-20 levels), x-outliers (0.5-3% of rows 20-100 sd out) |
| covariate location / scale | `N(0, 10)` / log-uniform over `1e-2 .. 1e2` |
| true shape per covariate | linear, sinusoid, step (1-3 jumps), spiky (narrow bump), flat |
| family | gaussian (identity), binomial (logit), poisson (log) |
| `n` | 30, 100, 1 000, 10 000 |

Each shape is evaluated on the covariate's robust unit scale and scaled so the
linear predictor stays bounded on the training hull (a heavy-tailed covariate
must not overflow the Poisson mean in the truth itself). Held-out rows see the
truth held at its boundary value outside the training range.

The `full` plan is 180 cases x `n` in {30, 100, 1 000} x 3 families, plus
the first 24 cases (which draw every `p` from 1 to 8) at `n = 10 000`:
**1 692 reps, 3 384 fits**. A rep at `n = 10 000` costs one to fifteen
single-threaded minutes, about a hundred times one at `n = 1 000`.
`quick` is the seeded fixture of every root cause this fuzzer found and fixed
(`run.FIXTURES`), and `test_quick.py` requires zero failures on it. A cause
that is still open, in this lane or another, shows up in the `full` report
and gets its fixture in `quick` in the same change that fixes it.

## What one rep checks (`worker.py`)

Each rep is one subprocess, launched through
`pygam_compare.run.run_isolated` (pinned single-thread pools, process-tree
RSS polling, scratch cwd). It runs, recording every exception:

1. `fit` - `gamfit.fit(train, formula, family=...)`;
2. `summary` - certificate, REML/LAML score, EDF, lambdas;
3. `predict` - posterior-mean prediction on the held-out rows;
4. `interval` - 95% interval on 500 held-out rows;
5. `refit` - the same model on a row permutation of the same data with every
   covariate under a random positive affine map `x -> a x + b`.

## Failure causes (`triage.py`)

A rep fails on the first of, in order: `hang` / `memcap` / `crash` (the
harness safety net), `raise:<phase>:<Type>: <message head>` (digits folded to
`#` so one cause clusters), `uncertified:<fit|refit>:<kind>` (a fit object
whose certificate does not say certified), `nonfinite:<what>`, and
`reml_mismatch:<fit|refit>_worse`.

The timeout (900 s) and memory cap are a safety net for the harness, never a
solver budget: a rep that trips either is recorded as a failure and never
retried or excused.

### REML comparison

The trusted comparison for a certified fit's REML/LAML score is the same
model's refit. A B-spline basis on the data range with data-driven knots is
equivariant under row permutation and a positive affine map of each
covariate, and a derivative penalty only rescales under that map, so both
problems have the same optimum and the same criterion value there. Two
certified optima of one problem agree to the certificate's own stationarity
accuracy; the gate is `|V_fit - V_refit| > 1e-6 + 1e-6 max(|V_fit|, |V_refit|)`,
and the worse of the two is the one whose search stopped short (or whose
problem has more than one optimum, which is a finding of its own).

## Results

`triage.py DIR_BEFORE DIR_AFTER` regenerates the before/after table from two
runs. The full plan on main at 99940493 (before) and with the latched
spectral-position block of the #784 correction (after), by primary cause,
message heads abbreviated:

| cause | before | after |
|---|---:|---:|
| `raise:fit` outer optimization did not certify a stationary optimum | 157 | 147 |
| `raise:fit` smooth term remains under-resolution-uncertain | 76 | 76 |
| `raise:fit` #784 block-local correction: order search refused | 32 | 32 |
| `hang` | 26 | 22 |
| `raise:refit` outer optimization did not certify a stationary optimum | 19 | 24 |
| `raise:fit` declined a certified optimum that an evaluated state beats | 8 | 8 |
| `raise:fit` Newton decrement above tolerance | 5 | 5 |
| `reml_mismatch:refit_worse` | 3 | 3 |
| `reml_mismatch:fit_worse` | 1 | 2 |
| `raise:fit` / `raise:refit` under-resolution-uncertain (other types) | 2 | 4 |
| `nonfinite:*_reml_score` exactly interpolating Gaussian fit | 2 | 2 |
| **total** | **331 / 1692 (19.56%)** | **325 / 1692 (19.21%)** |

The fixed cause's fixture is `run.FIXTURES` (`case0/binomial/n1000`). The
rest of the net change is outer-search trajectories moving, both ways, once
the criterion no longer jumps; those failures belong to the causes above.
