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

The `full` plan is 180 cases x 4 `n` x 3 families = **2 160 reps, 4 320 fits**.
`quick` is the seeded fixture of every root cause this fuzzer found and fixed
(`run.FIXTURES`) plus the first six cases at `n` in {30, 100}.

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

See the PR that introduced this directory for the before/after table by
cause; `triage.py DIR_BEFORE DIR_AFTER` regenerates it from two runs.
