# Getting started

## Installation

Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, Apple
silicon), and Windows. No Rust toolchain required for the prebuilt wheels.

```bash
uv add gamfit
```

Or as a one-off without a project:

```bash
uv pip install gamfit
```

`pip install gamfit` works too. The pre-built wheel ships with a vendored
Rust extension (`gamfit._rust`).

### Optional extras

```bash
uv add "gamfit[pandas]"     # pandas + pyarrow input/output
uv add "gamfit[plot]"       # matplotlib-based plotting
uv add "gamfit[sklearn]"    # scikit-learn integration
uv add "gamfit[all]"        # everything
```

Plain `gamfit` works without any of these. Adding `pandas`/`pyarrow` only
changes what `predict()` *returns* (and accepts) — the engine itself never
depends on them. Without `matplotlib`, `Model.plot()` and posterior trace
plots will raise. Without `scikit-learn`, `gamfit.sklearn` imports will fail.

### Verifying the install

```python
import gamfit
print(gamfit.__version__)
print(gamfit.build_info())
```

`build_info()` returns a dict including `available: True` when the Rust
extension loaded. If `available: False`, surface the diagnostic message with
`gamfit.explain_error(...)`.

## Your first model

```python
import gamfit

train = [
    {"y": 1.2, "x": 0.0},
    {"y": 1.9, "x": 1.0},
    {"y": 3.1, "x": 2.0},
    {"y": 4.5, "x": 3.0},
    {"y": 5.0, "x": 4.0},
]

model = gamfit.fit(train, "y ~ s(x)")
print(model)
```

Three things happened:

1. **Family inferred.** `y` is continuous, so the family is Gaussian and the
   default link is identity. (Override with `family=` or `link=`.)
2. **Smooth fit.** `s(x)` is a P-spline (cubic B-spline + difference penalty).
   Its complexity is chosen automatically — you don't pick `k` unless you want
   to.
3. **Smoothing parameter selected by REML.** No grid search, no manual
   tuning.

## Predict

```python
preds = model.predict([{"x": 1.5}, {"x": 2.5}])
```

Returns the linear predictor (`eta`) and the response-scale mean (`mean`),
in the same table format you passed in. For credible intervals:

```python
preds = model.predict([{"x": 1.5}, {"x": 2.5}], interval=0.95)
# Columns: eta, mean, effective_se, mean_lower, mean_upper
```

See [predictions.md](predictions.md) for `return_type`, `id_column`, and the
survival-specific `SurvivalPrediction` object.

## Inspect

```python
model.summary()                     # coefficients, deviance, REML score, etc.
model.diagnose(train).metrics       # n_obs, mae, rmse, bias, r_squared
model.check(test).ok                # schema check before predicting
model.plot(train, x="x")            # matplotlib (requires gamfit[plot])
model.report("out.html")            # standalone HTML report
```

See [diagnostics.md](diagnostics.md) for the full set.

## Persist

```python
model.save("model.gam")
loaded = gamfit.load("model.gam")
```

The `.gam` file is a binary blob that round-trips exactly. See
[persistence.md](persistence.md).

## Posterior sampling

Smoothing parameters are point estimates from REML, but you can draw from
the posterior of the coefficients conditional on those:

```python
posterior = model.sample(train, seed=42)
print(posterior)
# PosteriorSamples(n_draws=1024, n_coeffs=8, method='nuts',
#                  rhat=1.0042, ess=890.5, converged=True)

bands = posterior.predict(test, level=0.95)
```

NUTS is used where possible; a Gaussian Laplace fallback is used for model
classes that don't yet support exact NUTS (location-scale survival, latent
survival, transformation-normal, Bernoulli marginal-slope, link-wiggle). See
[posterior-sampling.md](posterior-sampling.md).

## Where to go next

- Real models have multiple smooths and constraints —
  [formulas.md](formulas.md) covers the full DSL.
- For classification, count, positive-continuous data, or non-default links —
  [families-and-links.md](families-and-links.md).
- For survival data, see [survival.md](survival.md) and (if you have a risk
  score) [marginal-slope.md](marginal-slope.md).
- For pandas / pipelines / cross-validation patterns —
  [sklearn.md](sklearn.md).
- For runnable end-to-end recipes — [cookbook.md](cookbook.md).
