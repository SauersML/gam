# gamfit

Formula-based generalized additive models for Python, backed by a Rust
engine. REML / LAML smoothing, NUTS or Gaussian Laplace posteriors,
survival, location-scale, and flexible links in one API.

`gamfit` fits Gaussian, binomial (including Bernoulli marginal-slope),
Poisson, and Gamma GLMs with smooth terms, random effects,
bounded/constrained coefficients, location-scale extensions, survival
likelihoods, and flexible/learnable links. Smoothing parameters are
selected by REML or LAML. Posterior sampling uses NUTS where supported,
and a Gaussian Laplace approximation otherwise.

This site is the Python documentation. The source repository is at
<https://github.com/SauersML/gam>; the PyPI page is at
<https://pypi.org/project/gamfit/>.

## Table of contents

Starting out

- [Getting started](getting-started.md) — install, fit a first model,
  understand the return value.
- [Data input formats](data-input.md) — pandas, polars, pyarrow, numpy,
  dict of columns, list of records.

Building a model

- [Formula DSL reference](formulas.md) — term types, options, links,
  families.
- [Difference smooths](difference-smooths.md) — pointwise and
  simultaneous contrasts between smooths.
- [Manifold smooths gallery](manifold-smooths.md) — periodic, cylinder,
  torus, sphere, Möbius double-cover, and boundary-conditioned smooths.
- [Families and link functions](families-and-links.md) — including
  `flexible(...)`, `blended(...)`, `sas`, `beta-logistic`.
- [Survival models](survival.md) — `Surv(...)` syntax, the four
  likelihood modes, parametric baselines, frailty.
- [Marginal-slope models](marginal-slope.md) — Bernoulli marginal-slope
  with a calibrated risk score.
- [Location-scale models](location-scale.md) — joint mean and variance.
- [Response geometry](response-geometry.md) — spherical and
  compositional responses via Fréchet-mean tangent-space GAMs.
- [Composition engine](composition_engine.md) — maintainer-facing map of
  the β / ψ / ρ engine, `LatentCoord`, analytic penalties, and SAE-manifold
  configurations.

Using a fitted model

- [Predictions](predictions.md) — `predict()`, intervals, `id_column`,
  `return_type`; `SurvivalPrediction` and chunked surface evaluation.
- [Posterior sampling](posterior-sampling.md) — NUTS, `SamplingConfig`,
  `PosteriorSamples`, `PosteriorPredictive`; convergence diagnostics.
- [Diagnostics, summary, plots, reports](diagnostics.md) — `summary()`,
  `diagnose()`, `check()`, `plot()`, `report()`.
- [scikit-learn integration](sklearn.md) — `GAMRegressor`,
  `GAMClassifier`, pipelines, cross-validation.
- [PyTorch integration](torch.md) — differentiable REML primitives,
  response-geometry transforms, frozen fitted-model modules.
- [Save and load](persistence.md) — `.gam` model files and `.npz`
  posteriors.

Reference

- [Full API reference](api-reference.md) — every public symbol.
- [Exceptions](exceptions.md) — exception hierarchy and
  `explain_error()`.
- [Cookbook](cookbook.md) — runnable recipes verified against the test
  suite.

## Example

```python
import gamfit
import pandas as pd

train = pd.DataFrame({
    "y": [1.2, 1.9, 3.1, 4.5, 5.2, 6.3, 7.1, 8.4],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "site": ["A", "A", "A", "B", "B", "B", "C", "C"],
})

# 1. Validate the formula against the data (no fit).
gamfit.validate_formula(train, "y ~ s(x) + group(site)")

# 2. Fit.
model = gamfit.fit(train, "y ~ s(x) + group(site)")

# 3. Predict with pointwise Wald intervals.
test = pd.DataFrame({"x": [1.5, 3.5], "site": ["A", "B"]})
print(model.predict(test, interval=0.95))

# 4. Inspect.
print(model.summary())
print(model.diagnose(train).metrics)

# 5. Posterior draws and bands.
posterior = model.sample(train, seed=42)
print(posterior)                              # convergence summary
print(posterior.predict(test, level=0.95))    # posterior mean bands

# 6. Persist.
model.save("model.gam")
posterior.save("posterior.npz")
```

## License

AGPL-3.0-or-later. See [LICENSE on GitHub](https://github.com/SauersML/gam/blob/main/LICENSE).

- [Difference smooths](difference-smooths.md)
