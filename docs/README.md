<div class="gamfit-hero" markdown>

# gamfit { .gamfit-hero-title }

<p class="gamfit-tagline" markdown>
Formula-first generalized additive models for Python, backed by a Rust
engine. REML/LAML smoothing, NUTS posteriors, survival, location-scale,
and flexible links in one API.
</p>

<div class="gamfit-cta" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[API reference](api-reference.md){ .md-button }
[GitHub](https://github.com/SauersML/gam){ .md-button }
</div>

</div>

`gamfit` fits Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms,
random effects, bounded/constrained coefficients, location-scale extensions,
survival likelihoods, and flexible/learnable links. Smoothing parameters are
selected by REML or LAML. Posterior sampling uses NUTS.

This site is the Python documentation. For a short overview, see the
[PyPI page](https://pypi.org/project/gamfit/).

## Table of contents

**Starting out**

- [Getting started](getting-started.md) — install, fit your first model,
  understand the return value.
- [Data input formats](data-input.md) — pandas, polars, pyarrow, numpy, dict
  of columns, list of records. How `gamfit` decides what to return.

**Building a model**

- [Formula DSL reference](formulas.md) — every term type, every option, every
  link, every family.
- [Manifold smooths gallery](manifold-smooths.md) — visual tour of the
  periodic, cylinder, torus, sphere, Möbius, and boundary-conditioned
  smooths recovered from noisy 3-D point clouds.
- [Families and link functions](families-and-links.md) — when to use which,
  including `flexible(...)`, `blended(...)`, `sas`, `beta-logistic`.
- [Survival models](survival.md) — `Surv(...)` syntax, the four likelihood
  modes, parametric baselines, frailty.
- [Marginal-slope models](marginal-slope.md) — Bernoulli marginal-slope with
  a calibrated risk score; the two-stage transformation-normal pipeline.
- [Location-scale models](location-scale.md) — jointly modelling mean and
  variance.
- [Response geometry](response-geometry.md) — spherical and compositional
  responses via Fréchet-mean tangent-space GAMs.

**Using a fitted model**

- [Predictions](predictions.md) — `predict()`, intervals, `id_column`,
  `return_type`; the `SurvivalPrediction` object and chunked surface
  evaluation.
- [Posterior sampling](posterior-sampling.md) — NUTS, `SamplingConfig`,
  `PosteriorSamples`, `PosteriorPredictive`; convergence diagnostics.
- [Diagnostics, summary, plots, reports](diagnostics.md) — `summary()`,
  `diagnose()`, `check()`, `plot()`, `report()`.
- [scikit-learn integration](sklearn.md) — `GAMRegressor`, `GAMClassifier`,
  pipelines, cross-validation.
- [Save and load](persistence.md) — `.gam` model files and `.npz` posteriors.

**Reference**

- [Full API reference](api-reference.md) — every public symbol in one place.
- [Exceptions](exceptions.md) — the exception hierarchy and `explain_error()`.
- [Cookbook](cookbook.md) — runnable recipes, verified against the test
  suite.

## A complete tour in 30 lines

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

# 3. Predict with credible intervals.
test = pd.DataFrame({"x": [1.5, 3.5], "site": ["A", "B"]})
print(model.predict(test, interval=0.95))

# 4. Inspect.
print(model.summary())
print(model.diagnose(train).metrics)

# 5. Posterior draws + bands.
posterior = model.sample(train, seed=42)
print(posterior)                              # convergence summary
print(posterior.predict(test, level=0.95))    # posterior predictive bands

# 6. Persist.
model.save("model.gam")
posterior.save("posterior.npz")
```

## License

AGPL-3.0-or-later. See [LICENSE on GitHub](https://github.com/SauersML/gam/blob/main/LICENSE).
