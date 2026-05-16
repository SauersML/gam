# gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/SauersML/gam/blob/main/LICENSE)

Formula-first generalized additive models for Python, backed by a Rust
engine.

`gamfit` fits Gaussian, binomial, Poisson, and Gamma GLMs with smooth
terms, random effects, bounded/constrained coefficients, location-scale
extensions, survival likelihoods, and flexible/learnable links. Smoothing
parameters are selected by REML or LAML. Posterior sampling uses NUTS
where supported, with a Gaussian Laplace fallback elsewhere.

Geometric / manifold smooths handle predictor spaces that wrap, including
circles, cylinders, tori, the sphere (intrinsic Wahba / spherical
harmonic kernels), and one-sided strips, with no seams or pole
artefacts:

![rotating recovery of a trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere, bumpy torus, and Möbius strip from noisy 3-D point clouds](https://raw.githubusercontent.com/SauersML/gam/main/docs/images/geometric_shapes_demo.gif)

**Docs:** <https://gamfit.readthedocs.io/>

## Install

```bash
uv add gamfit
```

Wheels for Linux (x86_64, aarch64), macOS (x86_64, Apple silicon), and
Windows. No Rust toolchain required.

## 30-second tour

```python
import gamfit

train = [
    {"y": 1.2, "x": 0.0},
    {"y": 1.9, "x": 1.0},
    {"y": 3.1, "x": 2.0},
    {"y": 4.5, "x": 3.0},
]

model = gamfit.fit(train, "y ~ s(x)")
print(model.predict([{"x": 1.5}, {"x": 2.5}], interval=0.95))
print(model.summary())
model.save("model.gam")
```

pandas, polars, pyarrow, numpy, dict-of-columns, and list-of-records inputs
all work without conversion.

## What's different

- **Three-part penalty structure.** Each smooth gets separate penalties for
  magnitude, gradient, and curvature. Most GAM libraries use one or two.
- **Flexible link functions.** Spline offsets from a base link
  (`link(type=flexible(probit))`), blended mixture links, and SAS /
  beta-logistic learnable shapes.
- **Surface smooths in arbitrary dimension.** Thin-plate, Duchon (with
  triple-operator regularization), and Matérn covariance, with automatic
  knot placement.
- **Geometric / manifold smooths.** Cyclic 1-D, cylinder / torus tensor,
  intrinsic sphere (Wahba + spherical harmonics), Möbius strip,
  boundary-conditioned B-splines. Predictor spaces that wrap or close
  are first-class — no seams, no pole artefacts.
- **Adaptive anisotropy.** Per-axis spatial anisotropy shrinks or stretches
  each feature axis independently inside a single joint smooth.
- **Composable basis/kernel.** Combine a spline kernel with a
  length-scale behaviour (e.g. Duchon kernel with Matérn-style global κ).
- **Marginal-slope models.** Separate baseline risk from a calibrated
  score's effect, for both Bernoulli and survival outcomes.
- **Posterior sampling.** `model.sample(...)` runs NUTS where supported,
  with a Gaussian Laplace fallback elsewhere, behind one API.

## Highlights from the API

```python
import gamfit
from gamfit.sklearn import GAMRegressor, GAMClassifier

# Validate before you fit
gamfit.validate_formula(train, "y ~ s(x) + group(site)")

# Posterior sampling and predictive bands
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)

# Survival
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + bmi + timewiggle(internal_knots=6)",
    survival_likelihood="transformation",
)

# scikit-learn
est = GAMRegressor(formula="y ~ s(x)")
est.fit(X, y)

# Diagnose, plot, report
model.diagnose(train).metrics
model.plot(train, x="x", kind="prediction")
model.report("report.html")
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `gamfit.fit(data, formula, **kwargs)` | Fit a model. |
| `gamfit.load(path)` / `gamfit.loads(bytes)` | Reload a saved model. |
| `gamfit.validate_formula(data, formula, ...)` | Type-check a formula without fitting. |
| `gamfit.build_info()` | Native extension build metadata. |
| `gamfit.explain_error(exc)` | Human-readable hint for a gamfit exception. |
| `gamfit.Model` | Fitted-model handle: `predict`, `summary`, `check`, `diagnose`, `plot`, `report`, `sample`, `save`. |
| `gamfit.SurvivalPrediction` | Per-row hazard / survival surface; on-demand evaluation. |
| `gamfit.SamplingConfig`, `PosteriorSamples`, `PosteriorPredictive` | NUTS / posterior interface. |
| `gamfit.sklearn.GAMRegressor` / `GAMClassifier` | scikit-learn estimators. |

Full reference at <https://gamfit.readthedocs.io/en/latest/api-reference/>.

## Optional extras

```bash
uv add "gamfit[pandas]"     # pandas + pyarrow input/output
uv add "gamfit[plot]"       # matplotlib-based plotting
uv add "gamfit[sklearn]"    # scikit-learn integration
uv add "gamfit[all]"        # everything
```

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/SauersML/gam/blob/main/LICENSE).
