# gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/SauersML/gam/blob/main/LICENSE)

A formula-first generalized additive model library for Python, backed by a
high-performance Rust engine.

`gamfit` fits Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms,
random effects, location-scale extensions, survival likelihoods, and
flexible/learnable links. Smoothing parameters are selected by REML or LAML.
Posterior sampling uses NUTS.

## Install

```bash
uv add gamfit
```

Or with a managed Python (one-off, no project required):

```bash
uv pip install gamfit
```

Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, Apple
silicon), and Windows. No Rust toolchain required.

## Quick start

```python
import gamfit

train = [
    {"y": 1.2, "x": 0.0},
    {"y": 1.9, "x": 1.0},
    {"y": 3.1, "x": 2.0},
    {"y": 4.5, "x": 3.0},
]

model = gamfit.fit(train, "y ~ s(x)")
predictions = model.predict([{"x": 1.5}, {"x": 2.5}], interval=0.95)
print(model.summary())
model.save("model.gam")
```

Pandas, pyarrow, dict-of-columns, and list-of-records inputs all work
without conversion.

## What's different

- **Three-part penalty structure.** Each smooth gets separate penalties
  for magnitude, gradient, and curvature. Most GAM libraries use one
  (curvature only) or two; the three-part decomposition gives the
  smoother more degrees of freedom to distinguish flat-but-offset
  functions from wiggly ones.
- **Flexible link functions.** A spline offset from a base link (e.g.
  probit) lets the data correct for link misspecification while
  encoding the belief that the base link is approximately right.
- **Surface smooths.** Thin-plate splines, Duchon radial bases with
  triple-operator regularization, and Matérn covariance smooths in
  arbitrary dimension, with automatic knot placement.
- **Adaptive anisotropy.** Per-axis spatial anisotropy lets the model
  shrink or stretch each feature axis independently within a single
  joint smooth.
- **Composable basis/kernel.** Mix and match the kernel of one spline
  family with the length-scale behavior of another (e.g. Duchon kernel
  with Matérn-style global κ scaling).

## scikit-learn integration

```python
from gamfit.sklearn import GAMRegressor

est = GAMRegressor(formula="y ~ s(x)")
est.fit(train)
preds = est.predict([{"x": 1.5}, {"x": 2.5}])
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `gamfit.fit(data, formula, **kwargs)` | Fit a model from a dataset and a Wilkinson-style formula. |
| `gamfit.load(path)` / `gamfit.loads(bytes)` | Reload a saved model. |
| `gamfit.validate_formula(data, formula, ...)` | Type-check a formula against a dataset without fitting. |
| `gamfit.build_info()` | Native-extension build metadata. |
| `gamfit.explain_error(exc)` | Convert a `gamfit` exception into a human-readable hint. |
| `gamfit.Model` | Fitted-model handle: `predict`, `summary`, `check`, `diagnose`, `plot`, `report`, `save`. |
| `gamfit.sklearn.GAMRegressor` / `GAMClassifier` | scikit-learn-compatible estimators. |
| `gamfit.pgs` | Polygenic-score helpers. |

See the [project documentation](https://github.com/SauersML/gam) for the
full guide, the formula DSL reference, and the CLI.

## Optional extras

```bash
uv add "gamfit[pandas]"     # pandas + pyarrow input/output
uv add "gamfit[plot]"       # matplotlib-based plotting
uv add "gamfit[sklearn]"    # scikit-learn integration
uv add "gamfit[all]"        # everything
```

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/SauersML/gam/blob/main/LICENSE).
A commercial license is available for closed-source or SaaS use — see
[COMMERCIAL.md](https://github.com/SauersML/gam/blob/main/COMMERCIAL.md).
