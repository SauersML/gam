# gam

[![PyPI](https://img.shields.io/pypi/v/gam.svg)](https://pypi.org/project/gam/)
[![Python](https://img.shields.io/pypi/pyversions/gam.svg)](https://pypi.org/project/gam/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/SauersML/gam/blob/main/LICENSE)

A formula-first generalized additive model library for Python, backed by a
high-performance Rust engine.

`gam` fits Gaussian, binomial, Poisson, and Gamma GLMs with smooth terms,
random effects, location-scale extensions, survival likelihoods, and
flexible/learnable links. Smoothing parameters are selected by REML or LAML.
Posterior sampling uses NUTS.

## Install

```bash
pip install gam
```

Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, Apple
silicon), and Windows. No Rust toolchain required.

## Quick start

```python
import gam

train = [
    {"y": 1.2, "x": 0.0},
    {"y": 1.9, "x": 1.0},
    {"y": 3.1, "x": 2.0},
    {"y": 4.5, "x": 3.0},
]

model = gam.fit(train, "y ~ s(x)")
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
from gam.sklearn import GAMRegressor

est = GAMRegressor(formula="y ~ s(x)")
est.fit(train)
preds = est.predict([{"x": 1.5}, {"x": 2.5}])
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `gam.fit(data, formula, **kwargs)` | Fit a model from a dataset and a Wilkinson-style formula. |
| `gam.load(path)` / `gam.loads(bytes)` | Reload a saved model. |
| `gam.validate_formula(data, formula, ...)` | Type-check a formula against a dataset without fitting. |
| `gam.build_info()` | Native-extension build metadata. |
| `gam.explain_error(exc)` | Convert a `gam` exception into a human-readable hint. |
| `gam.Model` | Fitted-model handle: `predict`, `summary`, `check`, `diagnose`, `plot`, `report`, `save`. |
| `gam.sklearn.GAMRegressor` / `GAMClassifier` | scikit-learn-compatible estimators. |
| `gam.pgs` | Polygenic-score helpers. |

See the [project documentation](https://github.com/SauersML/gam) for the
full guide, the formula DSL reference, and the CLI.

## Optional extras

```bash
pip install "gam[pandas]"    # pandas + pyarrow input/output
pip install "gam[plot]"      # matplotlib-based plotting
pip install "gam[sklearn]"   # scikit-learn integration
pip install "gam[all]"       # everything
```

## License

Apache-2.0. See [LICENSE](https://github.com/SauersML/gam/blob/main/LICENSE).
