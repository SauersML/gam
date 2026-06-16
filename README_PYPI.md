# gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/SauersML/gam/blob/main/LICENSE)

Formula-based generalized additive models for Python, backed by a Rust
engine.

`gamfit` fits Gaussian, binomial (including Bernoulli marginal-slope),
Poisson, negative-binomial, Gamma, Beta, Tweedie, and multinomial GLMs
with smooth terms, random effects,
bounded/constrained coefficients, location-scale extensions, survival
likelihoods, and flexible/learnable links. Smoothing parameters are
selected by REML or LAML. Posterior sampling uses NUTS where supported,
and a Gaussian Laplace approximation otherwise.

Manifold smooths handle predictor spaces that wrap or close: circles,
cylinders, tori, and the sphere (intrinsic Wahba and spherical-harmonic
kernels), plus periodic tensor products and boundary-conditioned
B-splines. The Möbius example in the gallery is a 4π-periodic
double-cover parameterization, not a twisted Möbius-strip basis.

![rotating recovery of a trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere, bumpy torus, and Möbius double-cover from noisy 3-D point clouds](https://raw.githubusercontent.com/SauersML/gam/main/docs/images/geometric_shapes_demo.gif)

Docs: <https://gamfit.readthedocs.io/>.

## Install

```bash
uv add gamfit
```

Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, Apple
silicon), and Windows. No Rust toolchain is required.

## Example

```python
import gamfit

# Smooth fits need enough rows for the basis to be identified; ~20 rows
# is the minimum the default `s(x)` basis (cubic B-spline) is well-posed
# on. Use more rows when the signal is noisier.
train = [
    {"y": 1.05, "x": 0.0}, {"y": 1.32, "x": 0.5}, {"y": 1.78, "x": 1.0},
    {"y": 2.41, "x": 1.5}, {"y": 3.10, "x": 2.0}, {"y": 3.95, "x": 2.5},
    {"y": 4.80, "x": 3.0}, {"y": 5.62, "x": 3.5}, {"y": 6.25, "x": 4.0},
    {"y": 6.71, "x": 4.5}, {"y": 6.94, "x": 5.0}, {"y": 6.88, "x": 5.5},
    {"y": 6.55, "x": 6.0}, {"y": 5.99, "x": 6.5}, {"y": 5.20, "x": 7.0},
    {"y": 4.30, "x": 7.5}, {"y": 3.42, "x": 8.0}, {"y": 2.65, "x": 8.5},
    {"y": 2.10, "x": 9.0}, {"y": 1.82, "x": 9.5},
]

model = gamfit.fit(train, "y ~ s(x)")
print(model.predict([{"x": 1.5}, {"x": 5.0}], interval=0.95))
print(model.summary())
model.save("model.gam")
```

pandas, polars, pyarrow, numpy, dict-of-columns, and list-of-records
inputs are all accepted without conversion.

## Features

- Polyharmonic / Duchon smooths combine magnitude, gradient, and
  curvature penalty operators on the same basis. P-spline and
  thin-plate smooths use their standard derivative penalties. Each
  penalized block has its own smoothing parameter.
- Flexible link functions: `flexible(base)` adds a spline offset on a
  base link; `blended(...)` learns a mixture weight; `sas` and
  `beta-logistic` learn shape parameters.
- Surface smooths in arbitrary dimension: thin-plate, Duchon (scale-free
  by default, hybrid with `length_scale=...`), and Matérn, with
  automatic knot placement.
- Tensor-product and manifold smooths: `te(...)` / `ti(...)` B-spline
  tensors, periodic 1-D, cylinder / torus tensor products,
  intrinsic sphere (Wahba kernel or spherical harmonics), and
  boundary-conditioned B-splines.
- Dispersion GAMLSS for Gamma, Beta, negative-binomial, and Tweedie via
  `noise_formula=`.
- Per-axis anisotropy inside a single joint smooth.
- Shape-constrained smooths: `s(x, shape=monotone_increasing)`,
  `convex`, `concave`.
- Difference smooths: `by=` factor smooths plus covariance-aware
  `model.difference_smooth(...)` contrasts with optional simultaneous
  bands.
- Marginal-slope models that separate baseline risk from a calibrated
  score's effect, for Bernoulli and survival outcomes.
- Survival in several likelihood modes (transformation, Weibull,
  location-scale, marginal-slope, latent-Gaussian frailty) plus
  competing-risks cumulative-incidence functions.
- Response geometry for spherical and compositional outcomes via
  Fréchet-mean tangent-space GAMs.
- Posterior sampling via NUTS where supported, Gaussian Laplace
  otherwise, behind one API; conformal prediction intervals via
  `interval="conformal"`.

## API examples

```python
import gamfit
from gamfit.sklearn import GAMRegressor, GAMClassifier

# Validate before you fit
gamfit.validate_formula(train, "y ~ s(x) + group(site)")

# Posterior sampling and mean bands
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)

# Survival
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + bmi + timewiggle(internal_knots=6)",
    survival_likelihood="transformation",
    baseline_target="weibull",
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
| `gamfit.load_posterior(path)` | Reload a `PosteriorSamples` archive. |
| `gamfit.validate_formula(data, formula, ...)` | Type-check a formula without fitting. |
| `gamfit.build_info()` | Native extension build metadata. |
| `gamfit.cuda_diagnostics()` / `gamfit.format_cuda_diagnostics()` | CUDA probe results. |
| `gamfit.explain_error(exc)` | Human-readable hint for a gamfit exception. |
| `gamfit.Model` | Fitted model: `predict`, `summary`, `check`, `diagnose`, `plot`, `report`, `sample`, `save`. |
| `gamfit.SurvivalPrediction` | Per-row hazard / survival surface. |
| `gamfit.CompetingRisksPrediction`, `competing_risks_cif` | Competing-risks CIF evaluation. |
| `gamfit.MultinomialModel` | Multinomial-logit / softmax model. |
| `gamfit.SamplingConfig`, `PosteriorSamples`, `PosteriorPredictive`, `PairedPosteriorSamples` | Posterior interface. |
| `gamfit.ResponseGeometryModel`, `sphere_frechet_mean`, `simplex_frechet_mean`, `alr`, `clr`, `closure` | Response-geometry utilities. |
| `gamfit.smooth.Duchon`, `Matern`, `BSpline`, `TensorBSpline`, `MeasureJet`, `Sphere` | Smooth descriptors for `smooths=` and torch. |
| `gamfit.sklearn.GAMRegressor` / `GAMClassifier` | scikit-learn estimators. |

Full reference: <https://gamfit.readthedocs.io/en/latest/api-reference/>.

## Optional extras

```bash
uv add "gamfit[pandas]"     # pandas + pyarrow input/output
uv add "gamfit[plot]"       # matplotlib-based plotting
uv add "gamfit[sklearn]"    # scikit-learn integration
uv add "gamfit[cuda]"       # NVIDIA CUDA 12 wheel libraries on Linux x86_64
uv add "gamfit[all]"        # pandas + plot + sklearn extras
uv add torch                # PyTorch bridge dependency
```

## GPU acceleration

CUDA support (cuBLAS / cuSOLVER / cuSPARSE) is built into the same
wheel; there is no separate `gamfit-gpu` package. Install
`gamfit[cuda]` on Linux x86_64 when you want PyPI's NVIDIA CUDA 12
runtime libraries instead of a system CUDA toolkit. Per-op dispatch
thresholds are derived at probe time from measured GPU FP64 throughput,
CPU FP64 throughput, and PCIe bandwidth, so small kernels stay on the
CPU. Inspect the calibrated thresholds with
`gamfit.build_info()["cuda_diagnostics"]` or
`gamfit.format_cuda_diagnostics()`.

If both a system CUDA toolkit and pip `nvidia-*-cu12` wheels are present
in the same environment, gamfit warns once per conflict-set and
continues; glibc resolves `dlopen(SONAME)` to a single file, so this is
usually benign. If you use gamfit with `torch`, install a torch build
whose CUDA suffix matches your driver.

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/SauersML/gam/blob/main/LICENSE).
