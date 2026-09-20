# gamfit

[![PyPI](https://img.shields.io/pypi/v/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Python](https://img.shields.io/pypi/pyversions/gamfit.svg)](https://pypi.org/project/gamfit/)
[![Docs](https://img.shields.io/readthedocs/gamfit.svg)](https://gamfit.readthedocs.io/)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](https://github.com/SauersML/gam/blob/main/LICENSE)

gamfit fits generalized additive models from a formula, chooses every
smoothing parameter by REML/LAML in one converged optimization, and returns
posterior-mean predictions with credible bands and observation intervals,
from a Rust engine.

```bash
uv add gamfit   # or: pip install gamfit
```

Wheels are published for Linux (x86_64, aarch64), macOS (x86_64, Apple
silicon), and Windows. No Rust toolchain is required.

## Example

```python
import pandas as pd
import gamfit

# 133 rows: head acceleration of a crash-test dummy, milliseconds after impact.
mcycle = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv")

# The mean and the noise level are both smooth functions of time.
model = gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")

bands = model.predict(mcycle, interval=0.95, observation_interval=True)
print(bands[["posterior_mean", "posterior_mean_lower", "posterior_mean_upper",
             "observation_lower", "observation_upper"]].head())
```

![mcycle location-scale fit: posterior mean, credible band and observation interval](https://raw.githubusercontent.com/SauersML/gam/main/docs/images/mcycle_location_scale.png)

## Coming from pyGAM

- **Smoothness is estimated, not searched.** REML/LAML picks every
  smoothing parameter, so there is no `gridsearch()` and no GCV. Against
  pyGAM's defaults gamfit wins 8, ties 28 and loses 13 of 49 held-out
  comparisons; the [benchmarks](https://gamfit.readthedocs.io/en/latest/benchmarks/) list every loss.
- **Predictions carry their uncertainty.** One `predict` call returns the
  posterior mean, a credible band for it and an observation interval
  ([predictions](https://gamfit.readthedocs.io/en/latest/predictions/)).
- **The noise can be modelled too.** `noise_formula=` fits a
  location-scale model like the one above, which pyGAM cannot express; on
  `mcycle` its 95% observation interval covers 97% of the data
  ([tour](https://gamfit.readthedocs.io/en/latest/tour/#heteroscedastic-noise-mcycle)).

The [migration guide](https://gamfit.readthedocs.io/en/latest/migrating-from-pygam/) maps pyGAM calls to
gamfit. Docs: <https://gamfit.readthedocs.io/>.

## Scope

`gamfit` fits Gaussian, binomial (including Bernoulli marginal-slope),
Poisson, negative-binomial, Gamma, Beta, Tweedie, and multinomial GLMs
with smooth terms, random effects,
bounded/constrained coefficients, location-scale extensions, survival
likelihoods, and flexible/learnable links. Posterior sampling uses NUTS
where supported, and a Gaussian Laplace approximation otherwise.

Manifold smooths handle predictor spaces that wrap or close: circles,
cylinders, tori, and the sphere (intrinsic Wahba and spherical-harmonic
kernels), plus periodic tensor products and boundary-conditioned
B-splines. The Möbius example in the gallery is a 4π-periodic
double-cover parameterization, not a twisted Möbius-strip basis.

![rotating recovery of a trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere, bumpy torus, and Möbius double-cover from noisy 3-D point clouds](https://raw.githubusercontent.com/SauersML/gam/main/docs/images/geometric_shapes_demo.gif)

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
import numpy as np
import pandas as pd
import gamfit
from gamfit.sklearn import GAMRegressor, GAMClassifier

rng = np.random.default_rng(0)
train = pd.DataFrame({"x": rng.uniform(0, 10, 300), "site": rng.choice(["A", "B", "C"], 300)})
train["y"] = np.sin(train.x) + (train.site == "B") + rng.normal(0, 0.3, 300)
test = train.drop(columns="y").head(5)
X, y = train[["x"]], train["y"].to_numpy()

# Validate before you fit
gamfit.validate_formula(train, "y ~ s(x) + group(site)")
model = gamfit.fit(train, "y ~ s(x) + group(site)")

# Posterior sampling and mean bands
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)

# Survival
age, bmi = rng.uniform(30, 80, 400), rng.normal(25, 4, 400)
t = 15 * rng.weibull(1.5, 400) * np.exp(-(age - 55) / 20 - (bmi - 25) / 10)
df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, 25), "event": (t < 25) * 1.0, "age": age, "bmi": bmi})
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
model.plot_terms()                        # each term's partial effect with bands
model.partial_dependence("s(x)").simultaneous_upper
model.report("report.html")
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `gamfit.fit(data, formula, **kwargs)` | Fit a model. |
| `gamfit.load(path)` / `gamfit.loads(bytes)` | Reload a saved model. |
| `gamfit.validate_formula(data, formula, ...)` | Type-check a formula without fitting. |
| `gamfit.build_info()` | Native extension build metadata. |
| `gamfit.cuda.cuda_diagnostics()` / `gamfit.cuda.format_cuda_diagnostics()` | CUDA probe results. |
| `gamfit.explain_error(exc)` | Human-readable hint for a gamfit exception. |
| `gamfit.Model` | Fitted model: `predict`, `summary`, `check`, `diagnose`, `plot`, `report`, `sample`, `save`. |
| `gamfit.results.SurvivalPrediction` | Per-row hazard / survival surface. |
| `gamfit.results.CompetingRisksPrediction`, `competing_risks_cif` | Competing-risks CIF evaluation. |
| `gamfit.MultinomialModel` | Multinomial-logit / softmax model. |
| `gamfit.results.SamplingConfig`, `PosteriorSamples`, `PosteriorPredictive`, `PairedPosteriorSamples` | Posterior interface. |
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
`gamfit.cuda.format_cuda_diagnostics()`.

The wheel uses the CUDA 12 ABI. If PyTorch has already mapped a complete CUDA
stack, gamfit continues that same stack rather than preloading a second system
toolkit. Without an existing stack it loads one complete system or packaged
NVIDIA stack. The GPU probe refuses a partial or mixed mapped stack because
CUDA context and library-handle ownership cannot be safely split across
implementations.

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/SauersML/gam/blob/main/LICENSE).
