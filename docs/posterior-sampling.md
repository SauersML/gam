# Posterior sampling

`gamfit` fits smoothing parameters by REML / LAML (a point estimate)
and then draws from the posterior of the coefficients. The sampler
dispatches among NUTS, Polya-Gamma Gibbs, and a Gaussian Laplace
approximation based on model class; see
[Sampler dispatch](#sampler-dispatch) below. The MCMC routes sample the
exact likelihood at the fitted smoothing parameters; on a standard GLM
(NUTS and Pólya-Gamma) the draws are then mapped about their mean through
the linear optimal-transport map `T = Vb^{-1/2}(Vb^{1/2} V_c Vb^{1/2})^{1/2}
Vb^{-1/2}` that carries the conditional `Vb` onto the published
smoothing-corrected `V_c` (`T Vb T = V_c`), so the draws integrate the
smoothing uncertainty for every family while keeping the exact likelihood's
shape. `V_c` may be wider or narrower than `Vb` in a given direction (the
sigma-point cubature correction averages the curvature over `ρ`); the map
reaches it either way. The Laplace route draws from the covariance the
fit *publishes* — the smoothing-corrected `Vp` whenever the fit carries one —
so its draw spread agrees with `summary().std_error` and with the default
`predict(interval=...)` band on the same object. Every draw set reports
which covariance it describes in `covariance_source`.

## Quick start

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

posterior = model.sample(train_df, seed=42)
print(posterior)
# PosteriorSamples(n_draws=..., n_coeffs=12, method='laplace',
#                  rhat=1.0000, ess=..., converged=True)   # a Gaussian fit

bands = posterior.predict(test_df, level=0.95)
# {"linear_predictor", "linear_predictor_lower", "linear_predictor_upper",
#  "posterior_mean",   "posterior_mean_lower",   "posterior_mean_upper"}
```

## Model.sample

```text
model.sample(
    data,
    *,
    samples: int | None = None,
    seed:    int | None = None,
) -> PosteriorSamples
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input matching the training schema. Survival models also consume the entry/exit/event columns. |
| `samples` | derived from coefficient count | Post-warmup draws per chain. |
| `seed` | `42` | RNG seed consumed by the sampler. |

Every run uses two chains, the fewest from which split R-hat can see chains
that disagree, so it returns `2 * samples` draws. Warmup has no count: it
ends when the step size and metric have stabilized and the chains meet the
convergence targets below.

## Posterior predictive replicates

`Model.sample(...)` draws coefficient uncertainty. For observation-level
synthetic responses from the fitted predictive distribution, use
`sample_replicates`:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

def consume(chunk):
    pass  # e.g. accumulate a posterior-predictive statistic

rep = model.sample_replicates(test_df, n_draws=200, seed=42)
# shape: (200, n_rows)

# For large jobs, bound the output allocation explicitly. Chunk boundaries do
# not affect the deterministic draw stream.
for rep_chunk in model.iter_replicates(
    test_df, n_draws=1_000_000, chunk_size=512, seed=42
):
    consume(rep_chunk)  # each shape is at most (512, n_rows)
```

The replicate path dispatches from the saved fitted-family variant and fitted
dispersion; callers never restate a family or refit the model. It covers the
standard, location-scale, transformation-normal, exact spline-scan, latent
survival, single-cause survival, and competing-risk saved-model paths. For a
single-cause or latent survival fit the response is a conditional
event-in-window indicator. For competing risks, zero means no event in the
window and positive integer labels identify the persisted cause. Censoring and
inspection records are study-design mechanisms rather than draws from those
event laws, so this API does not invent them.

Expectile fits persist their asymmetric target `tau`, but an expectile is an
estimating-loss target rather than an observation distribution. Replicate
generation therefore raises a typed unsupported-sampler error for an expectile
artifact instead of silently borrowing the Gaussian law used by its inner
weighted solver.

`sample_replicates` is the convenient allocating form. `iter_replicates`
requires an explicit positive `chunk_size` and retains only one draw chunk at a
time. Adjacent chunks use seekable global draw indices, so concatenating them
is bit-for-bit identical to the allocating call for the same data, draw count,
and seed. Both forms are useful for simulation, posterior-predictive checks,
and calibration probes.

Multinomial models expose the categorical analogue, `posterior_predict`, which
draws replicate class-label vectors (`Categorical(softmax(X·beta_hat))`) you can
feed into your own posterior-predictive check:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 300)
site = np.array(["A", "B", "C"])[np.digitize(x + rng.normal(0, 1, 300), [-1, 1])]
train_df = {"x": x, "site": site}

model = gamfit.fit(train_df, "site ~ s(x)", family="multinomial")
reps = model.posterior_predict(train_df, n_draws=200, seed=42)
# shape: (200, n_rows); object array of class labels
```

## Sampler dispatch

The dispatch is in `crates/gam-inference/src/sample.rs::sample_saved_model`:

| Model class | Sampler |
| --- | --- |
| Gaussian-identity standard GLM | Laplace (closed form; see note below) |
| Standard GLM (binomial probit/cloglog/latent-cloglog, Poisson, Tweedie, negative-binomial, Gamma) | NUTS |
| Bernoulli-logit standard GLM (no offset, unit weights) | Polya-Gamma Gibbs |
| Bernoulli-logit standard GLM under the Jeffreys prior (Firth fit; no offset, unit weights) | Polya-Gamma Gibbs with a Jeffreys Metropolis step |
| Bounded-coefficient standard GLM | Laplace (latent logit scale) |
| Standard GLM with beta regression or binomial SAS / beta-logistic / blended links | Not implemented; raises |
| Standard GLM with link-wiggle | NUTS (joint link-wiggle path) |
| Survival: Royston-Parmar, Weibull, marginal-slope | NUTS |
| Survival: latent, latent-binary, location-scale | Laplace |
| Gaussian location-scale | Laplace |
| Binomial location-scale | Laplace |
| Dispersion location-scale | Laplace |
| Bernoulli marginal-slope | Laplace |
| Transformation-normal | Laplace |

Royston-Parmar above refers to the transformation survival
likelihood; it is unrelated to the transformation-normal class.

The Laplace path draws iid samples from `N(beta_hat, V)`, where `V` is the
smoothing-corrected covariance `Vp` when the fit carries one and otherwise
the conditional `phi * H_penalized^{-1}` from the saved penalized Hessian's
Cholesky factor and the saved dispersion scale — the same choice
`summary()` makes, reported in `covariance_source`. Every Laplace draw set
reports `rhat == 1.0`,
`ess == 2 * samples`, and `converged == True` by construction: no chain
ran, so those numbers diagnose nothing. The `PosteriorSamples` API is
identical either way.

The exposed `method` string is stamped by the sampler that produced the
draws (`PosteriorSampler` in `gam-inference`), never derived from the
model class, so it is always one of:

| `method` | Sampler | `is_exact` |
| --- | --- | --- |
| `"nuts"` | No-U-Turn HMC on the exact posterior | `True` |
| `"polya-gamma"` | Polya-Gamma Gibbs on the exact Bernoulli-logit posterior | `True` |
| `"polya-gamma-jeffreys"` | Polya-Gamma Gibbs on the exact Bernoulli-logit posterior under the Jeffreys prior `det I(β)^½` (a Firth fit): each Gibbs coefficient draw is an independence proposal accepted with probability `min(1, det I(β')^½ / det I(β)^½)`; `acceptance_rate` is reported | `True` |
| `"laplace"` | Independent draws from the Gaussian (Laplace) approximation, including the Gaussian-identity closed form, the bounded latent-chart draws, and the transformation-normal rejection draws | `False` |
| `"truncated-laplace"` | Reflective HMC on the inequality-truncated Gaussian approximation (shape-constrained standard fits); `rhat` / `ess` are measured | `False` |

`is_exact` therefore means "the draws target the model's exact posterior
rather than a Gaussian approximation of it". A Gaussian-identity standard
GLM is sampled by the closed-form Laplace path and reports
`method == "laplace"`, `is_exact == False`.

## SamplingConfig

`posterior.config` echoes the configuration the sampler ran with.
Fields:

| Field | Type |
| --- | --- |
| `n_samples` | `int` |
| `n_warmup` | `int` |
| `n_chains` | `int` |
| `seed` | `int` |

`n_warmup` is the warmup the run spent per chain (`0` for independent
draws), and `n_chains` is always `2`. `posterior.config.to_dict()` returns
the same fields as a plain dict.

## PosteriorSamples

Frozen dataclass holding the draws and convergence diagnostics.

### Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `samples` | `numpy.ndarray` | `(n_draws, n_coeffs)` row-major float64 matrix. |
| `coefficient_names` | `tuple[str, ...]` | Currently emitted as `("beta_0", "beta_1", ...)`. |
| `mean`, `std` | `numpy.ndarray` | Per-coefficient posterior mean and standard deviation. |
| `rhat` | `float` | Maximum split-Rhat. `1.0` exactly for Laplace draws. |
| `ess` | `float` | Minimum effective sample size across coefficients. For Laplace draws this is `2 * samples`. |
| `converged` | `bool` | Sampler convergence flag. Laplace draws set this to `True`; NUTS and Gibbs paths require `rhat < 1.1` and `ess > 100`. |
| `method` | `str` | `"nuts"`, `"polya-gamma"`, `"polya-gamma-jeffreys"`, `"laplace"`, or `"truncated-laplace"` — the sampler that ran (table above). |
| `acceptance_rate` | `float \| None` | Fraction of Metropolis proposals accepted over the kept draws, for a sampler with an accept/reject step (`"polya-gamma-jeffreys"`); `None` otherwise. |
| `exact` | `bool` | Whether `method` targets the exact posterior; the value behind `is_exact`. |
| `covariance_source` | `str` | `"smoothing-corrected"` (standard-GLM NUTS / Pólya-Gamma draws transported onto `V_c`, and Laplace draws from the published `Vp`) or `"conditional"` (the other MCMC routes, and any fit without a smoothing correction). Same vocabulary as `predict()`. |
| `model_class` | `str` | Saved-model predictive class. |
| `family_kind` | `str` | Inverse-link tag (`"identity"`, `"logit"`, `"probit"`, `"cloglog"`, `"log"`, ...). |
| `config` | `SamplingConfig` | Echo of the sampler configuration. |

Properties: `n_draws`, `n_coeffs`, `shape`, `is_exact` (the `exact` flag).

### Indexing

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

posterior["beta_1"]                    # (n_draws,)
posterior[0]                           # (n_coeffs,)
posterior[:100]                        # (100, n_coeffs)
posterior[posterior["beta_0"] > 0]     # boolean mask over draws
```

A string key raises `KeyError` if it does not match `coefficient_names`.

### Summary and credible intervals

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

ci = posterior.interval(level=0.95)        # (n_coeffs, 2)
summary = posterior.summary(level=0.95)    # Summary object
print(summary)                             # text repr; HTML in notebooks
```

`interval` and `summary` reject `level` outside `(0, 1)`.

### Conversion

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

posterior.to_numpy()          # samples (no copy)
posterior.to_pandas()         # DataFrame with coefficient_names columns
```

### Posterior credible bands on new data

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

bands = posterior.predict(test_df, level=0.95)
# {"linear_predictor", "linear_predictor_lower", "linear_predictor_upper",
#  "posterior_mean",   "posterior_mean_lower",   "posterior_mean_upper"}
```

`predict` builds the saved model's standard design matrix, computes
`samples @ X.T`, and collapses the resulting link-scale draws to per-row
mean and quantiles inside Rust. The link-scale columns are keyed
`linear_predictor*`; no engine-internal `eta` key is exposed.

`predict` raises `RuntimeError` if the `PosteriorSamples` was loaded
from disk without bundled model bytes. Model classes lacking a
closed-form design matrix (link-wiggle, survival, others) raise from
the FFI; use `Model.predict(...)` for those.

### Full draws

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

pp = posterior.predict_draws(test_df)   # PosteriorPredictive
pp.eta      # (n_draws, n_rows), link scale
pp.mean     # (n_draws, n_rows), response scale (inverse link applied)
pp.shape    # (n_draws, n_rows)
pp.summary(level=0.95)   # same dict as posterior.predict
```

`predict_draws` materializes the full `(n_draws, n_rows)` matrix. For
large prediction sets prefer `posterior.predict(...)`.

The response-scale inverse link supports `identity`, `logit`, `probit`,
`cloglog`, and `log`; other tags raise a `gamfit.errors.GamfitError`.

### Trace plots

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

fig = posterior.plot_trace(coefficients=["beta_0", "beta_1"], max_panels=4)
```

Each row has two panels: the trace (draws vs iteration index) on the
left, a marginal density histogram on the right. With `coefficients=None`
the first `min(max_panels, n_coeffs)` coefficients are plotted.

`posterior.predict(...)` works for models with a closed-form design
matrix. Model classes that require the full saved-model predict path
(link-wiggle, survival, Bernoulli marginal-slope, transformation-normal,
and any model with a custom `predict` pipeline) raise from the FFI;
use `Model.predict(...)` for those.

## Convergence

`rhat < 1.01` is typical for well-mixed NUTS chains; `rhat < 1.1` is
the split-Rhat threshold used by `converged`. NUTS and Polya-Gamma Gibbs
paths also require `ess > 100`. The same two targets end warmup. Under the
Jeffreys prior a coefficient draw changes only when its proposal is
accepted, so the kept draws hold at most `accepted + chains` distinct
vectors; when that bound cannot exceed the `ess > 100` target the run ends
with an error naming the acceptance count rather than returning draws that
cannot mix. NUTS warms
up in doubling windows until its step size and metric have stabilized and a
window meets both targets; Gibbs burns in the same way. When two consecutive
windows have chains that each mix on their own but disagree with one
another, the run ends with an error rather than warming up forever. If a run
looks unhealthy:

1. Set `seed=` to retry from a different initialisation.
2. Increase `samples`.
3. Inspect `posterior.plot_trace(...)`.

## Default sampling parameters

`NutsConfig::for_dimension` in `crates/gam-inference/src/hmc_io.rs` derives defaults
from the coefficient count `p`:

| Parameter | Rule |
| --- | --- |
| `n_samples` | `clamp(floor(100 * p * (1 + 2 * max(1, sqrt(p))) * 1.5), 500, 10_000)`. |
| `seed` | `42` unless `seed=` is passed. |

Every keyword on `Model.sample`, and the matching `gam sample` flag, overrides the corresponding default.

## Recipes

### Derived quantity (odds ratio)

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 400)
train_df = {"x": x, "y": (rng.uniform(size=400) < 1 / (1 + np.exp(-0.8 * x))).astype(float)}
model = gamfit.fit(train_df, "y ~ x", family="binomial")
posterior = model.sample(train_df, samples=200, seed=42)

beta_contrast = posterior["beta_1"]
or_draws = np.exp(beta_contrast)
or_mean = or_draws.mean()
or_lo, or_hi = np.quantile(or_draws, [0.025, 0.975])
print(f"OR = {or_mean:.2f} (95% CI {or_lo:.2f}-{or_hi:.2f})")
```

### Posterior fitted-mean residual check

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)})
model = gamfit.fit(train_df, "y ~ s(x)")
posterior = model.sample(train_df, samples=200, seed=42)

pp = posterior.predict_draws(train_df)
y = train_df["y"].to_numpy()
fitted_mean = pp.mean.mean(axis=0)
sse_obs = ((y - fitted_mean) ** 2).sum()
sse_draw = ((pp.mean - y[None, :]) ** 2).sum(axis=1)
tail_area = (sse_draw > sse_obs).mean()
```

### Reproducibility

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(df, "y ~ s(x)")

posterior_a = model.sample(df, samples=200, seed=12345)
posterior_b = model.sample(df, samples=200, seed=12345)
assert np.allclose(posterior_a.samples, posterior_b.samples)
```
