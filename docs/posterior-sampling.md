# Posterior sampling

`gamfit` fits smoothing parameters by REML / LAML (a point estimate)
and then draws from the posterior of the coefficients conditional on
those smoothing parameters. The sampler dispatches among NUTS,
Polya-Gamma Gibbs, and a Gaussian Laplace approximation based on model
class; see [Sampler dispatch](#sampler-dispatch) below.

## Quick start

```python
posterior = model.sample(train_df, seed=42)
print(posterior)
# PosteriorSamples(n_draws=..., n_coeffs=8, method='nuts',
#                  rhat=1.0040, ess=..., converged=True)

bands = posterior.predict(test_df, level=0.95)
# {"linear_predictor", "linear_predictor_lower", "linear_predictor_upper",
#  "mean",             "mean_lower",             "mean_upper"}
```

## Model.sample

```python
model.sample(
    data,
    *,
    samples: int | None = None,
    warmup:  int | None = None,
    chains:  int | None = None,
    target_accept: float | None = None,
    seed:    int | None = None,
) -> PosteriorSamples
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input matching the training schema. Survival models also consume the entry/exit/event columns. |
| `samples` | derived from coefficient count | Post-warmup draws per chain. |
| `warmup` | matches `samples` | Warmup iterations per chain (discarded). |
| `chains` | `2` if `p <= 50`, else `4` | Independent chains. |
| `target_accept` | `0.9` | NUTS step-size adaptation target acceptance; NUTS paths require it to lie in `(0, 1)`. The sampler floors it to `0.90` (dim ≤ 50) or `0.92` (dim > 50) and caps it at `0.95` via `robust_target_accept`, so a requested value outside that band is clamped. Ignored by the Laplace and Polya-Gamma Gibbs paths. |
| `seed` | `42` | RNG seed consumed by the sampler. |

Total returned draws are `chains * samples`.

## Posterior predictive replicates

`Model.sample(...)` draws coefficient uncertainty. For observation-level
synthetic responses from the fitted predictive distribution, use
`sample_replicates`:

```python
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
reps = model.posterior_predict(train_df, n_draws=200, seed=42)
# shape: (200, n_rows); object array of class labels
```

## Sampler dispatch

The dispatch is in `crates/gam-inference/src/sample.rs::sample_saved_model`:

| Model class | Sampler |
| --- | --- |
| Gaussian-identity standard GLM | Laplace (closed form; see note below) |
| Standard GLM (binomial probit/cloglog/latent-cloglog, Poisson, Tweedie, negative-binomial, Gamma) | NUTS |
| Bernoulli-logit standard GLM (no Firth, no offset, unit weights) | Polya-Gamma Gibbs |
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

The Laplace path draws iid samples from `N(beta_hat, phi * H_penalized^{-1})`
using the saved penalized Hessian's Cholesky factor and the saved dispersion
scale. Regardless of which sampler actually ran, every Laplace draw set
reports `rhat == 1.0`, `ess == chains * samples`, and `converged == True`
by construction. The `PosteriorSamples` API is identical either way.

The exposed `method` string is derived from the saved-model **class**, not
from the sampler that ran (`nuts_method_label`): only the dedicated
location-scale / survival-Laplace / transformation-normal / marginal-slope
classes report `method == "laplace"`. The `Standard` class always reports
`method == "nuts"`, so a **Gaussian-identity standard GLM** and a
**bounded-coefficient standard GLM** are drawn from the closed-form Laplace
posterior above yet still surface as `method == "nuts"` (and therefore
`is_exact == True`). The Polya-Gamma Gibbs path likewise surfaces under
`method == "nuts"`.

## SamplingConfig

`posterior.config` echoes the configuration the sampler ran with.
Fields:

| Field | Type |
| --- | --- |
| `n_samples` | `int` |
| `n_warmup` | `int` |
| `n_chains` | `int` |
| `target_accept` | `float` |
| `seed` | `int` |

`posterior.config.to_dict()` returns the same fields as a plain dict.

## PosteriorSamples

Frozen dataclass holding the draws and convergence diagnostics.

### Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `samples` | `numpy.ndarray` | `(n_draws, n_coeffs)` row-major float64 matrix. |
| `coefficient_names` | `tuple[str, ...]` | Currently emitted as `("beta_0", "beta_1", ...)`. |
| `mean`, `std` | `numpy.ndarray` | Per-coefficient posterior mean and standard deviation. |
| `rhat` | `float` | Maximum split-Rhat. `1.0` exactly for Laplace draws. |
| `ess` | `float` | Minimum effective sample size across coefficients. For Laplace draws this is `chains * samples`. |
| `converged` | `bool` | Sampler convergence flag. Laplace draws set this to `True`; most NUTS / Gibbs paths require `rhat < 1.1` and enough ESS. |
| `method` | `str` | `"nuts"` or `"laplace"`. |
| `model_class` | `str` | Saved-model predictive class. |
| `family_kind` | `str` | Inverse-link tag (`"identity"`, `"logit"`, `"probit"`, `"cloglog"`, `"log"`, ...). |
| `config` | `SamplingConfig` | Echo of the sampler configuration. |

Properties: `n_draws`, `n_coeffs`, `shape`, `is_exact` (`method == "nuts"`).

### Indexing

```python
posterior["beta_1"]                    # (n_draws,)
posterior[0]                           # (n_coeffs,)
posterior[:100]                        # (100, n_coeffs)
posterior[posterior["beta_0"] > 0]     # boolean mask over draws
```

A string key raises `KeyError` if it does not match `coefficient_names`.

### Summary and credible intervals

```python
ci = posterior.interval(level=0.95)        # (n_coeffs, 2)
summary = posterior.summary(level=0.95)    # Summary object
print(summary)                             # text repr; HTML in notebooks
```

`interval` and `summary` reject `level` outside `(0, 1)`.

### Conversion

```python
posterior.to_numpy()          # samples (no copy)
posterior.to_pandas()         # DataFrame with coefficient_names columns
```

### Posterior credible bands on new data

```python
bands = posterior.predict(test_df, level=0.95)
# {"linear_predictor", "linear_predictor_lower", "linear_predictor_upper",
#  "mean",             "mean_lower",             "mean_upper"}
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
pp = posterior.predict_draws(test_df)   # PosteriorPredictive
pp.eta      # (n_draws, n_rows), link scale
pp.mean     # (n_draws, n_rows), response scale (inverse link applied)
pp.shape    # (n_draws, n_rows)
pp.summary(level=0.95)   # same dict as posterior.predict
```

`predict_draws` materializes the full `(n_draws, n_rows)` matrix. For
large prediction sets prefer `posterior.predict(...)`.

The response-scale inverse link supports `identity`, `logit`, `probit`,
`cloglog`, and `log`; other tags raise a `gamfit.GamError`.

### Trace plots

```python
fig = posterior.plot_trace(coefficients=["beta_0", "beta_2"], max_panels=4)
```

Each row has two panels: the trace (draws vs iteration index) on the
left, a marginal density histogram on the right. With `coefficients=None`
the first `min(max_panels, n_coeffs)` coefficients are plotted.

### Save and load

```python
posterior.save("posterior.npz")
loaded = gamfit.PosteriorSamples.load("posterior.npz")
# or
loaded = gamfit.load_posterior("posterior.npz")

bands = loaded.predict(new_data)
```

The `.npz` archive bundles the saved-model bytes, so `predict` works
after a round-trip. The archive uses `allow_pickle=True` on load
(the metadata is stored as a 0-d object array); only load files you
produced.

`posterior.predict(...)` works for models with a closed-form design
matrix. Model classes that require the full saved-model predict path
(link-wiggle, survival, Bernoulli marginal-slope, transformation-normal,
and any model with a custom `predict` pipeline) raise from the FFI;
use `Model.predict(...)` for those.

## Convergence

`rhat < 1.01` is typical for well-mixed NUTS chains; `rhat < 1.1` is
the split-Rhat threshold used by `converged`. Standard NUTS and
Polya-Gamma Gibbs paths also require `ess > 100`; survival NUTS currently
uses the R-hat threshold only. If a NUTS run looks unhealthy:

1. Set `seed=` to retry from a different initialisation.
2. Increase `warmup` and `samples`.
3. Raise `target_accept` (e.g. `0.92` or `0.95`).
4. Inspect `posterior.plot_trace(...)`.

## Default sampling parameters

`NutsConfig::for_dimension` in `crates/gam-inference/src/hmc_io.rs` derives defaults
from the coefficient count `p`:

| Parameter | Rule |
| --- | --- |
| `n_chains` | `2` if `p <= 50`, else `4`. |
| `n_samples` | `clamp(floor(100 * p * (1 + 2 * max(1, sqrt(p))) * 1.5), 500, 10_000)`. |
| `n_warmup` | Same as `n_samples`. |
| `target_accept` | `0.9`. |
| `seed` | `42` unless `seed=` is passed. |

Every keyword on `Model.sample` overrides the corresponding default.

## Recipes

### Derived quantity (odds ratio)

```python
beta_contrast = posterior["beta_1"]
or_draws = np.exp(beta_contrast)
or_mean = or_draws.mean()
or_lo, or_hi = np.quantile(or_draws, [0.025, 0.975])
print(f"OR = {or_mean:.2f} (95% CI {or_lo:.2f}-{or_hi:.2f})")
```

### Posterior fitted-mean residual check

```python
pp = posterior.predict_draws(train_df)
y = train_df["y"].to_numpy()
fitted_mean = pp.mean.mean(axis=0)
sse_obs = ((y - fitted_mean) ** 2).sum()
sse_draw = ((pp.mean - y[None, :]) ** 2).sum(axis=1)
tail_area = (sse_draw > sse_obs).mean()
```

### Reproducibility

```python
posterior_a = model.sample(df, seed=12345)
posterior_b = model.sample(df, seed=12345)
assert np.allclose(posterior_a.samples, posterior_b.samples)
```
