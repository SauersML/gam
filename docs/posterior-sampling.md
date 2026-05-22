# Posterior sampling

`gamfit` fits smoothing parameters by REML / LAML (a point estimate)
and then draws from the posterior of the coefficients conditional on
those smoothing parameters. The sampler dispatches between NUTS and a
Gaussian Laplace approximation based on model class; see
[When NUTS, when Laplace](#when-nuts-when-laplace) below.

## Quick start

```python
posterior = model.sample(train_df, seed=42)
print(posterior)
# PosteriorSamples(n_draws=..., n_coeffs=8, method='nuts',
#                  rhat=1.0040, ess=..., converged=True)

bands = posterior.predict(test_df, level=0.95)
# {"eta_mean", "eta_lower", "eta_upper",
#  "mean",     "mean_lower", "mean_upper"}
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
| `target_accept` | `0.8` | HMC target acceptance rate; must lie in `(0, 1)`. |
| `seed` | `42` | RNG seed consumed by the sampler. |

Total returned draws are `chains * samples`.

## When NUTS, when Laplace

The dispatch is in `src/inference/sample.rs::sample_saved_model`:

| Model class | Sampler |
| --- | --- |
| Standard GLM (Gaussian, binomial logit/probit/cloglog, Poisson, Gamma) | NUTS |
| Standard GLM with link-wiggle | NUTS (joint link-wiggle path) |
| Survival: Royston-Parmar (transformation), Weibull, marginal-slope | NUTS |
| Survival: latent, latent-binary, location-scale | Laplace |
| Gaussian location-scale | Laplace |
| Binomial location-scale | Laplace |
| Bernoulli marginal-slope | Laplace |
| Transformation-normal | Laplace |

The Laplace path draws iid samples from `N(beta_hat, H_penalized^{-1})`
using the saved penalized Hessian's Cholesky factor. The samples carry
`method == "laplace"`, `rhat == 1.0`, `ess == n_draws`, and
`converged == True` by construction. The `PosteriorSamples` API is
identical either way.

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
| `ess` | `float` | Minimum effective sample size across coefficients. |
| `converged` | `bool` | `rhat < 1.1`. |
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
bands = posterior.predict(test_df, chunk_size=4096, level=0.95)
# {"eta_mean", "eta_lower", "eta_upper",
#  "mean",     "mean_lower", "mean_upper"}
```

`predict` walks chunks of rows through `samples @ X_chunk.T`, collapsing
each chunk to per-row mean and quantiles. Peak memory is roughly
`n_draws * chunk_size * 8` bytes. Set `chunk_size=None` to materialize
the full matrix.

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
`cloglog`, and `log`; other tags raise `NotImplementedError`.

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

## PairedPosteriorSamples

Returned by `Model.sample_paired(...)`. Holds two `PosteriorSamples`
with draw rows paired by index. Exposes
`cumulative_incidence(new_data, times, level=0.95)` which returns a
`CumulativeIncidenceDraws` carrying `(n_draws, n_rows, n_times)`
target-cause CIF draws plus `mean`, `lower`, `upper` summaries.

## Convergence

`rhat < 1.01` is typical for well-mixed NUTS chains; `rhat < 1.1` is
the threshold `converged` reports. If a NUTS run looks unhealthy:

1. Set `seed=` to retry from a different initialisation.
2. Increase `warmup` and `samples`.
3. Raise `target_accept` (e.g. `0.85` or `0.9`).
4. Inspect `posterior.plot_trace(...)`.

## Default sampling parameters

`NutsConfig::for_dimension` in `src/inference/hmc.rs` derives defaults
from the coefficient count `p`:

| Parameter | Rule |
| --- | --- |
| `n_chains` | `2` if `p <= 50`, else `4`. |
| `n_samples` | `clamp(round(100 * p * (1 + 2*sqrt(p)) * 1.5), 500, 10_000)`. |
| `n_warmup` | Same as `n_samples`. |
| `target_accept` | `0.8`. |
| `seed` | `42` unless `seed=` is passed. |

Every keyword on `Model.sample` overrides the corresponding default.

## Recipes

### Derived quantity (odds ratio)

```python
beta_contrast = posterior["beta_treatment"]
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
