# Posterior sampling

`gamfit` fits smoothing parameters by REML / LAML (a point estimate) and
then samples from the **posterior of the coefficients conditional on those
smoothing parameters**. Sampling uses NUTS (No-U-Turn Sampler) where the
model class supports it, and a Gaussian Laplace approximation otherwise.

## Quick start

```python
posterior = model.sample(train_df, seed=42)
print(posterior)
# PosteriorSamples(n_draws=1024, n_coeffs=8, method='nuts',
#                  rhat=1.0042, ess=890.5, converged=True)

bands = posterior.predict(test_df, level=0.95)
# Dict with eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper
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
)
```

| Kwarg | Default | Meaning |
| --- | --- | --- |
| `data` | (required) | Same shape as the data you fit on. Survival models also consume `entry`, `exit`, `event` columns. |
| `samples` | auto from coeff count | Post-warmup draws **per chain**. |
| `warmup` | matches `samples` (or adaptive) | Warmup iterations per chain. Discarded. |
| `chains` | 2 or 4 (auto) | Independent chains run in parallel. |
| `target_accept` | ~0.8 | HMC target acceptance rate. Higher → smaller steps. |
| `seed` | random | RNG seed for reproducibility. |

Total returned draws are `chains × samples`.

## When NUTS, when Laplace

| Model class | Sampler |
| --- | --- |
| Gaussian, binomial (logit/probit/cloglog), Poisson, Gamma | **NUTS** |
| Standard GLM with `linkwiggle` | **NUTS** (joint link-wiggle path) |
| Survival (transformation, Weibull, marginal-slope) | **NUTS** |
| Survival (location-scale, latent, latent-binary) | Gaussian Laplace |
| Gaussian / binomial location-scale | Gaussian Laplace |
| Bernoulli marginal-slope | Gaussian Laplace |
| Transformation-normal | Gaussian Laplace |

On Laplace fallback, `posterior.method == "laplace"`, `rhat == 1.0`,
`ess == n_draws`, and `converged == True` — the draws are i.i.d. by
construction. The `PosteriorSamples` API is identical either way.

## SamplingConfig

`posterior.config` echoes what the sampler ran with:

```python
posterior.config.n_samples       # int
posterior.config.n_warmup        # int
posterior.config.n_chains        # int
posterior.config.target_accept   # float in (0, 1)
posterior.config.seed            # int
posterior.config.to_dict()       # serialise
```

## PosteriorSamples

Container for posterior draws.

### Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `samples` | `(n_draws, n_coeffs) ndarray` | Raw draws (row-major float64). |
| `coefficient_names` | `tuple[str, ...]` | Column labels (currently `beta_0`, `beta_1`, …). |
| `mean`, `std` | `(n_coeffs,) ndarray` | Per-coeff posterior moments. |
| `rhat` | `float` | Max split-Rhat. `< 1.01` excellent, `< 1.05` good, `< 1.1` OK. |
| `ess` | `float` | Min effective sample size across coefficients. |
| `converged` | `bool` | `rhat < 1.1`. |
| `method` | `str` | `"nuts"` or `"laplace"`. |
| `model_class`, `family_kind` | `str` | Model class string and inverse-link tag. |
| `config` | `SamplingConfig` | What the sampler ran with. |
| `n_draws`, `n_coeffs`, `shape`, `is_exact` | properties | Convenience. |

### Slicing and indexing

```python
posterior["beta_1"]           # 1-D array of draws for beta_1
posterior[0]                  # 1 draw, shape (n_coeffs,)
posterior[:100]               # first 100 draws
posterior[posterior["beta_0"] > 0]   # boolean mask
```

### Credible intervals and summary

```python
ci = posterior.interval(level=0.95)        # (n_coeffs, 2)
summary = posterior.summary(level=0.95)    # Summary object
print(summary)                              # pretty table, also HTML in notebooks
```

### Conversion

```python
posterior.to_numpy()          # (n_draws, n_coeffs)
posterior.to_pandas()         # DataFrame with coefficient names as columns
```

### Posterior predictive bands

```python
bands = posterior.predict(test_df, level=0.95)
# {"eta_mean": ..., "eta_lower": ..., "eta_upper": ...,
#  "mean":     ..., "mean_lower": ..., "mean_upper": ...}
```

The method streams in chunks of size `chunk_size` (default 4096 rows),
collapsing each chunk to quantiles on-the-fly. Memory stays bounded at
roughly `n_draws × chunk_size × 8` bytes regardless of total prediction-set
size. Set `chunk_size=None` to materialise the full matrix.

Supports standard non-link-wiggle GAMs. Other model classes raise with a
pointer to `Model.predict(interval=...)`.

### Full draws

For raw `(n_draws, n_rows)` matrices (e.g. to propagate uncertainty into a
derived quantity):

```python
pp = posterior.predict_draws(test_df)
pp.eta      # (n_draws, n_rows) on link scale
pp.mean     # (n_draws, n_rows) on response scale

# Per-row posterior means / quantiles
import numpy as np
row_mean   = pp.mean.mean(axis=0)
row_p025   = np.quantile(pp.mean, 0.025, axis=0)
row_p975   = np.quantile(pp.mean, 0.975, axis=0)
```

### Trace plots

```python
fig = posterior.plot_trace(coefficients=["beta_0", "beta_2"], max_panels=4)
```

Each row shows two panels: trace (draws vs iteration) on the left,
histogram on the right. Healthy traces look like white noise with no trend.

### Save / load

```python
posterior.save("posterior.npz")
loaded = gamfit.load_posterior("posterior.npz")
# or
loaded = gamfit.PosteriorSamples.load("posterior.npz")

bands = loaded.predict(new_data)   # works after a round-trip
```

The `.npz` archive bundles the fitted model bytes too, so `predict()` on a
loaded posterior does not require re-passing the model.

## Convergence diagnostics

| `rhat` | Status |
| --- | --- |
| `< 1.01` | Excellent. |
| `< 1.05` | Good — fine for publication. |
| `< 1.1` | OK — usable, consider a longer run. |
| `> 1.1` | Poor. Increase `warmup` and `samples`, raise `target_accept`, or inspect the model. |

Effective sample size: target ESS / n_draws > ~25%. Substantially lower
means the chain is autocorrelated; raise `target_accept` to 0.85–0.9 or
extend the run.

If the sampler looks unhealthy:

1. Re-fit with `seed` set — the initialisation may have been unlucky.
2. Double `warmup` and `samples`.
3. Raise `target_accept` (0.85 → 0.9).
4. Inspect `posterior.plot_trace(...)` for bimodality or trends.

## Numerical defaults

The engine derives sampling parameters from the coefficient count `p`
(`NutsConfig::for_dimension` in `src/inference/hmc.rs`):

| Parameter | Rule |
| --- | --- |
| `n_chains` | `2` if `p ≤ 50`, else `4`. |
| `n_samples` | `clamp(round(100·p·(1 + 2·√p)·1.5), 500, 10_000)` — target ≈ 100 effective samples per coefficient assuming NUTS autocorrelation `τ ≈ √p`. |
| `n_warmup` | Same as `n_samples`. |
| `target_accept` | `0.8` always. |
| `seed` | `42` (a `seed=` kwarg overrides). |

Override any of them with the corresponding kwarg.

## Recipes

### Propagate uncertainty to a derived quantity

```python
# Posterior of an odds ratio (logit model with a contrast coefficient)
beta_contrast = posterior["beta_treatment"]
or_draws = np.exp(beta_contrast)
or_mean = or_draws.mean()
or_lo, or_hi = np.quantile(or_draws, [0.025, 0.975])
print(f"OR = {or_mean:.2f} (95% CI {or_lo:.2f}–{or_hi:.2f})")
```

### Posterior-predictive p-value

```python
pp = posterior.predict_draws(train_df)
# Replicate residual sum of squares from each posterior draw
import numpy as np
y = train_df["y"].to_numpy()
sse_rep = ((pp.mean - y[None, :]) ** 2).sum(axis=1)
print("posterior-predictive p-value:", (sse_rep > sse_obs).mean())
```

### Reproducibility

```python
posterior_a = model.sample(df, seed=12345)
posterior_b = model.sample(df, seed=12345)
assert np.allclose(posterior_a.samples, posterior_b.samples)
```
