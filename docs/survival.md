# Survival models

`gamfit` fits delayed-entry / left-truncated, right-censored survival data
with smooth covariate effects. The response is `Surv(entry, exit, event)`
and you choose a **likelihood mode** that controls how the baseline and
covariate effects are parameterized.

## The `Surv(...)` response

```python
gamfit.fit(df, "Surv(entry, exit, event) ~ age + s(bmi)")
```

The three arguments are column names:

- `entry` — left-truncation time (use `0` if no truncation).
- `exit` — the observation time (event time or censoring time).
- `event` — 1 if the event occurred at `exit`, 0 if censored.

All three must be present in `data` as numeric columns. Negative entry
times are allowed; `exit >= entry` is required per row.

!!! note "Coming from mgcv / R"
    gam only supports the 3-column form `Surv(entry, exit, event)`. The
    R/mgcv 2-column `Surv(time, status)` form is rejected with a hint
    explaining how to add an `entry` column of zeros if there is no
    left truncation.

## Likelihood modes

Pick one via `survival_likelihood=` on `fit()`:

| Mode | Description |
| --- | --- |
| `"transformation"` | Flexible, semi-parametric. Default when `survival_likelihood=` is not set; recommended starting point. |
| `"weibull"` | Parametric Weibull baseline + linear covariate effects on log hazard. |
| `"location-scale"` | Joint location-scale; combine with the `noise_formula` config key. |
| `"marginal-slope"` | Separates baseline risk from a calibrated score's effect. See [marginal-slope.md](marginal-slope.md). |
| `"latent"` | Latent-frailty model with parametric baseline. |
| `"latent-binary"` | Binary-outcome variant of `"latent"`: same latent-frailty framework, Bernoulli response instead of survival times. Incompatible with `--predict-noise`. |

```python
gamfit.fit(df,
    "Surv(t0, t1, event) ~ s(age) + bmi",
    survival_likelihood="transformation",
)
```

## Parametric baselines

When the likelihood mode supports a parametric baseline (Weibull,
location-scale, latent), pick the family with `baseline_target=`:

| `baseline_target` | Extra parameters | Use when |
| --- | --- | --- |
| `"linear"` | — | No parametric baseline target; the default I-spline time basis estimates a monotone log-cumulative-hazard baseline. |
| `"weibull"` | `baseline_scale`, `baseline_shape` (both > 0) | Monotone hazard: increasing if shape > 1, decreasing if shape < 1, constant if shape = 1. |
| `"gompertz"` | `baseline_rate` (> 0) | Exponentially-rising hazard (e.g. mortality with age). |
| `"gompertz-makeham"` | `baseline_rate`, `baseline_makeham` (both > 0) | Gompertz + an age-independent additive hazard floor. |

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(bmi)",
    survival_likelihood="latent",
    baseline_target="gompertz",
    baseline_rate=0.08,
)
```

## Time-wiggle: flexible baseline departures

With a non-linear scalar baseline target, drop in a spline offset to the
baseline:

```
Surv(entry, exit, event) ~ s(bmi) + timewiggle(internal_knots=8)
```

`timewiggle` takes the same options as `linkwiggle` (`internal_knots`,
`degree`, `penalty_order`, `double_penalty`) and lets the baseline hazard
or cumulative hazard flex away from the parametric form. Use it when the
rough shape is, say, Weibull or Gompertz but the data should make small
corrections. With `survival_likelihood="transformation"`, set
`baseline_target` to `weibull`, `gompertz`, or `gompertz-makeham` when using
`timewiggle(...)`.

## Frailty

For unmeasured between-subject heterogeneity, enable frailty with
`frailty_kind=`:

| `frailty_kind` | Effect |
| --- | --- |
| `"gaussian-shift"` | Additive Gaussian shift on the linear predictor. |
| `"hazard-multiplier"` | Multiplicative hazard frailty (latent log-normal). |

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="latent",
    baseline_target="gompertz",
    baseline_rate=0.08,
    frailty_kind="hazard-multiplier",
    hazard_loading="full",       # or "loaded-vs-unloaded"
)
```

- **`frailty_sd`**: fix the frailty standard deviation. Omit to let
  hazard-multiplier models learn it.
- **`hazard_loading`**: only used with `frailty_kind="hazard-multiplier"`.
  `"full"` loads frailty into every observation; `"loaded-vs-unloaded"`
  splits observations into two regimes.

## Predicting from a survival model

`Model.predict(...)` returns a [`SurvivalPrediction`](predictions.md#survivalprediction)
object whose methods evaluate the survival surface on a user-supplied time
grid:

```python
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])         # (n, 4) survival probs
F = pred.failure_at([10, 20])                # 1 - S
h = pred.hazard_at([1, 5, 10, 20])           # hazard rate
H = pred.cumulative_hazard_at([10, 20])      # cumulative hazard
```

For dense surfaces on large cohorts, use the chunked iterators or stream
directly to CSV:

```python
for row_slice, time_slice, block in pred.survival_at_chunks([1, 5, 10, 20]):
    process(block)                            # avoids materialising the full matrix

pred.write_survival_at_csv("surv.csv", times=[1, 5, 10, 20])
```

For competing risks, predict each cause-specific endpoint and assemble CIFs
on the same grid:

```python
cif = gamfit.competing_risks_cif(
    {"disease": disease_pred, "death": death_pred},
    times=[1, 5, 10, 20],
)

disease_cif = cif.cif[0]               # (n, 4)
overall_survival = cif.overall_survival
```

## Uncertainty on the survival surface

For location-scale survival, pass `with_uncertainty=True` to get
delta-method standard errors:

```python
pred = model.predict(test_df, with_uncertainty=True)

S = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se_S).clip(0.0, 1.0)
lower = (S - 1.96 * se_S).clip(0.0, 1.0)
```

`with_uncertainty=True` is honored for location-scale survival models. For
other survival modes, `Model.sample(...)` gives posterior coefficient draws;
`PosteriorSamples.predict(...)` / `predict_draws(...)` are restricted to
standard non-link-wiggle GAMs. See [posterior-sampling.md](posterior-sampling.md).

## Paired competing-risks posterior CIF

For two cause-specific survival fits, use `sample_paired(...)` so posterior
draw row `k` from the target-cause fit is paired with row `k` from the
competing-cause fit. The CIF integration and equal-tailed intervals are
computed in the Rust engine from those paired coefficient draws:

```python
disease_post = disease_model.sample_paired(
    death_model,
    train_df,
    samples=1000,
    chains=4,
    seed=42,
)

times = [0, 1, 5, 10, 20]
cif = disease_post.cumulative_incidence(test_df, times, level=0.95)

cif.draws   # (n_draws, n_rows, n_times)
cif.mean    # (n_rows, n_times)
cif.lower   # lower credible band
cif.upper   # upper credible band
```

Pass `competing_data=` to `sample_paired(...)` if the two fits need different
training tables.

## Complete example

```python
import gamfit
import pandas as pd

# Synthetic delayed-entry / right-censored data
df = pd.DataFrame({
    "entry": [0, 0, 0, 5, 5, 0],
    "exit":  [12, 8, 30, 22, 14, 15],
    "event": [1, 0, 1, 1, 0, 1],
    "age":   [55, 60, 45, 70, 50, 65],
    "bmi":   [24, 31, 22, 28, 26, 30],
})

model = gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + s(bmi) + timewiggle(internal_knots=6)",
    survival_likelihood="transformation",
    baseline_target="weibull",
)

# Population survival curves on a grid
grid_df = pd.DataFrame({"age": [50, 60, 70], "bmi": [25, 27, 29]})
pred = model.predict(grid_df)
print(pred.survival_at([1, 5, 10, 20]))
```

## Beyond survival modes: marginal-slope for scores

If your survival problem has a *standardised risk score* (e.g. a polygenic
risk score whose effect varies across covariate space), the
`survival_likelihood="marginal-slope"` mode pairs with `z_column=` and
`logslope_formula=` to model the score's spatially-varying effect
separately from the baseline. See [marginal-slope.md](marginal-slope.md).
