# Survival models

`gamfit` fits left-truncated, right-censored, interval-censored survival
data with smooth covariate effects. The response is `Surv(entry, exit, event)`
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

## Likelihood modes

Pick one via `survival_likelihood=` on `fit()`:

| Mode | Description |
| --- | --- |
| `"transformation"` | Flexible, semi-parametric. Recommended starting point. |
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
| `"linear"` | — | Constant hazard. |
| `"weibull"` | `baseline_scale`, `baseline_shape` (both > 0) | Monotone hazard. |
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

In any survival formula, drop in a spline offset to the baseline:

```
Surv(entry, exit, event) ~ s(bmi) + timewiggle(internal_knots=8)
```

`timewiggle` takes the same options as `linkwiggle` (`internal_knots`,
`degree`, `penalty_order`, `double_penalty`) and lets the baseline hazard
or cumulative hazard flex away from the parametric form. Use it when the
rough shape is, say, Gompertz but the data should make small corrections.

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

## Uncertainty on the survival surface

For location-scale survival, pass `with_uncertainty=True` to get
delta-method standard errors:

```python
pred = model.predict(test_df, with_uncertainty=True)

S = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])

upper = S + 1.96 * se_S
lower = (S - 1.96 * se_S).clip(0.0, 1.0)
```

`with_uncertainty=True` is honored for location-scale survival models. For
other survival modes, use `Model.sample(...)` and the posterior-predictive
route — see [posterior-sampling.md](posterior-sampling.md).

## Complete example

```python
import gamfit
import pandas as pd

# Synthetic interval-censored data
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
