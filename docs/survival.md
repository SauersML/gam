# Survival models

`gamfit` fits left-truncated, right-censored survival data with smooth
covariate effects. The response is `Surv(entry, exit, event)` and the
likelihood mode controls how the baseline and covariate effects are
parameterised.

## The `Surv(...)` response

```python
gamfit.fit(df, "Surv(entry, exit, event) ~ age + s(bmi)")
```

The usual three-argument form names:

- `entry`: left-truncation time. Use `0` if there is no truncation.
- `exit`: observation time (event time or censoring time).
- `event`: integer event code. `0` means censored and `1` means the
  event occurred at `exit`; contiguous positive codes `1..K` select a
  joint competing-risks fit for the transformation and Weibull modes.

All three columns must be numeric. Negative or non-finite times are
rejected; zero times are accepted and internally floored, and `exit` is
advanced to at least `entry + 1e-9` during fitting/prediction.

The right-censored shorthand `Surv(time, event)` is also accepted and is
lowered to `Surv(0, time, event)` with a synthetic zero-entry column.

Interval-censored responses use a distinct spelling:

```python
gamfit.fit(df, "SurvInterval(left, right, event) ~ s(age)",
           survival_likelihood="latent")
```

`SurvInterval(L, R, event)` observes a bracket `T in (L, R]`.
Bracketed rows require finite `R >= L`. This path is dedicated to the
latent interval-censored likelihood and is not the same as
left-truncated `Surv(entry, exit, event)`.

## Likelihood modes

Pass one of the following via `survival_likelihood=`:

| Mode | Description |
| --- | --- |
| `"transformation"` | I-spline monotone log-cumulative-hazard baseline with linear or smooth covariate effects. CLI default. |
| `"weibull"` | Weibull parametric baseline with linear covariate effects on the log hazard. |
| `"location-scale"` | Joint location and log-scale survival model; `noise_formula` can override the log-scale terms. See [location-scale.md](location-scale.md). |
| `"marginal-slope"` | Separates a calibrated risk-score effect from the baseline. See [marginal-slope.md](marginal-slope.md). |
| `"latent"` | Parametric baseline with latent-Gaussian frailty integration. |
| `"latent-binary"` | Binary response under the same latent-Gaussian framework as `"latent"`. |

When omitted in `gamfit.fit(...)`, the Rust/Python fit path uses
`"location-scale"`; the `gam fit` CLI default is `"transformation"`.
`--predict-noise` requires `survival_likelihood="location-scale"`; it
is rejected for every other survival mode.

```python
gamfit.fit(df,
    "Surv(t0, t1, event) ~ s(age) + bmi",
    survival_likelihood="transformation",
)
```

## Parametric baselines

For modes that support a scalar parametric baseline (`"transformation"`,
`"weibull"`, `"location-scale"`, `"marginal-slope"`, `"latent"`, and
`"latent-binary"`), select it with `baseline_target=`:

| `baseline_target` | Fit-time parameter defaults | Notes |
| --- | --- | --- |
| `"linear"` | none | Linear-in-log-time baseline `[1, log(age)]`. Pair with `timewiggle(...)` for flexible departures. |
| `"weibull"` | `baseline_scale` defaults to the mean positive exit time; `baseline_shape` defaults to `1.0` | Monotone hazard. |
| `"gompertz"` | `baseline_rate` defaults to `1 / mean_positive_exit`; `baseline_shape` defaults to `0.01` | Exponentially-rising hazard. |
| `"gompertz-makeham"` | `baseline_rate` and `baseline_makeham` default to `0.5 / mean_positive_exit`; `baseline_shape` defaults to `0.01` | Gompertz hazard plus a constant additive floor. |

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(bmi)",
    survival_likelihood="latent",
    baseline_target="gompertz",
    baseline_rate=0.08,
)
```

## `timewiggle` for flexible baseline departures

`timewiggle(...)` adds a spline offset to a non-linear scalar baseline:

```
Surv(entry, exit, event) ~ s(bmi) + timewiggle(internal_knots=8)
```

It accepts the same options as `linkwiggle(...)`: `internal_knots`,
`degree`, `penalty_order`, and `double_penalty`. With
`survival_likelihood="transformation"`, set `baseline_target` to
`"weibull"`, `"gompertz"`, or `"gompertz-makeham"` when using
`timewiggle(...)`. `timewiggle(...)` is also available for
`"location-scale"`, `"marginal-slope"`, and `"weibull"` fits, but is
rejected for `"latent"` and `"latent-binary"`.

## Frailty

`frailty_kind=` enables a latent random effect:

| `frailty_kind` | Effect |
| --- | --- |
| `"gaussian-shift"` | Additive Gaussian shift on the linear predictor. |
| `"hazard-multiplier"` | Multiplicative log-normal frailty on the hazard. |

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="latent",
    baseline_target="gompertz",
    baseline_rate=0.08,
    frailty_kind="hazard-multiplier",
    hazard_loading="full",
)
```

- `frailty_sd`: fix the frailty standard deviation. Required for
  `gaussian-shift` (learnable sigma is not implemented for the exact
  marginal-slope outer solver) and for some other modes; omit to let
  hazard-multiplier latent models learn it where supported.
- `hazard_loading`: only used with `frailty_kind="hazard-multiplier"`.
  `"full"` loads frailty into every observation; `"loaded-vs-unloaded"`
  splits observations into two regimes.

Survival marginal-slope accepts only `frailty_kind="gaussian-shift"` with
a fixed `frailty_sd`; `"hazard-multiplier"` is rejected at fit time.

## Prediction

For one-cause survival fits, `Model.predict(...)` returns a
[`SurvivalPrediction`](predictions.md#survivalprediction) that evaluates
the survival surface on a user-supplied time grid:

```python
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])
F = 1.0 - pred.survival_at([10, 20])
h = pred.hazard_at([1, 5, 10, 20])
H = pred.cumulative_hazard_at([10, 20])
```

Restricted mean survival time (RMST) is the area under the survival
curve up to `tau`. In Python, evaluate `pred.survival_at(grid)` and
integrate over the grid you choose; the Rust survival prediction result
also carries a `restricted_mean_survival_time(tau)` helper for native
callers.

For dense surfaces on large cohorts use the chunked iterators or stream
to CSV:

```python
for row_slice, time_slice, block in pred.survival_at_chunks([1, 5, 10, 20]):
    process(block)

pred.write_survival_at_csv("surv.csv", times=[1, 5, 10, 20])
```

For separate cause-specific fits, predict each endpoint and assemble CIFs
on the same grid:

```python
cif = gamfit.competing_risks_cif(
    {"disease": disease_pred, "death": death_pred},
    times=[1, 5, 10, 20],
)

disease_cif = cif.cif[0]
overall_survival = cif.overall_survival
```

If the fitted event column contains contiguous positive event codes
`1..K`, `Model.predict(...)` returns a `CompetingRisksPrediction`
directly. Its `hazard`, `survival`, `cumulative_hazard`, and `cif`
arrays are endpoint-stacked with shape `(K * n_rows, n_times)`;
`overall_survival` has shape `(n_rows, n_times)`.

## Uncertainty on the survival surface

For location-scale survival, passing any `interval=...` produces
delta-method standard errors (issue #342 — the single `interval` knob
replaces the previous `with_uncertainty` boolean):

```python
pred = model.predict(test_df, interval=0.95)

S = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se_S).clip(0.0, 1.0)
lower = (S - 1.96 * se_S).clip(0.0, 1.0)
```

Other survival likelihood modes reject any non-`None` `interval` at the
Rust boundary. `Model.sample(...)` can draw posterior coefficients for
supported saved survival models, but `PosteriorSamples.predict(...)` /
`predict_draws(...)` are restricted to standard, non-link-wiggle GAMs;
see [posterior-sampling.md](posterior-sampling.md).

## Example

```python
import gamfit
import pandas as pd

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

grid_df = pd.DataFrame({
    "entry": [0, 0, 0],
    "exit": [20, 20, 20],
    "age": [50, 60, 70],
    "bmi": [25, 27, 29],
})
pred = model.predict(grid_df)
print(pred.survival_at([1, 5, 10, 20]))
```

## Marginal-slope for risk scores

When a risk score (e.g. a risk score) has an effect that varies
across covariate space, `survival_likelihood="marginal-slope"` with
`logslope_formula=` models the score's spatially-varying effect separately
from the baseline. Supply `transformation_normal_stage1=gamfit.CtnStage1(...)`
to condition the score on covariates and cross-fit it inside the one call,
or a raw `z_column=` when the score is already conditionally `N(0, 1)`. See
[marginal-slope.md](marginal-slope.md).
