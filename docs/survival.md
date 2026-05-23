# Survival models

`gamfit` fits left-truncated, right-censored survival data with smooth
covariate effects. The response is `Surv(entry, exit, event)` and the
likelihood mode controls how the baseline and covariate effects are
parameterised.

## The `Surv(...)` response

```python
gamfit.fit(df, "Surv(entry, exit, event) ~ age + s(bmi)")
```

The three arguments are column names:

- `entry`: left-truncation time. Use `0` if there is no truncation.
- `exit`: observation time (event time or censoring time).
- `event`: `1` if the event occurred at `exit`, `0` if censored.

All three columns must be numeric. `exit >= entry` is required per row.
The 2-column R/mgcv form `Surv(time, status)` is rejected with an error
message that suggests adding a zero `entry` column.

## Likelihood modes

Pass one of the following via `survival_likelihood=`:

| Mode | Description |
| --- | --- |
| `"transformation"` | I-spline monotone log-cumulative-hazard baseline with linear or smooth covariate effects. Default. |
| `"weibull"` | Weibull parametric baseline with linear covariate effects on the log hazard. |
| `"location-scale"` | Joint location and log-scale survival model; requires a `noise_formula`. See [location-scale.md](location-scale.md). |
| `"marginal-slope"` | Separates a calibrated risk-score effect from the baseline. See [marginal-slope.md](marginal-slope.md). |
| `"latent"` | Parametric baseline with latent-Gaussian frailty integration. |
| `"latent-binary"` | Binary response under the same latent-Gaussian framework as `"latent"`. |

`--predict-noise` requires `survival_likelihood="location-scale"`; it
is rejected for every other survival mode.

```python
gamfit.fit(df,
    "Surv(t0, t1, event) ~ s(age) + bmi",
    survival_likelihood="transformation",
)
```

## Parametric baselines

For modes that support a parametric baseline (`"weibull"`,
`"location-scale"`, `"latent"`, and `"transformation"` when used with
`timewiggle(...)`), select it with `baseline_target=`:

| `baseline_target` | Required extra parameters | Notes |
| --- | --- | --- |
| `"linear"` | none | Linear-in-log-time baseline `[1, log(age)]`. Pair with `timewiggle(...)` for flexible departures. |
| `"weibull"` | `baseline_scale > 0`, `baseline_shape > 0` | Monotone hazard. |
| `"gompertz"` | `baseline_rate > 0`; `baseline_shape` optional (default 0.01, must be finite) | Exponentially-rising hazard. |
| `"gompertz-makeham"` | `baseline_rate > 0`, `baseline_makeham > 0`; `baseline_shape` optional (default 0.01, must be finite) | Gompertz hazard plus a constant additive floor. |

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
`timewiggle(...)`.

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

`Model.predict(...)` returns a [`SurvivalPrediction`](predictions.md#survivalprediction)
that evaluates the survival surface on a user-supplied time grid:

```python
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])
F = pred.failure_at([10, 20])
h = pred.hazard_at([1, 5, 10, 20])
H = pred.cumulative_hazard_at([10, 20])
```

For dense surfaces on large cohorts use the chunked iterators or stream
to CSV:

```python
for row_slice, time_slice, block in pred.survival_at_chunks([1, 5, 10, 20]):
    process(block)

pred.write_survival_at_csv("surv.csv", times=[1, 5, 10, 20])
```

For competing risks, predict each cause-specific endpoint and assemble
CIFs on the same grid:

```python
cif = gamfit.competing_risks_cif(
    {"disease": disease_pred, "death": death_pred},
    times=[1, 5, 10, 20],
)

disease_cif = cif.cif[0]
overall_survival = cif.overall_survival
```

## Uncertainty on the survival surface

For location-scale survival, `with_uncertainty=True` produces delta-method
standard errors:

```python
pred = model.predict(test_df, with_uncertainty=True)

S = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se_S).clip(0.0, 1.0)
lower = (S - 1.96 * se_S).clip(0.0, 1.0)
```

For other survival modes, use `Model.sample(...)` to draw posterior
coefficients, then push them through `PosteriorSamples.predict(...)` /
`predict_draws(...)`. Those methods are restricted to standard,
non-link-wiggle GAMs; see [posterior-sampling.md](posterior-sampling.md).

## Paired competing-risks posterior CIF

For two cause-specific fits, `sample_paired(...)` aligns draw `k` from
the target-cause fit with draw `k` from the competing-cause fit. CIF
integration and equal-tailed bands are computed in the Rust engine from
those paired draws:

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
cif.lower
cif.upper
```

Pass `competing_data=` to `sample_paired(...)` when the two fits need
different training tables.

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

grid_df = pd.DataFrame({"age": [50, 60, 70], "bmi": [25, 27, 29]})
pred = model.predict(grid_df)
print(pred.survival_at([1, 5, 10, 20]))
```

## Marginal-slope for risk scores

When a standardised risk score (e.g. a polygenic score) has an effect
that varies across covariate space,
`survival_likelihood="marginal-slope"` together with `z_column=` and
`logslope_formula=` models the score's spatially-varying effect
separately from the baseline. See [marginal-slope.md](marginal-slope.md).
