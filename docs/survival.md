# Survival models

`gamfit` fits left-truncated, right-censored survival data with smooth
covariate effects. The response is `Surv(entry, exit, event)` and the
likelihood mode controls how the baseline and covariate effects are
parameterised.

## The `Surv(...)` response

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = rng.exponential(10 * np.exp(-0.04 * (age - 60) - 0.05 * (bmi - 27)))   # event time
c = rng.uniform(2, 25, n)                                                    # censoring time
df = {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": (t <= c).astype(float),
      "age": age, "bmi": bmi}

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
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
age = rng.uniform(40, 80, n)
t = 10 * rng.weibull(1.5, n) * np.exp(-0.03 * (age - 60))   # true (unobserved) event time
c = rng.uniform(2, 25, n)
event = (t <= c).astype(float)                               # 1: T bracketed in (left, right]
df = {"age": age, "event": event,
      "left": np.where(event == 1, t * rng.uniform(0.5, 0.95, n), c),
      "right": np.where(event == 1, t * rng.uniform(1.05, 1.5, n), c)}

gamfit.fit(df, "SurvInterval(left, right, event) ~ s(age)",
           survival_likelihood="latent",
           baseline_target="weibull",
           frailty_kind="hazard-multiplier",
           hazard_loading="full")
```

`SurvInterval(L, R, event)` observes a bracket `T in (L, R]`.
Bracketed rows require finite `R >= L`. This path is dedicated to the
latent interval-censored likelihood and is not the same as
left-truncated `Surv(entry, exit, event)`.

## Likelihood modes

Pass one of the following via `survival_likelihood=`:

| Mode | Description |
| --- | --- |
| `"transformation"` | I-spline monotone log-cumulative-hazard baseline with linear or smooth covariate effects. Default in every frontend. |
| `"weibull"` | Weibull parametric baseline with linear covariate effects on the log hazard. |
| `"location-scale"` | Joint location and log-scale survival model; `noise_formula` can override the log-scale terms. See [location-scale.md](location-scale.md). |
| `"marginal-slope"` | Separates a calibrated risk-score effect from the baseline. See [marginal-slope.md](marginal-slope.md). |
| `"latent"` | Parametric baseline with latent-Gaussian frailty integration. |
| `"latent-binary"` | Binary response under the same latent-Gaussian framework as `"latent"`. |

When omitted, every frontend (Python, Rust, CLI) resolves the same
canonical default, `"transformation"` — the default lives in exactly
one place (`FitConfig::resolved_survival_likelihood`), so identical
requests select the identical likelihood regardless of entrance.
`--predict-noise` requires `survival_likelihood="location-scale"`; it
is rejected for every other survival mode.

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = rng.exponential(10 * np.exp(-0.04 * (age - 60) - 0.05 * (bmi - 27)))
c = rng.uniform(2, 25, n)
df = {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": (t <= c).astype(float),
      "age": age, "bmi": bmi}

gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
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
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
bmi = rng.normal(27, 4, n)
# Gompertz event times: hazard 0.08 * exp(0.1 t) * exp(0.05 (bmi - 27))
t = np.log1p(rng.exponential(1.0, n) * 0.1 / (0.08 * np.exp(0.05 * (bmi - 27)))) / 0.1
c = rng.uniform(2, 25, n)
df = {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": (t <= c).astype(float), "bmi": bmi}

gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(bmi)",
    survival_likelihood="transformation",
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

## Left truncation and the baseline time anchor

`Surv(entry, exit, event)` with `entry > 0` is **left truncation** (delayed
entry): the row is only under observation from `entry` onward, so it
contributes `H(exit) − H(entry)` rather than `H(exit)`. A cohort where some
rows are followed from the time origin and others join later — staggered entry
— counts as left-truncated too, and is the ordinary shape of a registry
cohort.

The baseline time basis is *centered* at an anchor time before fitting.
Re-centering is an exact affine reparameterization of the baseline offset, so
it does not change the model being fitted — only the frame the smoothing
selection sees it in. That frame matters: anchoring a left-truncated design at
the earliest entry leaves the design's trend coordinate large and one-signed
across every row, and that coordinate is the unpenalized null space of the time
penalty, so the smoothing selection is driven by an inflated score. Left
unfixed it rejects every seed on marginal-slope fits and rails the
transformation baseline into a covariate-independent surface.

The anchor is therefore chosen as:

| Data | Anchor |
| --- | --- |
| `survival_likelihood="marginal-slope"` | The robust interior anchor (median exit), always. |
| Any row entering above the time origin (left truncation, including staggered entry) | The robust interior anchor (median exit). |
| Every row entering at the time origin (ordinary right censoring) | The earliest entry age, which is ≈ the origin, so centering is a near-no-op. |

Every frontend applies the same rule, from the same function, so the same
formula, data and configuration produce the same fit through `gamfit.fit`, the
CLI, and a `gam.fit-request` document alike.

To override it, name the anchor in the data's own time units. It is then
honored verbatim by every likelihood mode:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
age = rng.uniform(40, 60, n)                                 # age at recruitment = entry age
t = rng.exponential(15 * np.exp(-0.05 * (age - 50)))         # years from entry to event
c = rng.uniform(5, 20, n)
df = {"entry": age, "exit": age + np.minimum(t, c), "event": (t <= c).astype(float), "age": age}

gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age)",
    survival_likelihood="location-scale",
    survival_time_anchor=55.0,
)
```

The CLI takes it through a fit-request document (`gam fit --request`), whose
key is also `survival_time_anchor`. The chosen anchor is persisted on the saved
model as `survival_time_anchor`, and prediction re-centers at that value, so a
model always predicts in the frame it was fitted in. Supplying it without a
`Surv(...)` response is a configuration error, not a silent no-op.

## Frailty

`frailty_kind=` enables a latent random effect:

| `frailty_kind` | Effect |
| --- | --- |
| `"gaussian-shift"` | Additive Gaussian shift on the linear predictor. |
| `"hazard-multiplier"` | Multiplicative log-normal frailty on the hazard. |

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
risk = np.exp(0.03 * (age - 60) + 0.05 * (bmi - 27) + rng.normal(0, 0.3, n))   # log-normal frailty
t = np.log1p(rng.exponential(1.0, n) * 0.1 / (0.08 * risk)) / 0.1                # Gompertz times
c = rng.uniform(2, 25, n)
df = {"entry": np.zeros(n), "exit": np.minimum(t, c), "event": (t <= c).astype(float),
      "age": age, "bmi": bmi}

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
  `gaussian-shift` in marginal-slope survival, where the likelihood reads
  sigma only through the slope and so does not identify a learned one
  (gam#2938), and for some other modes; omit to let
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
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"age": rng.uniform(40, 80, n), "bmi": rng.normal(27, 4, n)})
t = rng.exponential(10 * np.exp(-0.04 * (df["age"] - 60) - 0.05 * (df["bmi"] - 27)))
c = rng.uniform(2, 25, n)
df["entry"], df["exit"], df["event"] = 0.0, np.minimum(t, c), (t <= c).astype(float)
train_df, test_df = df.iloc[:300], df.iloc[300:]

model = gamfit.fit(train_df, "Surv(entry, exit, event) ~ s(age) + bmi")
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
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"age": rng.uniform(40, 80, n), "bmi": rng.normal(27, 4, n)})
t = rng.exponential(10 * np.exp(-0.04 * (df["age"] - 60) - 0.05 * (df["bmi"] - 27)))
c = rng.uniform(2, 25, n)
df["entry"], df["exit"], df["event"] = 0.0, np.minimum(t, c), (t <= c).astype(float)
train_df, test_df = df.iloc[:300], df.iloc[300:]

def process(block):                     # stand-in for your own per-chunk work
    print(block.shape, block.mean(axis=0))

pred = gamfit.fit(train_df, "Surv(entry, exit, event) ~ s(age) + bmi").predict(test_df)
for row_slice, time_slice, block in pred.survival_at_chunks([1, 5, 10, 20]):
    process(block)

pred.write_survival_at_csv("surv.csv", times=[1, 5, 10, 20])
```

For separate cause-specific fits, predict each endpoint and assemble CIFs
on the same grid:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"age": rng.uniform(40, 80, n)})
t_disease, t_death = rng.exponential(15 * np.exp(-0.04 * (df["age"] - 60))), rng.exponential(25, n)
t, c = np.minimum(t_disease, t_death), rng.uniform(2, 25, n)
df["entry"], df["exit"] = 0.0, np.minimum(t, c)
df["disease"] = ((t <= c) & (t_disease < t_death)).astype(float)
df["death"] = ((t <= c) & (t_death < t_disease)).astype(float)
train_df, test_df = df.iloc[:300], df.iloc[300:]

disease_pred = gamfit.fit(train_df, "Surv(entry, exit, disease) ~ s(age)").predict(test_df)
death_pred = gamfit.fit(train_df, "Surv(entry, exit, death) ~ s(age)").predict(test_df)

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
delta-method standard errors:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"age": rng.uniform(40, 80, n), "bmi": rng.normal(27, 4, n)})
t = rng.exponential(10 * np.exp(-0.04 * (df["age"] - 60) - 0.05 * (df["bmi"] - 27)))
c = rng.uniform(2, 25, n)
df["entry"], df["exit"], df["event"] = 0.0, np.minimum(t, c), (t <= c).astype(float)
train_df, test_df = df.iloc[:300], df.iloc[300:]

model = gamfit.fit(
    train_df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="location-scale",
    noise_formula="s(age)",
)
pred = model.predict(test_df, interval=0.95)

S = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se_S).clip(0.0, 1.0)
lower = (S - 1.96 * se_S).clip(0.0, 1.0)
```

Joint competing-risks fits expose uncertainty for every cause-specific
hazard, survival, cumulative-hazard, and CIF surface, plus overall survival
and each cause's linear predictor:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"age": rng.uniform(40, 80, n)})
t1, t2 = rng.exponential(15 * np.exp(-0.04 * (df["age"] - 60))), rng.exponential(25, n)
t, c = np.minimum(t1, t2), rng.uniform(2, 25, n)
df["entry"], df["exit"] = 0.0, np.minimum(t, c)
df["cause"] = np.where(t > c, 0.0, np.where(t1 < t2, 1.0, 2.0))   # 0 censored, 1 or 2 cause
train_df, test_df = df.iloc[:300], df.iloc[300:]

# event codes 1..K in the event column select the joint competing-risks fit
model = gamfit.fit(train_df, "Surv(entry, exit, cause) ~ s(age)")
pred = model.predict(
    test_df,
    interval=0.95,
    covariance_mode="conditional",
)

pred.covariance_source  # "conditional"
pred.cif_se
pred.cif_band_refusal  # why the CIF carries no band; see below
pred.overall_survival_se
```

The cumulative incidence carries no band: `CIF_k(t) = ∫ h_k S du` depends on every cause's whole
linear-predictor curve, so no central interval of its posterior law is derived, and
`cif_band_refusal` gives that reason in place of `cif_lower`/`cif_upper`.

The covariance request is strict. The default and `covariance_mode="smoothing"`
require a saved smoothing-corrected covariance and raise when it is absent;
they never substitute the conditional covariance. Current cause-specific fits
save the complete joint conditional covariance, including cross-cause blocks,
so their supported interval spelling is `covariance_mode="conditional"`.

Latent and latent-binary survival do not expose these surface intervals.
`Model.sample(...)` can draw posterior coefficients for supported saved
survival models, but `PosteriorSamples.predict(...)` / `predict_draws(...)`
are restricted to standard, non-link-wiggle GAMs; see
[posterior-sampling.md](posterior-sampling.md).

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

When a risk score has an effect that varies across covariate space,
`survival_likelihood="marginal-slope"` with
`slope_formula=` models the score's spatially-varying effect separately
from the baseline. Supply `transformation_normal_stage1=gamfit.CtnStage1(...)`
to condition the score on covariates and cross-fit it inside the one call,
or a raw `z_column=` when the score is already conditionally `N(0, 1)`. See
[marginal-slope.md](marginal-slope.md).
