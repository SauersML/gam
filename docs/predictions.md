# Predictions

`Model.predict(...)` is the single prediction entry point. The return
shape depends on the fitted model class and on the keyword arguments
`interval`, `id_column`, and `return_type`.

## Signature

```text
model.predict(
    data,
    *,
    interval: float | str | None = None,
    conformal_level: float = 0.9,
    calibration: Any | None = None,
    covariance_mode: str | None = None,
    observation_interval: bool = False,
    return_type: str | None = None,
    id_column: str | None = None,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input matching the training schema. |
| `interval` | `None` | Single uncertainty knob. `None` returns point predictions only; a float in `(0, 1)` (e.g. `0.95`) requests the full uncertainty decomposition at that pointwise coverage. `"conformal"` requests a distribution-free conformal band: with `training_data` the exact full-conformal set for an eligible Gaussian-identity fit, or with `calibration` the split-conformal band. On standard GLMs and the location-scale families this populates `posterior_mean_standard_error`, `posterior_mean_lower`, and `posterior_mean_upper`; the transformation-normal and Bernoulli marginal-slope classes retain their class-specific `std_error` / `mean_lower` / `mean_upper` names. On supported single-event survival modes it populates `survival_se` and `eta_se`. On competing-risks survival it populates SE/lower/upper arrays for every cause-specific hazard, survival, cumulative hazard, CIF, overall survival, and eta surface. |
| `conformal_level` | `0.9` | Marginal coverage for `interval="conformal"`. Ignored for numeric Wald intervals. |
| `calibration` | `None` | Held-out labeled calibration table for the split-conformal band; `interval="conformal"` only. It must include the response column. |
| `training_data` | `None` | Labeled rows (normally the training table) for the exact full-conformal set; `interval="conformal"` only, exclusive with `calibration`. It must include the response column. |
| `covariance_mode` | `None` | Python accepts `"conditional"` or `"smoothing"` for interval covariance. `None` uses the covariance the fit *publishes* — the definition `summary()` prices its standard errors from: smoothing-corrected whenever the fit carries that matrix, otherwise conditional (a fit certified at an infinite-smoothing rail, for example) — and the result names the resolved definition in `covariance_source`. Naming a mode is a requirement: `"smoothing"` errors when the fit cannot supply the corrected matrix. Competing-risks predictions expose the resolved source as `covariance_source`; current cause-specific fits provide the full joint conditional covariance, so callers must request `"conditional"` until the fitter produces a smoothing correction. The CLI uses the equivalent `--covariance-mode conditional|corrected` names. |
| `observation_interval` | `False` | When `True` and `interval` is numeric, adds response-scale prediction interval columns for families with an observation variance. |
| `return_type` | `None` | One of `"dict"`, `"numpy"`, `"pandas"`, `"polars"`, `"pyarrow"` for table-shaped outputs. Defaults to the input table kind, falling back to the training table kind. |
| `id_column` | `None` | Name of a column in `data` whose stringified values are carried through into table outputs and `SurvivalPrediction`. |

When the table target is `"dict"` (either explicitly or because no input /
training table kind determines a richer container), the returned object is a
`PredictionResult`: it is still a normal mapping, so
`pred["posterior_mean"]` works for a standard model, and it exposes prediction
columns directly as attributes such as `pred.posterior_mean`,
`pred.posterior_mean_standard_error`, and `pred.posterior_mean_lower`.
For model-based intervals, a dict-shaped result also carries the scalar
`covariance_source` provenance field (`"conditional"` or
`"smoothing-corrected"`); pandas results expose the same value in
`result.attrs["covariance_source"]`.

## Return value by model class

| Model class | Default return | Columns / fields |
| --- | --- | --- |
| Gaussian, binomial, Poisson, negative-binomial, Gamma, Beta, Tweedie | 1-D `numpy.ndarray` | Response-scale posterior means. Table form has `linear_predictor_plugin`, `mean_plugin`, and `posterior_mean`; adds `posterior_mean_standard_error`, `posterior_mean_lower`, and `posterior_mean_upper` when `interval` is set. |
| Gaussian / binomial / dispersion location-scale | 1-D `numpy.ndarray` | Response-scale posterior means, on the same estimand-explicit schema as a standard fit: table form has `linear_predictor_plugin`, `mean_plugin`, `posterior_mean`, and `noise_scale` (the fitted scale channel); adds `posterior_mean_standard_error`, `posterior_mean_lower`, `posterior_mean_upper` when `interval` is set. |
| Transformation-normal | 1-D `numpy.ndarray` | Per-row response-scale conditional mean `E[Y|x]` (issue #1612). |
| Bernoulli marginal-slope | 1-D `numpy.ndarray` | Per-row probabilities clipped to `[0, 1]`. Table form has `mean`; with `interval=` it also includes `linear_predictor`, `std_error`, `mean_lower`, and `mean_upper`. |
| Survival (any likelihood mode) | `SurvivalPrediction` | Per-row hazard / survival evaluators. |
| Competing-risks survival | `CompetingRisksPrediction` | Endpoint-stacked hazard, survival, CIF, and overall survival arrays. |

### Standard prediction estimands are explicit

`linear_predictor_plugin` is `η̂ = Xβ̂` (and reproduces
`design_matrix(data) @ summary().coefficients` exactly).
`mean_plugin = g⁻¹(linear_predictor_plugin)` is its coherent response-scale
pair. `posterior_mean = E[g⁻¹(η) | data]` is the distinct, SPEC-mandated
default point prediction. For a curved inverse link, Jensen's inequality means
`posterior_mean` generally differs from `mean_plugin` by an
`O(Var(η̂)) = O(1/n)` term; for the identity link all three columns coincide.
The names expose that distinction directly rather than placing two estimands
under a generic `linear_predictor` / `mean` pair (#2785).

For all point-payload classes, passing `return_type=`, `id_column=`, or
numeric `interval=` switches output to a table. Transformation-normal
uses `"z"` as the value column; bernoulli marginal-slope uses `"mean"`
for probabilities. Passing `id_column=` adds that stringified id column
first.

## Wald intervals

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

preds = model.predict(test_df, interval=0.95)
# columns: linear_predictor_plugin, mean_plugin, posterior_mean,
#          posterior_mean_standard_error, posterior_mean_lower, posterior_mean_upper

pred_dict = model.predict(test_df, interval=0.95, return_type="dict")
mu = pred_dict["posterior_mean"]       # mapping access
mu_attr = pred_dict.posterior_mean     # same column on PredictionResult output
lo = pred_dict.posterior_mean_lower
```

Intervals are computed from the asymptotic covariance of the fitted
coefficients propagated through the inverse link.

For response-scale prediction intervals, also pass
`observation_interval=True`:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

preds = model.predict(test_df, interval=0.95, observation_interval=True)
# adds observation_lower, observation_upper when the family supports it
```

### Transformation-normal observation intervals

A conditional transformation-normal model is `F(y | x) = Phi(h(y | x))` with
`h(.|x)` strictly increasing, so its `p`-quantile is `h^-1(Phi^-1(p) | x)` —
quantiles map through the inverse transform, they are **not**
`E[Y|x] +- z * sigma` in latent units. `observation_interval=True` returns those
quantiles.

Two consequences of `h` being the whole model are worth knowing before reading a
limit:

* **The predictive law is supported on the whole real line, including past the
  training range.** Beyond the fitted knots the transformation is continued
  affinely at its own boundary derivative (the classical linear-tail
  extrapolation), so `Phi(h)` is a proper CDF there and an extreme limit is a
  genuine extrapolation rather than the largest response ever seen. Under a
  well-calibrated fit roughly `1/(n+1)` of the predictive mass sits beyond each
  end of the training range, so levels past about `1 - 2/n` are reporting that
  extrapolation.
* **The transformation is fitted on the raw response scale**, so nothing
  constrains an extrapolated limit to respect a bound the response happens to
  have — a strictly positive response can have a negative lower limit at an
  extreme level. If the bound is part of the model, fit the transformed response
  (`log y` for a positive response); the transformation then extrapolates on that
  scale and the bound is structural.

## Conformal intervals

`interval="conformal"` replaces the response-scale `posterior_mean_lower` /
`posterior_mean_upper` columns with a distribution-free conformal band at
`conformal_level`. `gam predict --conformal (--training-data FILE |
--calibration FILE) --level L` runs the same routes; exactly one of the two
labeled tables is required.

With `training_data` it is the full-conformal set of the fit that re-selects
the smoothing strength by REML on the labeled rows plus the candidate test row,
for a Gaussian-identity model fitted without prior weights, offsets, or a link
wiggle. The test row is treated exactly like a training row, so the
finite-sample coverage theorem holds. The saved model keeps only the `p x p`
frozen penalty `S_lambda` and its smoothing-parameter count, never per-row
training data, so the labeled rows (normally the training table, response
column included) are passed again at predict time. The output adds
`conformal_certificate`: `0` (exact_frozen, nothing to re-select) or `1`
(honest_refit) where the guarantee holds, and a negative code for a typed
refusal (`-1` several smoothing parameters, `-2` a model saved without the
count, `-3` to `-6` a degenerate criterion or refit), where the row carries the
frozen-smoothing set with no finite-sample guarantee.

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

full = model.predict(
    test_df, interval="conformal", training_data=train_df, conformal_level=0.95
)
```

With `calibration` it is the split-conformal band `mu_hat(x) ± q_hat · s(x)`
calibrated from a held-out labeled fold. It carries finite-sample marginal
coverage `≥ conformal_level` regardless of model misspecification, and applies
to any standard GAM family.

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 500)
data = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 500)}
train_df = {k: v[:300] for k, v in data.items()}
cal_df = {k: v[300:] for k, v in data.items()}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

preds = model.predict(
    test_df,
    interval="conformal",
    calibration=cal_df,        # held-out fold; MUST include the response column
    conformal_level=0.9,
)
# columns: linear_predictor_plugin, mean_plugin, posterior_mean,
#          posterior_mean_standard_error, posterior_mean_lower, posterior_mean_upper
```

`calibration` must contain the response column in addition to the predictors
(the conformal multiplier `q_hat` is computed from its held-out residuals).
It may be any size and is independent of the training set. `covariance_mode`,
`observation_interval`, `return_type`, and `id_column` behave as in
`predict`.

## Predicting from a numeric array

For models fitted via `gamfit.fit_array(...)` (positional columns
`x0, x1, ..., x{p-1}`), predict directly from a numeric feature matrix:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
X_train = rng.uniform(0, 1, (300, 2))
y = np.sin(6 * X_train[:, 0]) + X_train[:, 1] ** 2 + rng.normal(0, 0.2, 300)
X_test = rng.uniform(0, 1, (20, 2))

model = gamfit.fit_array(X_train, y, "y ~ s(x0) + s(x1)")
y_hat = model.predict_array(X_test)                    # 1-D ndarray of point predictions
table = model.predict_array(X_test, interval=0.95)     # adds posterior-mean uncertainty columns
```

`predict_array` accepts `interval`, `covariance_mode`, and
`observation_interval` (same semantics as `predict`); it does not take
`return_type` or `id_column`. It is rejected for models fitted from a named
table — call `predict` with a `dict` / DataFrame there so columns match by
name. The companion `model.design_matrix_array(X)` returns the same typed
[`AffineDesign`](#fitted-affine-design) contract as `design_matrix`.

## Carrying an identifier column

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 5, 300)
model = gamfit.fit({"x": x, "y": np.cos(x) + rng.normal(0, 0.3, 300)}, "y ~ s(x)")

preds = model.predict(
    [
        {"patient_id": "P001", "x": 1.5},
        {"patient_id": "P002", "x": 2.5},
    ],
    id_column="patient_id",
    return_type="dict",
)
# preds = {"patient_id": ["P001", "P002"], "linear_predictor_plugin": [...],
#          "mean_plugin": [...], "posterior_mean": [...]}
```

The id column is not used by the model. Values are copied through after
the same string conversion used for table normalization.

## SurvivalPrediction

`Model.predict` returns a `SurvivalPrediction` dataclass for survival
families. The dense hazard/survival surface is evaluated by the Rust
core on a default time grid (derived from the entry/exit columns in
`data`) and stored on the returned object; the `*_at` helpers
interpolate that surface at arbitrary user times.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 75, n), rng.normal(27, 4, n)
t = rng.exponential(1 / np.exp(-3 + 0.04 * (age - 55) + 0.05 * (bmi - 27)))
c = rng.exponential(20, n)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c) + 0.1,
                         "event": (t < c).astype(float), "age": age, "bmi": bmi})
test_df = train_df.head(20)

model = gamfit.fit(train_df, "Surv(entry, exit, event) ~ s(age) + bmi")
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])        # (n_rows, 4) survival probabilities
h = pred.hazard_at([1, 5, 10, 20])          # hazard rate
H = pred.cumulative_hazard_at([10, 20])     # cumulative hazard
```

### Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `model_class` | `str` | Fitted model class string. |
| `parameters` | `numpy.ndarray` | `(n_rows, n_params)` per-row parameters. Treat as opaque; use the `*_at` helpers. |
| `parameter_names` | `tuple[str, ...]` | Column labels for `parameters`. |
| `times` | `numpy.ndarray \| None` | Shared time grid for the dense surfaces. |
| `hazard`, `survival`, `cumulative_hazard` | `numpy.ndarray \| None` | `(n_rows, len(times))` dense surfaces when produced by the FFI. |
| `linear_predictor` | `numpy.ndarray \| None` | Linear predictor at each row's exit time. |
| `survival_se` | `numpy.ndarray \| None` | Delta-method standard error on `S(t)`. Populated only when `interval=...` is set and the model uses the location-scale survival likelihood. |
| `eta_se` | `numpy.ndarray \| None` | Delta-method standard error on the linear predictor under the same conditions as `survival_se`. |
| `id_column`, `row_ids` | `str \| None`, `Sequence[str] \| None` | Set when `id_column=` was passed to `predict`. |

### Methods

```text
pred.hazard_at(times)              # (n_rows, len(times))
pred.survival_at(times)            # (n_rows, len(times))
pred.cumulative_hazard_at(times)   # (n_rows, len(times))
pred.survival_se_at(times)         # SE on S(t), or None if not computed
```

Each `times` argument is coerced to a 1-D array of finite floats; an empty
input is rejected.

### Chunked iteration

When `n_rows * len(times)` exceeds roughly one million cells the dense
helpers chunk internally before assembling the result. To stream
without materializing the full matrix, iterate the chunk generators:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 75, n), rng.normal(27, 4, n)
t = rng.exponential(1 / np.exp(-3 + 0.04 * (age - 55) + 0.05 * (bmi - 27)))
c = rng.exponential(20, n)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c) + 0.1,
                         "event": (t < c).astype(float), "age": age, "bmi": bmi})
test_df = train_df.head(20)
blocks = []
process = blocks.append

pred = gamfit.fit(train_df, "Surv(entry, exit, event) ~ s(age) + bmi").predict(test_df)
for row_slice, time_slice, block in pred.survival_at_chunks(
    times=[1, 5, 10, 20, 50, 100],
    people_chunk=50_000,
    time_grid_chunk=64,
):
    process(block)  # shape (len(row_slice), len(time_slice))
```

`hazard_at_chunks` and `cumulative_hazard_at_chunks` are equivalent
generators for the matching surfaces.

### Stream to CSV

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 75, n), rng.normal(27, 4, n)
t = rng.exponential(1 / np.exp(-3 + 0.04 * (age - 55) + 0.05 * (bmi - 27)))
c = rng.exponential(20, n)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c) + 0.1,
                         "event": (t < c).astype(float), "age": age, "bmi": bmi})
test_df = train_df.head(20)

pred = gamfit.fit(train_df, "Surv(entry, exit, event) ~ s(age) + bmi").predict(test_df)
pred.write_survival_at_csv("surv.csv", times=[1, 5, 10, 20])
```

Writes one row per `(prediction_row, time)` pair. Columns are
`row, time, survival` when no id column is set, or
`row, <id_column>, time, survival` when `id_column=` was passed to
`predict`. The destination is truncated if it exists.

### Survival uncertainty

For the location-scale survival likelihood, passing any `interval=...`
populates delta-method standard errors:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 75, n), rng.normal(27, 4, n)
t = rng.exponential(1 / np.exp(-3 + 0.04 * (age - 55) + 0.05 * (bmi - 27)))
c = rng.exponential(20, n)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c) + 0.1,
                         "event": (t < c).astype(float), "age": age, "bmi": bmi})
test_df = train_df.head(20)

model = gamfit.fit(
    train_df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="location-scale",
    noise_formula="s(age)",
)
pred = model.predict(test_df, interval=0.95)
S = pred.survival_at([1, 5, 10])
se = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se).clip(0.0, 1.0)
lower = (S - 1.96 * se).clip(0.0, 1.0)
```

For a fitted competing-risks model, interval prediction propagates the complete
joint coefficient covariance through every cause-specific surface and the
Aalen-Johansen CIF recurrence:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age = rng.uniform(40, 75, n)
t1 = rng.exponential(1 / np.exp(-3.0 + 0.025 * (age - 55)))
t2 = rng.exponential(1 / np.exp(-3.2 - 0.02 * (age - 55)))
c = rng.exponential(22, n)
cause = np.select([(t1 < t2) & (t1 < c), (t2 < t1) & (t2 < c)], [1.0, 2.0], 0.0)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum.reduce([t1, t2, c]) + 0.1,
                         "cause": cause, "age": age})
test_df = train_df.head(20)

# event codes 1..K in the event column select the joint competing-risks fit
model = gamfit.fit(train_df, "Surv(entry, exit, cause) ~ s(age)")
pred = model.predict(
    test_df,
    interval=0.95,
    covariance_mode="conditional",
)

pred.covariance_source       # "conditional"
pred.cif_se                  # (K * n_rows, n_times)
pred.cif_lower
pred.cif_upper
pred.overall_survival_se     # (n_rows, n_times)
```

Covariance selection is exact. Omitting `covariance_mode` means required
smoothing-corrected covariance; if the fit has only conditional covariance,
prediction raises instead of silently substituting it. Latent and
latent-binary survival still do not expose these surfaces.

### Competing-risks CIF

Fit one cause-specific survival endpoint per event type, then assemble
Aalen-Johansen cumulative incidence functions on a shared grid:

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
n = 400
age = rng.uniform(40, 75, n)
t1 = rng.exponential(1 / np.exp(-3.0 + 0.025 * (age - 55)))
t2 = rng.exponential(1 / np.exp(-3.2 - 0.02 * (age - 55)))
c = rng.exponential(22, n)
train_df = pd.DataFrame({"entry": 0.0, "exit": np.minimum.reduce([t1, t2, c]) + 0.1,
                         "disease": ((t1 < t2) & (t1 < c)).astype(float),
                         "death": ((t2 < t1) & (t2 < c)).astype(float), "age": age})
test_df = train_df.head(20)

disease_pred = gamfit.fit(train_df, "Surv(entry, exit, disease) ~ s(age)").predict(test_df)
death_pred = gamfit.fit(train_df, "Surv(entry, exit, death) ~ s(age)").predict(test_df)

cif = gamfit.competing_risks_cif(
    {"disease": disease_pred, "death": death_pred},
    times=[1, 5, 10, 20],
)

disease_cif = cif.cif[0]              # (n_rows, 4)
joint_survival = cif.overall_survival # (n_rows, 4)
```

`cif.cif` is an endpoint-ordered sequence of `(n_rows, n_times)` arrays.
Endpoint names are taken from the mapping keys, or supplied via
`endpoint_names=` when passing a sequence.

## Fitted affine design

`Model.design_matrix(data)` returns a typed `AffineDesign` for every standard
GAM that has a finite coefficient-frame representation. Its defining identity
is:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

affine = model.design_matrix(test_df)
fitted_eta = affine.offset + affine.matrix @ affine.coefficients
```

The object has ten explicit fields:

- `offset`: one value per row;
- `matrix`: the materialised VALUE operator in the named coefficient frame —
  the matrix in the identity above;
- `eta_gradient`: `d eta / d coefficients` in that same frame — the operator to
  use with the covariances below. It is the same array object as `matrix`
  whenever the fitted predictor is linear in its coefficients;
- `coefficients`: the exact fitted vector multiplied by `matrix`;
- `coefficient_frame`: `"full"` or `"link_wiggle_joint"`;
- `coefficient_start` / `coefficient_stop`: the represented half-open slice in
  that frame (`coefficient_slice` exposes the corresponding Python `slice`).
- `covariance_conditional`: conditional Bayesian coefficient covariance `Vb`,
  or `None` when the fit did not produce it;
- `covariance_smoothing_corrected`: smoothing-parameter-corrected Bayesian
  covariance `Vp`, or `None` when unavailable;
- `covariance_frequentist`: frequentist sandwich covariance `Ve`, or `None`
  when unavailable.

Every available covariance is in exactly `coefficient_frame` and has one row
and column per entry of `coefficients`. Definitions are never substituted: in
particular, a missing smoothing-corrected covariance stays `None` rather than
silently becoming the conditional covariance. Pointwise linear-predictor
variance can therefore be computed without constructing the full row-by-row
covariance — always through `eta_gradient`, never through `matrix`:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

affine = model.design_matrix(test_df)
covariance = affine.covariance_smoothing_corrected
if covariance is None:
    raise RuntimeError("this fit has no smoothing-corrected covariance")
eta_variance = np.einsum(
    "ij,jk,ik->i", affine.eta_gradient, covariance, affine.eta_gradient
)
```

For an ordinary GAM, `offset` is the model offset, `matrix` is the full saved
design (including deployment extensions), the coefficient frame is `"full"`,
and `eta_gradient` IS `matrix` (`eta` is linear in `beta`, so the design is its
own derivative and no second buffer is allocated). Posterior draws use that same
frame, so custom fitted-linear-predictor draws are:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train_df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
test_df = {"x": np.linspace(0.5, 9.5, 20)}
model = gamfit.fit(train_df, "y ~ s(x)")

affine = model.design_matrix(test_df)
posterior = model.sample(train_df, samples=200)
eta_draws = affine.offset + posterior.samples @ affine.matrix.T
```

For a link-wiggle fit, the final predictor at the fitted state is
`model_offset + X @ beta_mean + B(warp_index) @ beta_w`. Accordingly, `offset`
is the model row offset, `matrix` is the joint `[X, B]` design with `B`
evaluated at the exact saved warp index (including the frozen-index shift used
by the fit), and `coefficient_frame` is `"link_wiggle_joint"`. The returned
coefficients and covariances are the exact complete saved `[Mean, LinkWiggle]`
frame. Keeping both blocks is what preserves mean uncertainty and the
mean--wiggle covariance in custom contrasts. The basis is frozen at the fitted
state; this is an exact representation of the fitted predictor and its saved
coefficient covariance, not a claim that `B` is globally independent of the
coefficients away from that state.

That last point is exactly why `matrix` and `eta_gradient` separate here. The
warp index is `X @ beta_mean + offset + X @ shift`, so it moves with the mean
coefficients and

```text
d eta / d beta_mean = diag(1 + B'(index) @ beta_w) @ X    (not X)
d eta / d beta_w    = B(index)
```

`matrix` is `[X, B(index)]`, which reproduces the fitted `eta` exactly; the
missing warp slope on its mean block makes it the wrong operator for variance.
`eta_gradient` carries that slope and is built by the same code that produces
the standard errors `predict` reports, so `eta_gradient @ V @ eta_gradient.T`
and `predict` cannot disagree. `eta_gradient` is a distinct array only in this
frame; the wiggle block is shared verbatim with `matrix`.

Exact scan smoothers and coupled multi-surface model classes do not possess one
finite affine coefficient frame; `design_matrix` rejects them with a typed,
actionable error instead of fabricating a matrix.

## Difference-smooth contrasts

Use `Model.difference_smooth(view="x", group="group", data=data)` for covariance-aware pairwise smooth differences and optional simultaneous bands. See [Difference smooths](difference-smooths.md) for parameterisation choices and interval interpretation.
