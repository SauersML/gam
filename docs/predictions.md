# Predictions

`Model.predict(...)` is the single prediction entry point. The return
shape depends on the fitted model class and on the keyword arguments
`interval`, `id_column`, and `return_type`.

## Signature

```python
model.predict(
    data,
    *,
    interval: float | str | None = None,
    conformal_level: float = 0.9,
    covariance_mode: str | None = None,
    observation_interval: bool = False,
    return_type: str | None = None,
    id_column: str | None = None,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input matching the training schema. |
| `interval` | `None` | Single uncertainty knob. `None` returns point predictions only; a float in `(0, 1)` (e.g. `0.95`) requests the full uncertainty decomposition at that pointwise coverage. `"conformal"` requests exact jackknife+ intervals for eligible Gaussian-identity fits; `"full_conformal"` requests the exact full-conformal set. On standard GLMs / location-scale this populates `std_error`, `mean_lower`, `mean_upper`. On supported single-event survival modes it populates `survival_se` and `eta_se`. On competing-risks survival it populates SE/lower/upper arrays for every cause-specific hazard, survival, cumulative hazard, CIF, overall survival, and eta surface. |
| `conformal_level` | `0.9` | Marginal coverage for `interval="conformal"` or `"full_conformal"`. Ignored for numeric Wald intervals. |
| `covariance_mode` | `None` | Python accepts `"conditional"` or `"smoothing"` for interval covariance. `None` requires smoothing-corrected covariance and errors when the fit cannot supply it. Competing-risks predictions expose the resolved source as `covariance_source`; current cause-specific fits provide the full joint conditional covariance, so callers must request `"conditional"` until the fitter produces a smoothing correction. The CLI uses the equivalent `--covariance-mode conditional|corrected` names. |
| `observation_interval` | `False` | When `True` and `interval` is numeric, adds response-scale prediction interval columns for families with an observation variance. |
| `return_type` | `None` | One of `"dict"`, `"numpy"`, `"pandas"`, `"polars"`, `"pyarrow"` for table-shaped outputs. Defaults to the input table kind, falling back to the training table kind. |
| `id_column` | `None` | Name of a column in `data` whose stringified values are carried through into table outputs and `SurvivalPrediction`. |

When the table target is `"dict"` (either explicitly or because no input /
training table kind determines a richer container), the returned object is a
`PredictionResult`: it is still a normal mapping, so `pred["mean"]` works, and
it also exposes prediction columns as attributes such as `pred.mean`,
`pred.std_error`, `pred.mean_lower`, and `pred.mean_upper`. For convenience,
`pred.lower`, `pred.upper`, and `pred.se_mean` alias the interval columns
`mean_lower`, `mean_upper`, and `std_error`.
For model-based intervals, a dict-shaped result also carries the scalar
`covariance_source` provenance field (`"conditional"` or
`"smoothing-corrected"`); pandas results expose the same value in
`result.attrs["covariance_source"]`.

## Return value by model class

| Model class | Default return | Columns / fields |
| --- | --- | --- |
| Gaussian, binomial, Poisson, negative-binomial, Gamma, Beta, Tweedie | 1-D `numpy.ndarray` | Response-scale point predictions. Table form has `linear_predictor`, `mean`; adds `std_error`, `mean_lower`, `mean_upper` when `interval` is set. |
| Gaussian / binomial / dispersion location-scale | 1-D `numpy.ndarray` | Response-scale point predictions. Table form has `linear_predictor`, `mean`, `noise_scale`; adds `std_error`, `mean_lower`, `mean_upper` when `interval` is set. |
| Transformation-normal | 1-D `numpy.ndarray` | Per-row response-scale conditional mean `E[Y|x]` (issue #1612). |
| Bernoulli marginal-slope | 1-D `numpy.ndarray` | Per-row probabilities clipped to `[0, 1]`. Table form has `mean`; with `interval=` it also includes `linear_predictor`, `std_error`, `mean_lower`, and `mean_upper`. |
| Survival (any likelihood mode) | `SurvivalPrediction` | Per-row hazard / survival evaluators. |
| Competing-risks survival | `CompetingRisksPrediction` | Endpoint-stacked hazard, survival, CIF, and overall survival arrays. |

For all point-payload classes, passing `return_type=`, `id_column=`, or
numeric `interval=` switches output to a table. Transformation-normal
uses `"z"` as the value column; bernoulli marginal-slope uses `"mean"`
for probabilities. Passing `id_column=` adds that stringified id column
first.

## Wald intervals

```python
preds = model.predict(test_df, interval=0.95)
# columns: linear_predictor, mean, std_error, mean_lower, mean_upper

pred_dict = model.predict(test_df, interval=0.95, return_type="dict")
mu = pred_dict["mean"]       # mapping access
mu_attr = pred_dict.mean     # same column on PredictionResult output
lo = pred_dict.mean_lower    # also available as pred_dict.lower
```

Intervals are computed from the asymptotic covariance of the fitted
coefficients propagated through the inverse link.

For response-scale prediction intervals, also pass
`observation_interval=True`:

```python
preds = model.predict(test_df, interval=0.95, observation_interval=True)
# adds observation_lower, observation_upper when the family supports it
```

For eligible Gaussian-identity models, use conformal intervals:

```python
preds = model.predict(test_df, interval="conformal", conformal_level=0.95)
full = model.predict(test_df, interval="full_conformal", conformal_level=0.95)
```

## Split-conformal intervals

`Model.predict_conformal(...)` runs the standard predictor on `data`, then
replaces the response-scale `mean_lower` / `mean_upper` columns with the
split-conformal interval `mu_hat(x) ± q_hat · s(x)` calibrated from a
held-out labeled `calibration` fold. The interval carries finite-sample
marginal coverage `≥ conformal_level` regardless of model misspecification,
and applies to standard GAM models (not only the Gaussian-identity fits the
jackknife+ `interval="conformal"` path requires).

```python
preds = model.predict_conformal(
    test_df,
    calibration=cal_df,        # held-out fold; MUST include the response column
    conformal_level=0.9,
)
# columns: linear_predictor, mean, std_error, mean_lower, mean_upper
```

`calibration` must contain the response column in addition to the predictors
(the conformal multiplier `q_hat` is computed from its held-out residuals).
It may be any size and is independent of the training set. `covariance_mode`,
`observation_interval`, `return_type`, and `id_column` behave as in
`predict`. `conformal_level` is required (no default).

## Predicting from a numeric array

For models fitted via `gamfit.fit_array(...)` (positional columns
`x0, x1, ..., x{p-1}`), predict directly from a numeric feature matrix:

```python
y = model.predict_array(X)                       # 1-D ndarray of point predictions
table = model.predict_array(X, interval=0.95)    # adds std_error / mean_lower / mean_upper
```

`predict_array` accepts `interval`, `covariance_mode`, and
`observation_interval` (same semantics as `predict`); it does not take
`return_type` or `id_column`. It is rejected for models fitted from a named
table — call `predict` with a `dict` / DataFrame there so columns match by
name. The companion `model.design_matrix_array(X)` returns the same typed
[`AffineDesign`](#fitted-affine-design) contract as `design_matrix`.

## Carrying an identifier column

```python
preds = model.predict(
    [
        {"patient_id": "P001", "x": 1.5},
        {"patient_id": "P002", "x": 2.5},
    ],
    id_column="patient_id",
    return_type="dict",
)
# preds = {"patient_id": ["P001", "P002"], "linear_predictor": [...], "mean": [...]}
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

```python
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
affine = model.design_matrix(test_df)
fitted_eta = affine.offset + affine.matrix @ affine.coefficients
```

The object has six explicit fields:

- `offset`: one value per row;
- `matrix`: the materialised matrix in the named coefficient frame;
- `coefficients`: the exact fitted vector multiplied by `matrix`;
- `coefficient_frame`: `"full"` or `"link_wiggle"`;
- `coefficient_start` / `coefficient_stop`: the represented half-open slice in
  that frame (`coefficient_slice` exposes the corresponding Python `slice`).

For an ordinary GAM, `offset` is the model offset, `matrix` is the full saved
design (including deployment extensions), and the coefficient frame is
`"full"`. Posterior draws use that same frame, so custom fitted-linear-predictor
draws are:

```python
affine = model.design_matrix(test_df)
posterior = model.sample(train_df)
eta_draws = affine.offset + posterior.samples @ affine.matrix.T
```

For a link-wiggle fit, the final predictor at the fitted state is
`base + B(warp_index) @ beta_w`. Accordingly, `offset` is the saved fitted base
predictor, `matrix` is `B` evaluated at the exact saved warp index (including
the frozen-index shift used by the fit), and `coefficient_frame` is
`"link_wiggle"`. The returned `coefficients` are the exact saved
standard-basis prediction coordinates. They can be a lift of the identifiable
reduced coordinates used by the joint optimizer, so do not slice a joint
posterior sample vector by shape and treat it as this frame.

Exact scan smoothers and coupled multi-surface model classes do not possess one
finite affine coefficient frame; `design_matrix` rejects them with a typed,
actionable error instead of fabricating a matrix.

## Difference-smooth contrasts

Use `Model.difference_smooth(view="x", group="group", data=data)` for covariance-aware pairwise smooth differences and optional simultaneous bands. See [Difference smooths](difference-smooths.md) for parameterisation choices and interval interpretation.
