# Predictions

`Model.predict(...)` is the single prediction entry point. The return
shape depends on the fitted model class and on the keyword arguments
`interval`, `id_column`, `return_type`, and `with_uncertainty`.

## Signature

```python
model.predict(
    data,
    *,
    interval: float | None = None,
    return_type: str | None = None,
    id_column: str | None = None,
    with_uncertainty: bool = False,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input matching the training schema. |
| `interval` | `None` | Pointwise Wald-interval coverage probability in `(0, 1)`. Honored by standard GLM families (Gaussian, binomial, Poisson, Gamma) and Gaussian/binomial location-scale models. Ignored by survival, transformation-normal, and bernoulli marginal-slope predictions. |
| `return_type` | `None` | One of `"dict"`, `"numpy"`, `"pandas"`, `"polars"`, `"pyarrow"`. Defaults to the input table kind, falling back to the training table kind. |
| `id_column` | `None` | Name of a column in `data` whose values are carried through into the output. |
| `with_uncertainty` | `False` | Survival location-scale models only. Populates `survival_se` and `eta_se` on the returned `SurvivalPrediction`. |

## Return value by model class

| Model class | Default return | Columns / fields |
| --- | --- | --- |
| Gaussian, binomial, Poisson, Gamma | Table | `eta`, `mean`; adds `effective_se`, `mean_lower`, `mean_upper` when `interval` is set. |
| Gaussian / binomial location-scale | Table | `eta`, `mean`; adds `effective_se`, `mean_lower`, `mean_upper` when `interval` is set. |
| Transformation-normal | 1-D `numpy.ndarray` | Per-row conditional z-scores. |
| Bernoulli marginal-slope | 1-D `numpy.ndarray` | Per-row probabilities clipped to `[0, 1]`. |
| Survival (any likelihood mode) | `SurvivalPrediction` | Per-row hazard / survival evaluators. |
| Competing-risks survival | `CompetingRisksPrediction` | Endpoint-stacked hazard, survival, CIF, and overall survival arrays. |

For the array-returning classes (transformation-normal and bernoulli
marginal-slope), passing `id_column=` or `return_type=` switches the
output to a two-column table: `(id_column, "z")` for transformation-normal
or `(id_column, "mean")` for bernoulli marginal-slope.

## Wald intervals

```python
preds = model.predict(test_df, interval=0.95)
# columns: eta, mean, effective_se, mean_lower, mean_upper
```

Intervals are computed from the asymptotic covariance of the fitted
coefficients propagated through the inverse link.

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
# preds = {"patient_id": ["P001", "P002"], "eta": [...], "mean": [...]}
```

The id column is not used by the model. It is copied through verbatim
and may be of any type.

## SurvivalPrediction

`Model.predict` returns a `SurvivalPrediction` dataclass for survival
families. The dense hazard/survival surface is evaluated by the Rust
core on a default time grid (derived from the entry/exit columns in
`data`) and stored on the returned object; the `*_at` helpers
interpolate that surface at arbitrary user times.

```python
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])        # (n_rows, 4) survival probabilities
F = pred.failure_at([10, 20])               # 1 - S
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
| `survival_se` | `numpy.ndarray \| None` | Delta-method standard error on `S(t)`. Populated only when `with_uncertainty=True` for the location-scale likelihood. |
| `eta_se` | `numpy.ndarray \| None` | Delta-method standard error on the linear predictor under the same conditions as `survival_se`. |
| `id_column`, `row_ids` | `str \| None`, `Sequence[str] \| None` | Set when `id_column=` was passed to `predict`. |

### Methods

```python
pred.hazard_at(times)              # (n_rows, len(times))
pred.survival_at(times)            # (n_rows, len(times))
pred.cumulative_hazard_at(times)   # (n_rows, len(times))
pred.failure_at(times)             # 1 - survival_at(times), clipped to [0, 1]
pred.survival_se_at(times)         # SE on S(t), or None if not computed
```

Each `times` argument is coerced to a 1-D array of finite non-negative
floats; an empty input is rejected.

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

For the location-scale survival likelihood, `with_uncertainty=True`
populates delta-method standard errors:

```python
pred = model.predict(test_df, with_uncertainty=True)
S = pred.survival_at([1, 5, 10])
se = pred.survival_se_at([1, 5, 10])

upper = (S + 1.96 * se).clip(0.0, 1.0)
lower = (S - 1.96 * se).clip(0.0, 1.0)
```

Other survival likelihood modes (transformation, Weibull, marginal-slope,
latent) reject `with_uncertainty=True` at the Rust boundary. For those,
use `Model.sample(...)` to draw posterior coefficients; note that
`PosteriorSamples.predict_draws(...)` is restricted to standard
non-link-wiggle GAMs (see `posterior-sampling.md`).

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

`cif.cif` has shape `(n_endpoints, n_rows, n_times)`. Endpoint names
are taken from the mapping keys, or supplied via `endpoint_names=` when
passing a sequence.

## Raw design matrix

For standard non-link-wiggle GAMs, `Model.design_matrix(data)` returns
the `(n_rows, n_coeffs)` matrix the engine uses for the linear
predictor:

```python
X = model.design_matrix(test_df)        # (n_rows, n_coeffs)
posterior = model.sample(train_df)
custom_eta = posterior.samples @ X.T    # (n_draws, n_rows)
```

Use this to compose your own posterior quantity that
isn't a straightforward `predict()` call. Restricted to standard
non-link-wiggle GAMs.

* [Difference smooths](difference-smooths.md)
