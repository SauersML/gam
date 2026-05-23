# Diagnostics, summaries, plots, reports

A fitted `Model` exposes five inspection methods:

| Method | Returns | Contents |
| --- | --- | --- |
| `summary()` | `Summary` | Formula, family/link name, model class, deviance, REML/LAML score (in the `reml_score` field), per-coefficient table, smoothing parameters (`lambdas`), group metadata, and deployment extensions. |
| `diagnose(data)` | `Diagnostics` | Observed values, predicted columns, residuals, and aggregate metrics. |
| `check(data)` | `SchemaCheck` | Schema validation result with structured issues. |
| `plot(data, x=, kind=)` | `matplotlib.axes.Axes` | Prediction / residual / observed-vs-predicted plot. |
| `report(path=None)` | `str` | Self-contained HTML report (string, or written path). |

`gamfit.validate_formula(...)` validates a formula and data against the
parser and schema without fitting.

## summary()

```python
s = model.summary()
print(s)                       # text repr; HTML in notebooks
s["formula"]
s["family_name"]
s["model_class"]
s["deviance"]
s["reml_score"]
s["iterations"]
s["coefficients"]              # list of dicts (per-term records)
s.coefficients                 # same list via property
s.to_dict()                    # full payload as a dict
s.coefficients_frame()         # pandas.DataFrame; requires pandas
```

`Summary` supports `__getitem__` and `get(key, default)` like a dict.
The result is cached on the `Model` after the first call.

`model.smoothing_parameters()` is a thin helper returning a
`{penalty_index: lambda}` dict derived from `summary()["lambdas"]`.

## diagnose()

```python
diag = model.diagnose(data, *, y=None, interval=0.95)
```

`diagnose` calls `predict` on the feature columns of `data` with the
given interval, then packages the result against the observed
response.

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Table-like input that includes the response column. |
| `y` | `None` | Response column name. Defaults to `model.response_name`. Required when the formula does not name a single response (e.g. survival `Surv(...)` formulas). |
| `interval` | `0.95` | Coverage probability forwarded to `predict`. Set `None` to skip interval columns. |

Returns a frozen `Diagnostics` dataclass:

| Field | Type | Meaning |
| --- | --- | --- |
| `formula` | `str` | Fitted formula. |
| `response_name` | `str` | Resolved response column. |
| `observed` | `list[float]` | Observed values. |
| `residuals` | `list[float]` | `observed - predicted["mean"]`. |
| `predicted` | `dict[str, list[float]]` | The predict-table columns (at least `"mean"`; with `interval` also `"mean_lower"`, `"mean_upper"`). |
| `metrics` | `dict[str, float]` | `n_obs`, `mae`, `rmse`, `bias`; adds `r_squared` when the response variance is positive. |
| `interval_lower`, `interval_upper` | `list[float] \| None` | Aliases for the matching `predicted` columns. |

`diagnose` raises `ValueError` when the response column cannot be
inferred or is missing from `data`.

## check()

```python
check = model.check(test_df)

if check.ok:
    preds = model.predict(test_df)
else:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
    check.raise_for_error()       # raises ValueError
```

`SchemaCheck.ok` is `True` when no issues were reported.
`bool(check)` returns `check.ok`. `SchemaIssue` has three fields:
`kind`, `column`, `message`; typical kinds include `missing_column`
and `type_mismatch`.

## validate_formula()

```python
v = gamfit.validate_formula(
    data,
    "y ~ s(x) + group(site)",
    family="auto",
    # additional kwargs mirror gamfit.fit(...)
)
v["formula"]
v["model_class"]
v["family_name"]
v["response_column"]
v.supported_by_python      # bool
```

Returns a `FormulaValidation` dataclass that wraps the parsed payload.
Accepts the same keyword arguments as `gamfit.fit` (family, offset,
weights, transformation/survival/baseline settings, link/logslope
formula, frailty, adaptive regularization, Firth, etc.) but does no
fitting.

## plot()

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model.plot(train_df, x="x", kind="prediction",            ax=axes[0])
model.plot(train_df, x="x", kind="residuals",             ax=axes[1])
model.plot(train_df, x="x", kind="observed_vs_predicted", ax=axes[2])
plt.tight_layout()
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `data` | required | Held-out data, with the response column present (same requirements as `diagnose`). |
| `x` | inferred | Feature column to put on the x-axis for `"prediction"`. Inferred when there is exactly one non-response column. |
| `y` | `None` | Response column override. |
| `interval` | `0.95` | Wald-band coverage for `"prediction"`. Ignored for the other kinds. |
| `kind` | `"prediction"` | One of `"prediction"`, `"residuals"`, `"observed_vs_predicted"`. |
| `ax` | `None` | Existing axes; a fresh one is created via `plt.subplots()` when omitted. |

| `kind` | Contents |
| --- | --- |
| `"prediction"` | Sorted mean curve over `x`, shaded Wald band when `interval` is set, observed scatter overlay. |
| `"residuals"` | Residuals vs predicted mean with a horizontal zero line. |
| `"observed_vs_predicted"` | Observed vs predicted with a `y = x` reference line. |

Returns the `matplotlib.axes.Axes` drawn on. Raises `ValueError` for
unknown `kind`, ambiguous `x`, or a missing `x` column. Requires
matplotlib (install `gamfit[plot]`).

## report()

```python
model.report("report.html")       # writes the file and returns its path
html = model.report()             # returns the HTML string
```

The report is a self-contained HTML document containing the summary
table, smooth-term plots (per smooth term in the model), and residual
diagnostics rendered by `src/report.rs`. The HTML is also used as
`Model._repr_html_` for notebook display.

## Inspecting the model object

```python
model.formula                   # str
model.family_name               # str, e.g. "Gaussian Identity"
model.model_class               # str, e.g. "standard", "survival marginal-slope"
model.is_survival               # bool
model.is_marginal_slope         # bool
model.is_transformation_normal  # bool
model.response_name             # str | None (None for survival Surv(...) formulas)
model.training_table_kind       # "pandas" | "polars" | "pyarrow" | "numpy" | "mapping" | "records" | "rows" | None
model.group_metadata            # dict | None, persisted per-group metadata
model.deployment_extensions     # tuple of dicts, no-refit group extensions
```

These are read-only properties.

## When something looks wrong

| Symptom | Try this |
| --- | --- |
| `diag.metrics["r_squared"]` low on training | The model is under-flexed. Raise `k` on smooths or add interactions via `te(...)` / multi-d smooths. |
| `rmse` low on training, high on test | Over-flexed. Reduce `k` or rely on the default complexity. |
| `diagnose()` raises about the response column | Pass `y="column_name"` explicitly. |
| `check()` reports `missing_column` | The prediction data is missing a required feature. |
| `predict` raises `SchemaMismatchError` | Run `check()` first to identify the offending column. |
| `predict` raises `PredictionError` | The fitted class is not supported in Python; the error message lists supported alternatives. |

See [exceptions.md](exceptions.md) for the full exception hierarchy.
