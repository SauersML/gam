# Diagnostics, summaries, plots, reports

A fitted `Model` exposes five tools for inspecting it:

| Method | Returns | What you get |
| --- | --- | --- |
| `summary()` | `Summary` | Fitted coefficients, family, deviance, REML score, iterations. |
| `diagnose(data)` | `Diagnostics` | Residuals + RMSE / MAE / R² / bias on a dataset. |
| `check(data)` | `SchemaCheck` | Validate new data against the training schema. |
| `plot(data, x=…, kind=…)` | matplotlib axes | Prediction / residual / observed-vs-predicted plots. |
| `report(path=None)` | HTML string or path | Standalone HTML summary with plots. |

`gamfit.validate_formula(...)` checks a formula before you fit.

## summary()

```python
s = model.summary()
print(s)                       # pretty-printed; HTML in notebooks
print(s["formula"])
print(s["deviance"])
print(s["reml_score"])
print(s["iterations"])
print(s["model_class"])
print(s["family_name"])
print(s["coefficients"])       # list of dicts: index, estimate, std_error
```

`Summary` behaves like a dict; `summary.to_dict()` returns the full payload.

If pandas is installed, `summary.coefficients_frame()` returns the
coefficients as a `DataFrame` with one row per coefficient.

## diagnose()

`diagnose()` computes residuals and accuracy metrics against observed data:

```python
diag = model.diagnose(train_df)
print(diag.metrics)
# {'n_obs': 200.0, 'mae': 0.18, 'rmse': 0.24, 'bias': -0.003,
#  'r_squared': 0.957}

print(diag.residuals[:5])
print(diag.predicted["mean"][:5])
print(diag.interval_lower[:5], diag.interval_upper[:5])
```

| Field | Type | Meaning |
| --- | --- | --- |
| `formula` | `str` | The fitted formula. |
| `response_name` | `str` | Response column inferred from formula. |
| `observed` | `list[float]` | Observed `y`. |
| `residuals` | `list[float]` | `observed − predicted_mean`. |
| `predicted` | `dict[str, list[float]]` | Columns: `eta`, `mean`, `mean_lower`, `mean_upper`. |
| `metrics` | `dict[str, float]` | `n_obs`, `mae`, `rmse`, `bias`, `r_squared` (when `total_ss > 0`). |
| `interval_lower`, `interval_upper` | `list[float] \| None` | Aliases for the interval columns. |

`y=` overrides the inferred response column. `interval=0.95` is the default;
pass `interval=None` to skip interval bounds.

## check()

Validate new data against the training schema before calling `predict()`:

```python
check = model.check(test_df)

if check.ok:
    preds = model.predict(test_df)
else:
    for issue in check.issues:
        print(issue.kind, issue.column, issue.message)
    check.raise_for_error()       # raises ValueError
```

`SchemaIssue` has three fields: `kind`, `column`, `message`. Typical kinds:
`missing_column`, `type_mismatch`. `SchemaCheck` is truthy when clean,
so `if model.check(df): ...` is the short form.

## validate_formula()

Equivalent to `check()` for the formula — runs the parser and schema
checks but does no fitting:

```python
v = gamfit.validate_formula(df, "y ~ s(x) + group(site)")
print(v["model_class"])         # "standard"
print(v["family_name"])         # "Gaussian Identity"
print(v["response_column"])     # "y"
print(v.supported_by_python)    # True
```

Use it in tests, CLI front-ends, or when constructing a formula
programmatically.

## plot()

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
model.plot(train_df, x="x", kind="prediction",            ax=axes[0])
model.plot(train_df, x="x", kind="residuals",             ax=axes[1])
model.plot(train_df, x="x", kind="observed_vs_predicted", ax=axes[2])
plt.tight_layout()
```

| `kind` | What it shows |
| --- | --- |
| `"prediction"` | Fitted curve with credible band, observed points overlaid. |
| `"residuals"` | Residuals vs. predicted mean. |
| `"observed_vs_predicted"` | Observed vs predicted with identity line. |

`x=` is required when the model has multiple features, so `plot` knows
which axis to use. `y=` overrides the inferred response column. Requires
`gamfit[plot]`.

## report()

A standalone HTML report:

```python
model.report("report.html")       # writes to disk; returns the path
html = model.report()             # returns the HTML string
```

The report contains the summary table, smooth visualisations, and
diagnose-style residual diagnostics for the data you pass in.

## Inspecting the model object

```python
model.formula                   # str
model.family_name               # str
model.model_class               # str
model.is_survival               # bool
model.is_marginal_slope         # bool
model.is_transformation_normal  # bool
model.response_name             # str | None
model.training_table_kind       # "pandas", "polars", "pyarrow", "dict", "records", "rows"
```

These are read-only properties. Use them to inspect a model loaded from
disk before calling `predict()`.

## When something looks wrong

| Symptom | Try this |
| --- | --- |
| `diag.metrics["r_squared"]` low on training | The model is under-flexed. Raise `k` on smooths, or add interactions via `te(...)` / multi-d smooths. |
| `rmse` low on training, high on test | Over-flexed. Reduce `k` or rely on the default automatic complexity. |
| `diagnose()` raises about response column | Pass `y=...` explicitly. |
| `check()` fails on `missing_column` | Your prediction data is missing a feature the model needs. |
| Predict raises `SchemaMismatchError` | Run `check()` first to identify which column. |
| Predict raises `PredictionError` | The fitted model's class isn't supported in Python; the error message names the supported alternatives. |

See [exceptions.md](exceptions.md) for the full exception hierarchy.
