# Data input formats

`gamfit.fit()` and `Model.predict()` accept several rectangular Python
inputs. The Python layer normalises the input to `(headers, rows)`
before crossing the Rust FFI boundary.

## Supported input types

| Input | Notes |
| --- | --- |
| `pandas.DataFrame` | Columns taken from `df.columns`. |
| `polars.DataFrame` | Columns taken from `df.columns`. |
| `pyarrow.Table` | Columns taken from `table.column_names`. |
| `numpy.ndarray` (1-D or 2-D) | Columns auto-named `x0`, `x1`, …. 1-D becomes a single column `x0`. |
| `Mapping[str, sequence]` | Keys are column names, values are 1-D sequences. |
| `Sequence[Mapping[str, Any]]` | Records. The full set of keys across rows defines the column order; each row must contain every key. |
| `Sequence[Sequence]` (2-D) | Columns auto-named `x0`, `x1`, …. All rows must have the same width. |

pandas/polars/pyarrow are detected at runtime via `_try_import`. They
are not required at install time.

Equivalent inputs for a two-column dataset:

```python
import pandas as pd
import numpy as np
import pyarrow as pa

pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
pa.table({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
{"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}
[{"y": 1.0, "x": 0.0}, {"y": 2.0, "x": 1.0}, {"y": 3.0, "x": 2.0}]
np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])  # columns become x0, x1
```

## Validation rules

Cells are stringified before crossing the FFI boundary. Two hard rules
enforced in `stringify_cell`:

1. `None` is rejected.
2. NaN floats and empty strings are rejected.

Booleans become `"1"` / `"0"`. Numbers use `repr()`. Other values are
passed through `str(...)`. The engine handles numeric coercion from
strings; explicit casting to float is unnecessary.

Column lengths must agree. Mismatches raise `ValueError` before the
engine sees the data.

Tables must have at least one column and at least one row. Duplicate
column names in pandas, polars, and pyarrow inputs are rejected because
prediction columns are matched by name.

String columns are accepted for terms like `group(site)` and are
encoded by the engine.

## Missing data

`gamfit` does not impute. Drop or impute rows upstream:

- `df.dropna(subset=[...])` in pandas.
- `sklearn.impute` or equivalent.
- For survival, ensure entry/exit/event columns are complete.

## What `predict()` returns

For standard scalar GAM/GLM models, `model.predict(data)` returns a 1-D
`numpy.ndarray` of response-scale point predictions by default.

Tabular output is returned when `interval=`, `id_column=`, or
`return_type=` is supplied. In that tabular path, `return_type=None`
mirrors the prediction input kind for pandas/polars/numpy/pyarrow inputs,
else the training kind, else `dict`. Override with `return_type=`:

| `return_type` | Returns |
| --- | --- |
| `None` | Tabular path only: input kind for pandas/polars/numpy/pyarrow inputs, else training kind, else `dict`. |
| `"dict"` | `PredictionResult`, a `dict[str, list]` with attribute access to prediction columns. |
| `"numpy"` | 2-D `numpy.ndarray` with columns in fixed order. |
| `"pandas"` | `pandas.DataFrame`. |
| `"polars"` | `polars.DataFrame`. |
| `"pyarrow"` | `pyarrow.Table`. |

```python
pred = model.predict(test_df, return_type="dict")
pred["mean"]
pred.mean
model.predict(test_df, return_type="numpy")
model.predict(test_df, return_type="pandas")
```

## Array-returning model classes

Transformation-normal models and Bernoulli marginal-slope models return
a 1-D `numpy.ndarray` of shape `(n_samples,)` by default. Passing
`id_column=` or `return_type=` switches them to tabular output.

```python
# 1-D numpy by default
z = model.predict(test_df)                       # shape (n,)

# Two-column table when id_column is set
df = model.predict(test_df, id_column="patient", return_type="pandas")
z = df["z"].to_numpy()                           # transformation-normal
```

The value column is named `z` for transformation-normal output and
`mean` for Bernoulli marginal-slope output. Passing `return_type=`
without `id_column=` produces a one-column table; including
`id_column=` adds the id column. Extract the value column explicitly
when a 1-D array is wanted.

## Identifier columns

A column that is not part of the model can be carried through to the
output by naming it with `id_column=`:

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

The id column is excluded from the model and may be any type that
`stringify_cell` accepts.
