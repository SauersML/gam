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
| `list[Mapping[str, Any]]` | List of records. The full set of keys across rows defines the column order; each row must contain every key. |
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

String columns are accepted for terms like `group(site)` and are
encoded by the engine.

## Missing data

`gamfit` does not impute. Drop or impute rows upstream:

- `df.dropna(subset=[...])` in pandas.
- `sklearn.impute` or equivalent.
- For survival, ensure entry/exit/event columns are complete.

## What `predict()` returns

By default, the return container matches the input kind, falling back to
the training kind, then to `dict`. Override with `return_type=`:

| `return_type` | Returns |
| --- | --- |
| `None` (default) | Input kind, else training kind, else `dict`. |
| `"dict"` | `dict[str, list]`. |
| `"numpy"` | 2-D `numpy.ndarray` with columns in fixed order. |
| `"pandas"` | `pandas.DataFrame`. |
| `"polars"` | `polars.DataFrame`. |
| `"pyarrow"` | `pyarrow.Table`. |

```python
model.predict(test_df, return_type="dict")
model.predict(test_df, return_type="numpy")
model.predict(test_df, return_type="pandas")
```

## Array-returning model classes

Transformation-normal models and Bernoulli marginal-slope models return
a 1-D `numpy.ndarray` of shape `(n_samples,)` by default. Passing
`id_column=` or `return_type=` switches them to a two-column table.

```python
# 1-D numpy by default
z = model.predict(test_df)                       # shape (n,)

# Two-column table when id_column or return_type is set
df = model.predict(test_df, id_column="patient", return_type="pandas")
z = df["z"].to_numpy()                           # transformation-normal
```

The value column is named `z` for transformation-normal output and
`mean` for Bernoulli marginal-slope output. Flattening the two-column
table with `np.asarray(...)` produces a shape `(n, 2)` array; extract
the column explicitly when an array is wanted.

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
# preds = {"patient_id": ["P001", "P002"], "eta": [...], "mean": [...]}
```

The id column is excluded from the model and may be any type that
`stringify_cell` accepts.
