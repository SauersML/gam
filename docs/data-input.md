# Data input formats

`gamfit` accepts almost any rectangular Python object as `data`. The Rust
engine works on (headers, rows-of-strings), and the Python layer takes care
of normalising whatever you pass in.

## Supported input types

| Input | Notes |
| --- | --- |
| `pandas.DataFrame` | Column names are taken from `df.columns`. |
| `polars.DataFrame` | Column names are taken from `df.columns`. |
| `pyarrow.Table` | Column names are taken from `table.column_names`. |
| `numpy.ndarray` (2-D) | Column names are auto-generated as `x0, x1, …`. |
| `dict[str, sequence]` | Keys are column names, values are 1-D sequences. |
| `list[dict[str, Any]]` (records) | Each dict is one row; key set must be consistent. |
| `list[list]` (2-D) | Columns are auto-named `x0, x1, …`. |

These are all equivalent inputs for a two-column dataset:

```python
import pandas as pd
import numpy as np

# pandas
pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

# pyarrow
import pyarrow as pa
pa.table({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

# dict of columns
{"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}

# list of records
[{"y": 1.0, "x": 0.0}, {"y": 2.0, "x": 1.0}, {"y": 3.0, "x": 2.0}]

# numpy (column names become x0, x1)
np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
```

## What `predict()` returns

By default, `predict()` returns the same format you trained on, when that
makes sense. Override with `return_type=`:

| `return_type` | Returns |
| --- | --- |
| `None` (default) | Match input or training kind (pandas in → pandas out). |
| `"dict"` | `dict[str, list]` (always available, no optional deps). |
| `"numpy"` | 2-D `numpy.ndarray` (columns in fixed order). |
| `"pandas"` | `pandas.DataFrame` (requires `gamfit[pandas]`). |
| `"polars"` | `polars.DataFrame` (requires polars). |
| `"pyarrow"` | `pyarrow.Table` (requires pyarrow). |

```python
model.predict(test_df, return_type="dict")
model.predict(test_df, return_type="numpy")
model.predict(test_df, return_type="pandas")
```

## Validation rules

Cells are stringified before they cross the FFI boundary. Two rules:

1. **No `None`, empty strings, or NaN** anywhere in the data. The engine
   rejects these at the input layer with a clear error. Drop or impute first.
2. **All columns the same length.** Mismatched lengths raise `ValueError`
   before the engine sees them.

Numeric coercion happens engine-side; you don't need to cast to float
yourself. Categorical columns (strings) are accepted by terms like
`group(site)` and are encoded by the engine.

## Missing data

`gamfit` does not impute. If your data has missing values:

- Drop rows: `df.dropna(subset=[…])`.
- Impute upstream with the tool of your choice (e.g. `sklearn.impute`).
- For survival, ensure entry/exit/event columns are complete.

## NumPy gotcha (transformation-normal & marginal-slope)

For model classes that return a 1-D z-score or probability rather than a
two-column table, `predict()` returns a `numpy.ndarray` of shape
`(n_samples,)` by default. Passing `return_type="pandas"` or an `id_column=`
opts back into a two-column table — be careful when you flatten:

```python
# 1-D numpy (default)
z = model.predict(test_df)                       # shape (n,)

# 2-column DataFrame (id_column opts in)
df = model.predict(test_df, id_column="patient", return_type="pandas")
z = df["z"].to_numpy()                           # extract the column
```

## Passing through identifier columns

If your data has a row-id column that isn't a model feature, name it with
`id_column=` and `gamfit` preserves it in the output:

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

The id column is not used by the model and may be of any type.
