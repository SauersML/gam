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

numpy is the only required dependency. pandas, polars and pyarrow are
recognised only when the caller has already imported them (gamfit checks
`sys.modules` and never imports a table library itself), and pandas frames need
no pyarrow: their columns cross as NumPy arrays. Polars and pyarrow tables
cross through the Arrow C stream they export.

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

Every column crosses the FFI boundary in the layout its source declares: a
numeric NumPy or pandas column as `float64` (nullable `Int64`/`boolean` NA
becomes a missing cell), a pandas categorical as codes plus its levels, and a
string, object or plain-Python column as its raw values. Polars and pyarrow
tables are decoded from Arrow memory directly. The engine then applies one rule
to every untyped column: it is a factor if any cell is a string, otherwise
numeric. A cell that is neither a number, a string nor missing (a date, a
timestamp, a complex number, an arbitrary object) raises `gamfit.errors.DataError`
naming its type, row and column, whichever library it came from. A column
whose declared dtype is neither numbers nor labels (NumPy or pandas
`datetime64`, `timedelta64`, `complex`; an Arrow date or timestamp) raises
`gamfit.errors.DataError` naming the column; encode such values as numbers first.
Labels are trimmed on every path, and a label of only whitespace raises
`gamfit.errors.DataError` naming its row and column.
Table normalization itself enforces only shape rules, because it runs before
any formula is known:

- Column lengths must agree. Mismatches raise `ValueError` before the engine
  sees the data.
- Tables must have at least one column and at least one row.
- Duplicate column names in pandas, polars, and pyarrow inputs are rejected,
  because prediction columns are matched by name.

Booleans become `1` / `0`. String columns are accepted for terms like
`group(site)` and are encoded by the engine as factor levels; a string column
whose labels happen to parse as numbers (`"0"`, `"1"`, `"2"`) is still a
factor, one level per label. The engine handles numeric coercion; explicit
casting to float is unnecessary.

## Missing data

A missing cell — `NaN`, `±inf`, `None`, or an empty string — is preserved by
table normalization and refused **only where a term consumes it**. The
refusal names the role, the column and the 1-based row, for example
`response column 'y' contains non-finite value at row 42` or
`model term column 'x' contains a non-finite value at row 42`. A column the
formula never references may carry any number of missing cells without
affecting the fit, so a frame such as R's `airquality` fits `Temp ~ s(Wind)`
on all 153 rows even though `Ozone` has 37 NAs (#2775). `Model.check(data)`
reports a missing cell in a modelled column as a `non_finite` issue rather
than raising (#2776).

`gamfit` does not impute. When a modelled column has gaps, drop or impute
those rows upstream:

- `df.dropna(subset=[...])` in pandas, listing the modelled columns.
- `sklearn.impute` or equivalent.
- For survival, the entry/exit/event columns must be complete.

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
| `"numpy"` | Structured `numpy.ndarray` of shape `(n_samples,)` with one named field per prediction column, the same names as the DataFrame result. |
| `"pandas"` | `pandas.DataFrame`. |
| `"polars"` | `polars.DataFrame`. |
| `"pyarrow"` | `pyarrow.Table`. |

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
model = gamfit.fit(pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}), "y ~ s(x)")
test_df = pd.DataFrame({"x": [1.5, 2.5, 3.5]})

pred = model.predict(test_df, return_type="dict")
pred["posterior_mean"]
pred.posterior_mean
table = model.predict(test_df, interval=0.95, return_type="numpy")
table["posterior_mean_lower"]
model.predict(test_df, return_type="pandas")
```

A positional NumPy array passed to `predict` for a model fitted on a named
table binds to the model's predictor columns in training-table order when its
width equals their count; any other width raises
`gamfit.errors.SchemaMismatchError` naming the expected columns. A model fitted
on an array keeps reading its columns as `x0`, `x1`, ….

## Array-returning model classes

Transformation-normal models and Bernoulli marginal-slope models return
a 1-D `numpy.ndarray` of shape `(n_samples,)` by default. Passing
`id_column=` or `return_type=` switches them to tabular output.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 4, 300)
train_df = pd.DataFrame({"x": x, "y": np.sin(x) + rng.gamma(2.0, 0.3, 300)})   # right-skewed noise
test_df = pd.DataFrame({"patient": ["P001", "P002", "P003"], "x": [0.5, 2.0, 3.5]})

model = gamfit.fit(train_df, "y ~ s(x)", transformation_normal=True)

# 1-D numpy by default: the response-scale conditional mean E[Y|x]
mean = model.predict(test_df)                    # shape (n,)

# Two-column table when id_column is set
df = model.predict(test_df, id_column="patient", return_type="pandas")
mean = df["mean"].to_numpy()
```

The value column is named `mean` for both classes: the response-scale
conditional mean for transformation-normal output and the probability for
Bernoulli marginal-slope output. The latent score of labelled data comes
from `Model.transformation_score`. Passing `return_type=`
without `id_column=` produces a one-column table; including
`id_column=` adds the id column. Extract the value column explicitly
when a 1-D array is wanted.

## Identifier columns

A column that is not part of the model can be carried through to the
output by naming it with `id_column=`:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
model = gamfit.fit({"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}, "y ~ s(x)")

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

The id column is excluded from the model and may hold numbers or strings.
