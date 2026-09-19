# scikit-learn integration

`gamfit.sklearn` exposes two scikit-learn estimators that wrap
`gamfit.fit`:

- `GAMRegressor` (inherits `RegressorMixin`) — continuous responses.
- `GAMClassifier` (inherits `ClassifierMixin`) — binary classification.

Install with `pip install gamfit[sklearn]`.

## GAMRegressor

```python
from gamfit.sklearn import GAMRegressor
import pandas as pd
import numpy as np

X = pd.DataFrame({"x": np.linspace(0, 10, 50)})
y = 2 * X["x"] + np.random.normal(0, 0.5, len(X))

est = GAMRegressor(formula="y ~ s(x)")
est.fit(X, y)

preds = est.predict(X)        # ndarray, shape (n,)
r2    = est.score(X, y)       # r2_score
```

### Constructor

```text
GAMRegressor(
    formula: str,
    family: str = "auto",
    offset: str | None = None,
    config: dict[str, Any] | None = None,
)
```

All four arguments are surfaced as `get_params()` keys, so they work with
`GridSearchCV` and related utilities. Per-row weights are data, not a
hyperparameter: pass them as `fit(X, y, sample_weight=w)`.

### Binding the response

If the formula contains `~`, the LHS column is the response unless `y` is a
string, in which case `y` names the response column already present in `X`
and replaces the formula LHS for fitting. If `y` is an array-like, it is
bound to `X` under the response name implied by the formula (defaulting to
`y`). If `y` is `None`, `X` must already contain the response.

```python
from gamfit.sklearn import GAMRegressor

GAMRegressor(formula="y ~ s(x)").fit(X, y)        # array y
GAMRegressor(formula="y ~ s(x)").fit(df)          # df contains "y"
GAMRegressor(formula="y ~ s(x)").fit(df, y="y")   # name a column
```

If the formula has no `~`, `y` must be supplied as an array-like target or
response-column name, and gamfit prepends `<target> ~`.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y=None, sample_weight=None)` | `self` | Sets `model_`, `formula_`, `n_features_in_`, and `feature_names_in_` when `X` names its columns. `sample_weight` becomes the likelihood's prior weights. |
| `predict(X)` | `ndarray (n,)` | Predicted mean. |
| `score(X, y, sample_weight=None)` | `float` | `r2_score`. |
| `summary()` | `Summary` | Delegates to `model_.summary()`. |
| `check(X)` | `SchemaCheck` | Delegates to `model_.check()`. Scalar models only. |
| `report(path)` | `str` | Delegates to `model_.report(path)`. Scalar models only. |

The fitted model is available at `est.model_` for access to the full
`gamfit.Model` API (`sample`, `predict(..., interval=...)`, etc.) when
the fit is a scalar GAM.

## GAMClassifier

```python
from gamfit.sklearn import GAMClassifier
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": np.linspace(0, 10, 50)})
y = (X["x"] + rng.normal(0, 2, len(X)) > 5).astype(int)   # binary labels

est = GAMClassifier(formula="y ~ s(x)", family="binomial")
est.fit(X, y)

probs = est.predict_proba(X)   # (n, 2): [P(classes_[0]), P(classes_[1])]
hard  = est.predict(X)         # (n,), highest-probability class label
acc   = est.score(X, y)        # accuracy
```

`classes_` is the sorted pair of labels observed at fit time. The wrapper
encodes `classes_[1]` as the positive class before fitting, so string
labels and non-`{0, 1}` binary labels round-trip. `predict_proba()` clips
the positive-class probability to `[0, 1]` and stacks
`[P(classes_[0]), P(classes_[1])]`. `predict()` returns
`classes_[argmax(predict_proba(X), axis=1)]`.

`score(X, y, sample_weight=None)` is accuracy, as for every scikit-learn
classifier; use `scoring="roc_auc"` in `cross_val_score` / `GridSearchCV`
for AUC. Use `metrics(X, y)` for the full panel: `auc`, `pr_auc`,
`brier`, `logloss`, `nagelkerke_r2`, and `ece`. Only binary targets are
supported: the estimator declares `classifier_tags.multi_class = False`
and rejects a multiclass `y` with a `ValueError`.

Like `GAMRegressor`, `GAMClassifier` also inherits the pass-through helpers
`summary()`, `check(X)`, and `report(path)` from the shared base estimator,
delegating to the underlying `gamfit.Model` (scalar models only).

## Pipeline

```python
from gamfit.sklearn import GAMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gam",    GAMRegressor(formula="y ~ s(x0) + s(x1)")),
])
pipe.fit(X, y)
preds = pipe.predict(X_test)
```

The GAM step accepts a `pandas.DataFrame`, `polars.DataFrame`,
`pyarrow.Table`, numpy array, dict of columns, list of records, or 2-D row
sequence. Numpy arrays and 2-D row sequences use generated feature names
`x0`, `x1`, ...

## Input validation

The estimators follow the scikit-learn estimator contract and pass
`sklearn.utils.estimator_checks.check_estimator`:

- Named inputs (DataFrames, Arrow tables, dicts, records) set
  `feature_names_in_`; at `predict` the columns must match those names in
  the same order (a response column carried over from fit is ignored).
- Unnamed inputs are validated with `sklearn.utils.check_array` (numeric,
  finite, dense, 2-D) and must have `n_features_in_` columns at `predict`.
  Sparse matrices and arrays are refused with a `TypeError` naming sparse
  input.
- A formula that reads a column `X` does not have (for example `x0 + x1`
  against a one-column array) is a `ValueError` naming the column and the
  features `X` has.
- `sample_weight` must be non-negative, one weight per row, and contain at
  least one non-zero weight. It is the likelihood's prior weight, the same as
  `gamfit.fit(..., weights=...)`, and may be fractional. For the binomial
  classifier an integer weight `k` is exactly `k` repeated rows. For a
  Gaussian regressor it is a precision (`y_i ~ N(mu_i, phi / w_i)`), so the
  REML scale estimate counts rows, not the weight total.
- A column-vector `y` of shape `(n, 1)` is ravelled with a
  `DataConversionWarning`; methods called before `fit` raise
  `NotFittedError`.

`GAMRegressor` sets `regressor_tags.poor_score = True`. The formula fixes
which columns the model reads, so scikit-learn's fixed scoring dataset (ten
columns, one of them informative) cannot be scored against a formula written
for other data; every other check runs unchanged.

## Cross-validation

```python
from gamfit.sklearn import GAMRegressor
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
```

## Grid search

```python
from gamfit.sklearn import GAMRegressor
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    GAMRegressor(formula="y ~ s(x)"),
    param_grid={
        "formula": ["y ~ s(x)", "y ~ s(x, k=10)", "y ~ s(x, k=20)"],
    },
    cv=5,
)
grid.fit(X, y)
```

## No survival wrapper

There is no sklearn wrapper for survival models. `Surv(...)` responses do
not match sklearn's `(X, y)` contract, and survival prediction produces a
per-time-grid surface rather than a single vector. Call `gamfit.fit(...)`
directly for survival; see [survival.md](survival.md).
