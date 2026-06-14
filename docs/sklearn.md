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

```python
GAMRegressor(
    formula: str,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    config: dict[str, Any] | None = None,
)
```

All five arguments are surfaced as `get_params()` keys, so they work with
`GridSearchCV` and related utilities.

### Binding the response

If the formula contains `~`, the LHS column is the response unless `y` is a
string, in which case `y` names the response column already present in `X`
and replaces the formula LHS for fitting. If `y` is an array-like, it is
bound to `X` under the response name implied by the formula (defaulting to
`y`). If `y` is `None`, `X` must already contain the response.

```python
GAMRegressor(formula="y ~ s(x)").fit(X, y)        # array y
GAMRegressor(formula="y ~ s(x)").fit(df)          # df contains "y"
GAMRegressor(formula="y ~ s(x)").fit(df, y="y")   # name a column
```

If the formula has no `~`, `y` must be supplied as an array-like target or
response-column name, and gamfit prepends `<target> ~`.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y=None)` | `self` | Sets `model_`, `formula_`, `feature_names_in_`, `n_features_in_`. |
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

est = GAMClassifier(formula="y ~ s(x)", family="binomial")
est.fit(X, y)

probs = est.predict_proba(X)   # (n, 2): [P(classes_[0]), P(classes_[1])]
hard  = est.predict(X)         # (n,), highest-probability class label
auc   = est.score(X, y)        # ROC AUC
```

`classes_` is the sorted pair of labels observed at fit time. The wrapper
encodes `classes_[1]` as the positive class before fitting, so string
labels and non-`{0, 1}` binary labels round-trip. `predict_proba()` clips
the positive-class probability to `[0, 1]` and stacks
`[P(classes_[0]), P(classes_[1])]`. `predict()` returns
`classes_[argmax(predict_proba(X), axis=1)]`.

`score(X, y, sample_weight=None)` returns AUC, not accuracy. If
`sample_weight` is supplied, rows with weight `<= 0` are dropped before
computing AUC. Use `metrics(X, y)` for the full panel: `auc`, `pr_auc`,
`brier`, `logloss`, `nagelkerke_r2`, and `ece`.

## Pipeline

```python
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

## Cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
```

## Grid search

```python
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
