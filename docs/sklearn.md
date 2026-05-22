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

If the formula contains `~`, the LHS column is the response. If `y` is an
array-like, it is bound to `X` under the response name implied by the
formula (defaulting to `y`). If `y` is a string, it names a column already
present in `X`. If `y` is `None`, `X` must already contain the response.

```python
GAMRegressor(formula="y ~ s(x)").fit(X, y)        # array y
GAMRegressor(formula="y ~ s(x)").fit(df)          # df contains "y"
GAMRegressor(formula="y ~ s(x)").fit(df, y="y")   # name a column
```

If the formula has no `~`, gamfit prepends `<target> ~`.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y=None)` | `self` | Sets `model_`, `formula_`, `feature_names_in_`, `n_features_in_`. |
| `predict(X)` | `ndarray (n,)` | Predicted mean. |
| `score(X, y, sample_weight=None)` | `float` | `r2_score`. |
| `summary()` | `Summary` | Delegates to `model_.summary()`. |
| `check(X)` | `SchemaCheck` | Delegates to `model_.check()`. Scalar models only. |
| `report(path)` | `str` | Delegates to `model_.report(path)`. Scalar models only. |

The fitted `Model` is available at `est.model_` for access to the full
`gamfit.Model` API (`sample`, `predict(..., interval=...)`, etc.).

## GAMClassifier

```python
from gamfit.sklearn import GAMClassifier

est = GAMClassifier(formula="y ~ s(x)", family="binomial")
est.fit(X, y)

probs = est.predict_proba(X)   # (n, 2): [P(y=0), P(y=1)]
hard  = est.predict(X)         # (n,) int, threshold 0.5
acc   = est.score(X, y)        # accuracy_score
```

`classes_` is `np.array([0, 1])` after `fit()`. `predict_proba()` clips the
positive-class probability to `[0, 1]` and stacks `[1 - p, p]`. The
threshold for `predict()` is hardcoded to `0.5`.

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

The GAM step accepts a `pandas.DataFrame`, a numpy array, a dict of
columns, or a list of records.

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
