# scikit-learn integration

`gamfit.sklearn` provides two estimators that play nicely with sklearn's
pipeline, cross-validation, and grid-search machinery:

- `GAMRegressor` — for continuous responses (extends `RegressorMixin`).
- `GAMClassifier` — for binary classification (extends `ClassifierMixin`).

Install with `pip install gamfit[sklearn]` so you get `scikit-learn` and
`numpy` as dependencies.

## GAMRegressor

```python
from gamfit.sklearn import GAMRegressor
import pandas as pd
import numpy as np

X = pd.DataFrame({"x": np.linspace(0, 10, 50)})
y = 2 * X["x"] + np.random.normal(0, 0.5, len(X))

est = GAMRegressor(formula="y ~ s(x)")
est.fit(X, y)

preds = est.predict(X)                           # ndarray of shape (n,)
r2    = est.score(X, y)                          # R²
```

### Constructor

```python
GAMRegressor(
    formula: str,                          # required
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    config: dict[str, Any] | None = None,
)
```

All five parameters are surfaced as sklearn `get_params()` keys, so
`GridSearchCV` and friends just work.

### Two ways to bind the response

If your formula uses `y ~ ...`, pass `y` separately to `fit()`:

```python
GAMRegressor(formula="y ~ s(x)").fit(X, y)
```

If your formula is RHS-only (e.g. `"s(x)"`), `gamfit` auto-prepends `"y ~"`
for you:

```python
GAMRegressor(formula="s(x)").fit(X, y)  # internally rewrites to "y ~ s(x)"
```

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y=None)` | `self` | Fits the model. |
| `predict(X)` | `ndarray (n,)` | Point predictions on the response scale. |
| `score(X, y, sample_weight=None)` | `float` | `sklearn.metrics.r2_score`. |
| `summary()` | `Summary` | Delegates to `model_.summary()`. |
| `check(X)` | `SchemaCheck` | Delegates to `model_.check()`. |
| `report(path=None)` | `str` | Delegates to `model_.report()`. |

After `fit()`, the underlying `Model` is at `est.model_` so you can drop
down to the full `gamfit.Model` API at any time:

```python
est.model_.sample(X, seed=42)
est.model_.predict(X, interval=0.95)
```

## GAMClassifier

```python
from gamfit.sklearn import GAMClassifier

est = GAMClassifier(formula="y ~ s(x)", family="binomial")
est.fit(X, y)

probs = est.predict_proba(X)        # (n, 2): P(y=0), P(y=1)
hard  = est.predict(X)              # (n,):  0 or 1, threshold 0.5
acc   = est.score(X, y)             # accuracy
```

`classes_` is set to `np.array([0, 1])` after `fit()` for sklearn
compatibility. `predict_proba()` clips to `[0, 1]` and column 0 is `1 - p`.

The threshold for `predict()` is fixed at `0.5`. If you want a different
threshold, use `predict_proba()` and threshold manually.

## In a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from gamfit.sklearn import GAMRegressor

# Standardise features before fitting the GAM
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gam",    GAMRegressor(formula="y ~ s(x0) + s(x1)")),
])
pipe.fit(X, y)
preds = pipe.predict(X_test)
```

The GAM step accepts whatever the previous step emits — `pandas.DataFrame`,
numpy array, dict-of-columns are all fine.

## Cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
print(scores.mean(), "±", scores.std())
```

For classification:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    GAMClassifier(formula="y ~ s(x)"),
    X, y, cv=5, scoring="roc_auc",
)
```

## Grid search

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    GAMRegressor(formula="y ~ s(x)"),
    param_grid={
        "formula": [
            "y ~ s(x)",
            "y ~ s(x, k=10)",
            "y ~ s(x, k=20)",
        ],
    },
    cv=5,
)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
```

## No GAMSurvival wrapper (yet)

There is no sklearn-style wrapper for survival models — `Surv(...)` doesn't
fit the `(X, y)` contract that sklearn expects. For survival, call
`gamfit.fit(...)` directly and use the `SurvivalPrediction` object returned
by `Model.predict(...)`. See [survival.md](survival.md).
