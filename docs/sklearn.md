# scikit-learn integration

`gamfit.sklearn` exposes two scikit-learn estimators that wrap
`gamfit.fit`:

- `GAMRegressor` (inherits `RegressorMixin`) — continuous responses.
- `GAMClassifier` (inherits `ClassifierMixin`) — binary and multiclass
  classification.

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
    formula: str | None = None,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    config: dict[str, Any] | None = None,
)
```

All five arguments are surfaced as `get_params()` keys, so the estimator
works with `sklearn.base.clone`, `cross_val_score` and pipelines.

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

With no formula (`GAMRegressor().fit(X, y)`), the estimator fits the
automatic formula `y ~ .`: one term per column of `X`, chosen from the
column's type (see
[Every remaining column](formulas.md#every-remaining-column)). Every such
term is penalized and can shrink to zero. `formula_` holds the formula actually
fitted.

```python
from gamfit.sklearn import GAMRegressor

est = GAMRegressor().fit(X, y)
print(est.formula_)            # y ~ s(x)
```

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
import pandas as pd
import numpy as np

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": np.linspace(0, 10, 50)})
y = (X["x"] + rng.normal(0, 2, len(X)) > 5).astype(int)   # binary labels

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

The modelled event is therefore the label that sorts last: with labels
`"no"` and `"yes"`, `classes_` is `["no", "yes"]` and the second column of
`predict_proba` is `P(y == "yes")`. `gamfit.fit` follows the same rule for
a string response, which needs an explicit `family="binomial"` there:

```python
import numpy as np
import pandas as pd
import gamfit
from gamfit.sklearn import GAMClassifier

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": np.linspace(0, 10, 200)})
labels = np.where(X["x"] + rng.normal(0, 2, len(X)) > 5, "yes", "no")

est = GAMClassifier(formula="y ~ s(x)", family="binomial").fit(X, labels)
model = gamfit.fit(X.assign(y=labels), "y ~ s(x)", family="binomial")

print(est.classes_)                                   # ['no' 'yes']
print(np.allclose(est.predict_proba(X)[:, 1], model.predict(X), atol=1e-6))
```

If the event you care about sorts first (for example `"case"` against
`"control"`), recode the labels to `1`/`0` before fitting.

`score(X, y, sample_weight=None)` returns AUC, not accuracy, for a binary
model. A model with more than two classes has no single ROC curve, so its
score is accuracy. If `sample_weight` is supplied, rows with weight `<= 0` are dropped before
computing AUC. Use `metrics(X, y)` for the full panel: `auc`, `pr_auc`,
`brier`, `logloss`, `nagelkerke_r2`, and `ece`.

With three or more classes (or `family="multinomial"` at any class count),
`GAMClassifier` fits one joint multinomial-logit GAM: `K − 1` linear
predictors, each with its own smooths and REML-selected smoothing
parameters, estimated together. It is a single model, not `K` one-vs-rest
fits, so the class probabilities sum to 1 by construction.
`predict_proba(X)` has shape `(n, K)`, with column `j` for `classes_[j]`. A
binary family such as `family="binomial"` with more than two classes is an
error.

```python
import numpy as np
import pandas as pd
from gamfit.sklearn import GAMClassifier

rng = np.random.default_rng(1)
X = pd.DataFrame({"x": rng.uniform(0, 3, 300)})
species = np.array(["a", "b", "c"])[np.clip((X["x"] + rng.normal(0, 0.5, 300)).astype(int), 0, 2)]

clf = GAMClassifier(formula="y ~ s(x)").fit(X, species)
probs = clf.predict_proba(X)
print(clf.classes_, probs.shape)                      # ['a' 'b' 'c'] (300, 3)
print(np.allclose(probs.sum(axis=1), 1.0))            # True
```

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

## Cross-validation

```python
from gamfit.sklearn import GAMRegressor
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
```

`n_jobs=-1` runs the folds in parallel processes, each with its own full-width
thread pool; see [Threads](threads.md) for how that shares the CPUs.

## Choosing between formulas

The estimator has no hyperparameters to tune: REML chooses every
smoothing parameter inside `fit`, and `k` is only an upper bound on each
smooth's flexibility. To choose between structurally different formulas,
fit each one and rank the underlying models with `gamfit.compare_models`,
which scores them by AIC corrected for smoothing-parameter selection:

```python
import numpy as np
import pandas as pd
import gamfit
from gamfit.sklearn import GAMRegressor

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": rng.uniform(0, 1, 300), "z": rng.uniform(0, 1, 300)})
y = np.sin(2 * np.pi * X["x"]) + 4 * (X["z"] - 0.5) ** 2 + rng.normal(0, 0.3, len(X))

formulas = ["y ~ s(x)", "y ~ s(x) + z", "y ~ s(x) + s(z)"]
estimators = [GAMRegressor(formula=formula).fit(X, y) for formula in formulas]
comparison = gamfit.compare_models([est.model_ for est in estimators], names=formulas)
print("winner:", comparison["winner"])
```

`cross_val_score` (above) still works when you want an out-of-sample
check of the winner.

## No survival wrapper

There is no sklearn wrapper for survival models. `Surv(...)` responses do
not match sklearn's `(X, y)` contract, and survival prediction produces a
per-time-grid surface rather than a single vector. Call `gamfit.fit(...)`
directly for survival; see [survival.md](survival.md).
