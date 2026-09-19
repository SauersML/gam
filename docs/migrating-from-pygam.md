# Migrating from pyGAM

This page is for people who already fit GAMs with
[pyGAM](https://pygam.readthedocs.io/). It explains the two changes in how
you think about a model, maps pyGAM calls to gamfit, and lists the places
where the same-looking code means something different.

## What changes

**Formulas on named columns, one `fit` for every family.** pyGAM builds
terms from column indices (`s(0) + f(2)`) and has one class per family
(`LinearGAM`, `LogisticGAM`, `PoissonGAM`, ...). gamfit takes a DataFrame
and a formula on its column names, and `gamfit.fit` covers every family.
The family is inferred from the response unless you pass `family=`.

**Smoothness is estimated, never searched.** In pyGAM, each term's `lam`
is a fixed number, 0.6 by default, and `gridsearch()` tries a grid of
values and keeps the one with the best GCV or UBRE score. gamfit has no
`lam`: it estimates every smoothing parameter by maximizing the REML (or,
outside the Gaussian family, LAML) marginal likelihood, in one
gradient-based optimization with exact derivatives. There is no grid to
choose and no grid to be too coarse, and a fit with twenty smooths costs
one optimization, not a grid whose size multiplies with every term.

REML is the criterion mgcv's author recommends over GCV (Wood, 2011,
*JRSS B* 73:3–36): the GCV score tends to be flat and multi-modal and
occasionally undersmooths badly, while the REML score is smoother and
more stable. The marginal likelihood also gives each smoothing parameter a
Bayesian reading: the penalty is a prior on the roughness of the function
itself (`∫ f''(x)² dx`), not on differences of spline coefficients.

**Predictions are posterior means with their uncertainty.** `predict`
returns the posterior mean on the response scale, and the same call can
return a credible band for it and an observation interval for a new
response. See [predictions](predictions.md).

The [benchmarks](benchmarks.md) page compares both libraries on accuracy
and speed, and lists the cases where pyGAM wins.

## A first model in both libraries

The ISLR `Wage` data, with smooths of age and year and a factor for
education. In pyGAM:

```python no-exec
import pandas as pd
from pygam import LinearGAM, s, f

wage = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Wage.csv")
X = wage[["age", "year"]].assign(education=wage["education"].astype("category").cat.codes).to_numpy()
y = wage["wage"].to_numpy()

gam = LinearGAM(s(0) + s(1) + f(2)).gridsearch(X, y)
mean = gam.predict(X)
lower, upper = gam.confidence_intervals(X, width=0.95).T
```

In gamfit:

```python
import pandas as pd
import gamfit

wage = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Wage.csv")

model = gamfit.fit(wage, "wage ~ s(age) + s(year, k=5) + education")
mean = model.predict(wage)
bands = model.predict(wage, interval=0.95)
lower, upper = bands["posterior_mean_lower"], bands["posterior_mean_upper"]

print(model.summary().smooth_terms_frame())
```

The string column `education` needs no encoding. `year` has only seven
distinct values, so `k=5` keeps its basis within what the data can
support; REML then shrinks `s(year)` to an almost straight line.

## Mapping table

| pyGAM | gamfit |
| --- | --- |
| `LinearGAM(terms).fit(X, y)` | `gamfit.fit(df, "y ~ ...")` |
| `LogisticGAM`, `PoissonGAM`, `GammaGAM` | inferred from the response, or `family="binomial"`, `"poisson"`, `"gamma"` |
| `ExpectileGAM(expectile=0.9)` | `family="expectile", expectile_tau=0.9` |
| `GAM(distribution="binomial", link="probit")` | `family="binomial-probit"` (also `-logit`, `-cloglog`) |
| `InvGaussGAM` | no inverse-Gaussian family; see the `family=` list in [formulas](formulas.md#response-left-of) |
| `s(0)` | `s(age)` |
| `s(0, n_splines=20)` | `s(age, k=20)`, an upper bound on flexibility; see [Choosing `k`](formulas.md#choosing-k) |
| `s(0, lam=0.6)`, `gam.gridsearch(X, y, lam=...)` | nothing: every smoothing parameter is estimated by REML/LAML |
| `s(0, spline_order=3)` | `s(age, degree=3)` |
| `s(0, constraints="monotonic_inc")` | `s(age, shape=monotone_increasing)`; also `monotone_decreasing`, `convex`, `concave` |
| `s(0, basis="cp")` | `cyclic(hour, period=24)` |
| `s(0, by=2)` | `s(age, by=group)` for a factor or a numeric `by` |
| `f(2)` | `education` (a string column) or `factor(education)`; both penalized, see [below](#categorical-terms-are-penalized) |
| `l(0)` | `age` or `linear(age)`, a penalized slope whose penalty REML estimates; `linear(age, min=0)` for a sign constraint |
| `te(0, 1)` | `te(lon, lat)` |
| `gam.fit(X, y, weights=w)` | `gamfit.fit(df, formula, weights="w")`, naming a column |
| `gam.predict(X)`, `gam.predict_mu(X)` | `model.predict(new_df)`, the posterior mean on the response scale |
| `gam.confidence_intervals(X, width=0.95)` | `model.predict(new_df, interval=0.95)`: `posterior_mean_lower`, `posterior_mean_upper` |
| `gam.prediction_intervals(X, width=0.95)` | `model.predict(new_df, interval=0.95, observation_interval=True)`: `observation_lower`, `observation_upper` |
| `gam.sample(X, y, n_draws=100)` | `model.sample(new_df, samples=100, seed=0)`: posterior draws, no bootstrap refits |
| `gam.summary()` | `model.summary()`; `.smooth_terms_frame()` for the per-term table |
| `gam.statistics_["edof"]` | `model.summary().edf_total` |
| `gam.statistics_["p_values"]` | `model.summary().smooth_terms_frame()["p_value"]` |
| `gam.deviance_residuals(X, y)` | `model.diagnose(df).residuals` |
| choosing between term sets by GCV | `gamfit.compare_models([fit_a, fit_b], names=[...])` |
| `LinearGAM` in scikit-learn pipelines | `gamfit.sklearn.GAMRegressor`, `GAMClassifier`; see [scikit-learn](sklearn.md) |
| `pygam.datasets.wage()` | no bundled data; the [tour](tour.md) reads public CSVs |

## What gamfit deliberately leaves out

- **`gridsearch()`, GCV and UBRE.** Smoothing parameters come from the
  REML/LAML marginal likelihood only. A grid search can only return a grid
  point, and its cost multiplies with the number of terms.
- **`lam=`.** A hand-set smoothing parameter overrides the estimate with a
  guess. To compare structurally different models, fit each one and pass
  them to `gamfit.compare_models`, which ranks them by AIC corrected for smoothing-parameter
  selection.
- **`n_splines` as the smoothness knob.** In pyGAM, 20 splines with a
  fixed `lam` fixes the flexibility. In gamfit `k` only caps it, and REML
  decides how much of the basis the data support. Raise `k` when
  `basis_check()` says the basis ran out, not to make a curve wigglier.

## Gotchas

### Categorical terms are penalized

pyGAM's `f()` puts a ridge penalty of fixed strength `lam` on the level
effects. In gamfit a string column, `factor(g)` and `group(g)` also put a
ridge penalty on them, but its strength is estimated by REML like every
other smoothing parameter. With plenty of rows per level the penalty is
negligible and the estimates match an unpenalized factor; with sparse
levels they are pulled toward the overall mean. A level that was not seen in training raises `gamfit.errors.GamError` for a
string column or `factor(g)`, and is predicted at the population level for
`group(g)`. The [formula reference](formulas.md#factor-terms) has the
details.

### String labels need `family="binomial"`

A 0/1 response is detected as binomial automatically. A response of
strings such as `"no"`/`"yes"` is not: pass `family="binomial"`. The
modelled event is the label that sorts last in plain string order, the
same class scikit-learn's `classes_[1]` names.

```python
import pandas as pd
import gamfit

mroz = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/carData/Mroz.csv")
model = gamfit.fit(mroz, "lfp ~ s(age) + s(inc) + k5 + wc", family="binomial")
print("P(lfp == 'yes') for the first rows:", model.predict(mroz.head()).round(3))
```

If the event sorts first (for example `"case"` against `"control"`), map
the column to `1`/`0` before fitting.

### Leave `k` unset: the data size the basis

pyGAM fixes the basis at `n_splines=20` and tunes only λ. Without `k=` or
`knots=`, a gamfit `s(x)` starts from a lean basis of about a dozen
functions and doubles its knot count while the fit shows the basis is too
small, until `basis_check` passes. Only the covariate's distinct values and
the design rank bound that growth. REML then decides how much of the basis
to use, so a null or linear effect still shrinks to about 0 or 1 edf.
Setting `k=` fixes the size instead. See
[Choosing `k`](formulas.md#choosing-k).

### Formulas name columns

Terms refer to DataFrame columns by name, so there is no `X` matrix to
assemble and no column order to keep in sync with the term list. A NumPy
array works through `gamfit.fit_array`, whose columns are named `x0`,
`x1`, ...
