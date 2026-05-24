# Cookbook

Runnable recipes. Each one matches a pattern in the test suite.

## Fit a Gaussian GAM with intervals

```python
import gamfit
import pandas as pd

train = pd.DataFrame({
    "y": [1.2, 1.9, 3.1, 4.5, 5.2, 6.3],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
})

model = gamfit.fit(train, "y ~ s(x)")
preds = model.predict([{"x": 1.5}, {"x": 2.5}], interval=0.95)
# columns: eta, mean, effective_se, mean_lower, mean_upper
```

## Validate a formula before fitting

```python
v = gamfit.validate_formula(train, "y ~ s(x)")
assert v.supported_by_python
assert v["model_class"] == "standard"
assert v["family_name"] == "Gaussian Identity"
```

## Mixed input shapes

`fit` and `predict` accept pandas DataFrames, dict-of-columns, and
list-of-records:

```python
gamfit.fit(train_df, "y ~ s(x)")
gamfit.fit({"y": [1.0, 2.0], "x": [0.0, 1.0]}, "y ~ x")
gamfit.fit([{"y": 1.0, "x": 0.0}, {"y": 2.0, "x": 1.0}], "y ~ x")
```

## Binary classification with `GAMClassifier`

```python
from gamfit.sklearn import GAMClassifier

train = pd.DataFrame({
    "y": [0, 0, 1, 1, 1, 1],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
})

est = GAMClassifier(formula="y ~ s(x)", family="binomial")
est.fit(train)

probs = est.predict_proba([{"x": 1.5}, {"x": 3.5}])   # (2, 2)
pred  = est.predict([{"x": 1.5}, {"x": 3.5}])         # (2,) int
acc   = est.score(train[["x"]], train["y"])
```

## Random intercept per site

```python
train = pd.DataFrame({
    "outcome": [1.0, 1.5, 2.0, 1.8, 2.5, 3.0],
    "treatment": [0, 0, 0, 1, 1, 1],
    "site_id": ["A", "A", "A", "B", "B", "B"],
})

model = gamfit.fit(train, "outcome ~ treatment + group(site_id)")
```

## Non-negative and bounded coefficients

```python
gamfit.fit(df, "y ~ age + nonnegative(dose)")
gamfit.fit(df, "y ~ age + bounded(dose, min=0, max=1)")
gamfit.fit(df, "y ~ age + bounded(prop, min=0, max=1, target=0.5, strength=3)")
```

## Anisotropic spatial smooth

```python
gamfit.fit(
    df,
    "z ~ matern(pc1, pc2, pc3, pc4)",
    scale_dimensions=True,
)
```

## 4-D Duchon with three-part penalty

```python
gamfit.fit(
    df,
    "z ~ duchon(pc1, pc2, pc3, pc4, centers=50)",
    scale_dimensions=True,
)
```

## Cyclic 1-D smooth

```python
# Day-of-week (period = 7).
gamfit.fit(df, "y ~ s(dow, periodic=true, period=7)")

# Hour-of-day with explicit half-open domain.
gamfit.fit(df, "y ~ cyclic(hour, period_start=0, period_end=24)")

# Angles in radians; the DSL accepts `pi` / `tau` (case-insensitive),
# optionally multiplied by a single literal (e.g. `2*pi`, `.5*pi`).
gamfit.fit(df, "y ~ s(theta, periodic=true, period=2*pi)")
```

## Tensor product with a periodic axis

```python
# Cylinder: theta wraps, h is open.
gamfit.fit(df, "y ~ te(theta, h, periodic=[0], period=[2*pi, None])")

# Calendar surface: day-of-week × hour-of-day, both periodic.
gamfit.fit(df,
    "y ~ te(dow, hour, bc=['periodic','periodic'],"
    "       periods=[7, 24], origins=[0, 0])")

# Torus: both axes periodic.
gamfit.fit(df, "y ~ te(u, v, periodic=[0,1], period=[2*pi, 2*pi])")
```

## Intrinsic sphere smooth (lat / lon)

```python
# Wahba reproducing kernel: isotropic on S^2, no pole artefacts.
gamfit.fit(df, "y ~ sphere(lat, lon, radians=true)")

# Spherical harmonics; max_degree=L gives basis dim L(L+2).
gamfit.fit(df, "y ~ sphere(lat, lon, method=harmonic, max_degree=8, radians=true)")

# mgcv-style alias.
gamfit.fit(df, "y ~ s(lat, lon, bs=sos)")
```

## Manifold-valued response (simplex / sphere)

```python
# Simplex response (e.g. composition). Predictions are strictly positive
# and sum to 1 row-wise.
model = gamfit.fit(
    train,
    "composition ~ s(x)",          # LHS is a label; RHS is reused per coord
    response_geometry="simplex",   # or "alr"
    response_columns=["sand", "silt", "clay"],
)
pred = model.predict(test)         # columns: sand, silt, clay
```

```python
# Spherical response (e.g. surface normals). Predictions are unit-norm.
model = gamfit.fit(
    train,
    "direction ~ s(x)",
    response_geometry="spherical",
    response_columns=["nx", "ny", "nz"],
)
pred = model.predict(test)         # columns: nx, ny, nz
```

See [response-geometry.md](response-geometry.md) for the full discussion.

## Boundary-conditioned 1-D smooth

```python
# Both endpoints have zero first derivative.
gamfit.fit(df, "y ~ s(x, bc=clamped)")

# Pin the start to zero, leave the end free.
gamfit.fit(df, "y ~ s(x, bc_left=anchored, anchor_left=0)")
```

## Flexible link

```python
gamfit.fit(
    df,
    "case ~ s(age) + link(type=flexible(probit)) + linkwiggle(internal_knots=6)",
)
```

## Survival with a flexible baseline

```python
gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi + timewiggle(internal_knots=6)",
    survival_likelihood="transformation",
    baseline_target="weibull",
)
```

## Survival with parametric Gompertz baseline + frailty

```python
gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age)",
    survival_likelihood="latent",
    baseline_target="gompertz",
    baseline_rate=0.08,
    frailty_kind="hazard-multiplier",
    hazard_loading="full",
)
```

## Survival surface on a time grid

```python
model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="transformation",
)
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])      # (n_rows, 4)
H = pred.cumulative_hazard_at([10])       # (n_rows, 1)
F = pred.failure_at([10])                 # 1 - S
```

## Stream survival predictions to CSV

```python
pred.write_survival_at_csv(
    "surv.csv",
    times=[1, 5, 10, 20, 50, 100],
    people_chunk=50_000,
    time_grid_chunk=64,
)
```

## Two-stage marginal-slope pipeline

```python
# Stage 1: condition the score on PCs (transformation-normal).
calib = gamfit.fit(
    df,
    "PGS ~ matern(pc1, pc2, pc3, pc4, centers=20)",
    transformation_normal=True,
    scale_dimensions=True,
)
df["pgs_z"] = calib.predict(df)

# Stage 2: Bernoulli marginal-slope.
model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3, pc4, centers=20)",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="pgs_z",
    logslope_formula="matern(pc1, pc2, pc3, pc4, centers=20)",
    scale_dimensions=True,
)
probs = model.predict(test_df, return_type="dict")["mean"]
```

## Survival marginal-slope

```python
gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi) + s(hba1c)",
    survival_likelihood="marginal-slope",
    z_column="pgs_z",
    logslope_formula="s(bmi) + s(hba1c)",
)
```

## Joint mean and variance (location-scale)

```python
gamfit.fit(
    df,
    "y ~ s(x1) + s(x2)",
    config={"noise_formula": "s(x1)"},
)
```

## Pass through an identifier column

```python
preds = model.predict(
    [
        {"patient_id": "P001", "x": 1.5},
        {"patient_id": "P002", "x": 2.5},
    ],
    id_column="patient_id",
    return_type="dict",
)
# preds["patient_id"] is preserved verbatim.
```

## Posterior mean bands

```python
posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)
# columns: eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper
```

## Posterior of a derived quantity

```python
import numpy as np

posterior = model.sample(train, seed=42)
beta_t = posterior["beta_treatment"]      # (n_draws,)
or_draws = np.exp(beta_t)
print(f"OR = {or_draws.mean():.2f} "
      f"(95% CI {np.quantile(or_draws, 0.025):.2f}–"
      f"{np.quantile(or_draws, 0.975):.2f})")
```

## Save and reload

```python
model.save("model.gam")
posterior.save("posterior.npz")

m  = gamfit.load("model.gam")
ps = gamfit.load_posterior("posterior.npz")
```

## Catch schema errors

```python
def safe_predict(model, data):
    check = model.check(data)
    if not check.ok:
        for issue in check.issues:
            print(issue.kind, issue.column, issue.message)
        check.raise_for_error()
    return model.predict(data)
```

## sklearn cross-validation

```python
from sklearn.model_selection import cross_val_score
from gamfit.sklearn import GAMRegressor

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
```

## sklearn grid search over formulas

```python
from sklearn.model_selection import GridSearchCV
from gamfit.sklearn import GAMRegressor

gs = GridSearchCV(
    GAMRegressor(formula="y ~ s(x)"),
    param_grid={
        "formula": ["y ~ s(x)", "y ~ s(x, k=10)", "y ~ s(x, k=20)"],
    },
    cv=5,
)
gs.fit(X, y)
```

## HTML report

```python
model.report("report.html")     # writes to disk
html = model.report()           # returns the string for inline display
```

## Per-group trajectories (factor by smooth)

`y ~ fac + s(time, by=fac)` fits separate time trajectories by level; include the main `fac` effect for level offsets.

## Hierarchical / partial-pooling smooths (`bs="fs"`)

`y ~ s(time) + s(time, subject, bs="fs")` models a population curve plus shrinkage-stabilized subject-specific departures.

## Treatment vs control difference smooth (`bs="sz"`)

`y ~ s(time) + s(time, treatment, bs="sz")` estimates a population time effect and sum-to-zero treatment deviations.
