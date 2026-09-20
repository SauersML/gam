# Cookbook

Runnable examples. Each one matches a pattern in the test suite.

## Fit a Gaussian GAM with intervals

```python
import gamfit
import pandas as pd

train = pd.DataFrame({
    "y": [1.2, 1.9, 3.1, 4.5, 5.2, 6.3, 7.1, 7.8, 8.0, 7.7, 7.2, 6.5],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
})

model = gamfit.fit(train, "y ~ s(x)")
preds = model.predict([{"x": 1.5}, {"x": 2.5}], interval=0.95)
# columns: linear_predictor_plugin, mean_plugin, posterior_mean,
#          posterior_mean_standard_error, posterior_mean_lower, posterior_mean_upper
```

## Validate a formula before fitting

```python
import gamfit

train = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
         "y": [1.2, 1.9, 3.1, 4.5, 5.2, 6.3, 7.1, 7.8, 8.0, 7.7, 7.2, 6.5]}

v = gamfit.validate_formula(train, "y ~ s(x)")
assert v.supported_by_python
assert v["model_class"] == "standard"
assert v["family_name"] == "Gaussian Identity"
```

## Mixed input shapes

`fit` and `predict` accept pandas DataFrames, dict-of-columns, and
list-of-records:

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 200)
train_df = pd.DataFrame({"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 200)})

gamfit.fit(train_df, "y ~ s(x)")
gamfit.fit({"y": [1.0, 2.0], "x": [0.0, 1.0]}, "y ~ x")
gamfit.fit([{"y": 1.0, "x": 0.0}, {"y": 2.0, "x": 1.0}], "y ~ x")
```

## Binary classification with `GAMClassifier`

```python
import pandas as pd
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
import gamfit
import pandas as pd

train = pd.DataFrame({
    "outcome": [1.0, 1.5, 2.0, 1.8, 2.5, 3.0],
    "treatment": [0, 0, 0, 1, 1, 1],
    "site_id": ["A", "A", "A", "B", "B", "B"],
})

model = gamfit.fit(train, "outcome ~ treatment + group(site_id)")
```

## Non-negative and bounded coefficients

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
n = 300
df = {"age": rng.uniform(20, 80, n), "dose": rng.uniform(0, 1, n), "prop": rng.uniform(0, 1, n)}
df["y"] = 0.02 * df["age"] + 1.5 * df["dose"] + 0.5 * df["prop"] + rng.normal(0, 0.3, n)

gamfit.fit(df, "y ~ age + nonnegative(dose)")
gamfit.fit(df, "y ~ age + bounded(dose, min=0, max=1)")
gamfit.fit(df, "y ~ age + bounded(prop, min=0, max=1, target=0.5, strength=3)")
```

## Anisotropic spatial smooth

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
u = rng.uniform(-1, 1, (300, 4))
df = {f"pc{j + 1}": u[:, j] * scale for j, scale in enumerate([1, 2, 5, 10])}
df["z"] = np.exp(-(u ** 2).sum(axis=1)) + rng.normal(0, 0.1, 300)

gamfit.fit(
    df,
    "z ~ matern(pc1, pc2, pc3, pc4)",
    scale_dimensions=True,
)
```

## 4-D Duchon with three-part penalty

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
u = rng.uniform(-1, 1, (300, 4))
df = {f"pc{j + 1}": u[:, j] * scale for j, scale in enumerate([1, 2, 5, 10])}
df["z"] = np.exp(-(u ** 2).sum(axis=1)) + rng.normal(0, 0.1, 300)

gamfit.fit(
    df,
    "z ~ duchon(pc1, pc2, pc3, pc4, centers=50)",
    scale_dimensions=True,
)
```

## Cyclic 1-D smooth

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
n = 400
df = {"dow": rng.integers(0, 7, n).astype(float), "hour": rng.uniform(0, 24, n), "theta": rng.uniform(0, 2 * np.pi, n)}
df["y"] = np.sin(2 * np.pi * df["dow"] / 7) + np.cos(2 * np.pi * df["hour"] / 24) + np.sin(df["theta"]) + rng.normal(0, 0.3, n)

# Day-of-week (period = 7).
gamfit.fit(df, "y ~ s(dow, periodic=true, period=7)")

# Hour-of-day with explicit half-open domain.
gamfit.fit(df, "y ~ cyclic(hour, period_start=0, period_end=24)")

# Angles in radians; the DSL accepts `pi` / `tau` (case-insensitive),
# optionally multiplied by a single literal (e.g. `2*pi`, `.5*pi`).
gamfit.fit(df, "y ~ s(theta, periodic=true, period=2*pi)")
```

## Intrinsic sphere smooth (lat / lon)

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
n = 400
df = {"lat": np.arcsin(rng.uniform(-1, 1, n)), "lon": rng.uniform(-np.pi, np.pi, n)}
df["y"] = np.sin(df["lat"]) + np.cos(df["lat"]) * np.cos(df["lon"]) + rng.normal(0, 0.2, n)

# Wahba reproducing kernel: isotropic on S^2, no pole artefacts.
gamfit.fit(df, "y ~ sphere(lat, lon, radians=true)")

# Spherical harmonics; max_degree=L gives basis dim L(L+2).
gamfit.fit(df, "y ~ sphere(lat, lon, method=harmonic, max_degree=8, radians=true)")

# Pseudo-spline kernel.
gamfit.fit(df, "y ~ s(lat, lon, bs=sos, method=pseudo)")
```

## Manifold-valued response (simplex / sphere)

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x = rng.uniform(0, 4, 300)
parts = np.exp(np.column_stack([0.4 * np.sin(x), 0.3 * np.cos(x), np.zeros(300)]) + rng.normal(0, 0.1, (300, 3)))
train = pd.DataFrame(parts / parts.sum(axis=1, keepdims=True), columns=["sand", "silt", "clay"]).assign(x=x)
test = pd.DataFrame({"x": [0.5, 2.0, 3.5]})

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
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
x = rng.uniform(0, 3, 300)
normals = np.column_stack([np.sin(x), np.cos(x), np.full(300, 0.5)]) + rng.normal(0, 0.05, (300, 3))
train = pd.DataFrame(normals / np.linalg.norm(normals, axis=1, keepdims=True), columns=["nx", "ny", "nz"]).assign(x=x)
test = pd.DataFrame({"x": [0.5, 1.5, 2.5]})

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
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 300)
df = {"x": x, "y": np.sin(np.pi * x) ** 2 + rng.normal(0, 0.1, 300)}

# Both endpoints have zero first derivative.
gamfit.fit(df, "y ~ s(x, bc=clamped)")

# Pin the start to zero, leave the end free.
gamfit.fit(df, "y ~ s(x, bc_left=anchored, anchor_left=0)")
```

## Flexible link

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
age = rng.uniform(30, 80, 500)
df = {"age": age, "case": (rng.uniform(size=500) < 1 / (1 + np.exp(-(age - 55) / 8))).astype(float)}

gamfit.fit(
    df,
    "case ~ s(age) + link(type=flexible(probit)) + linkwiggle(internal_knots=6)",
)
```

## Survival with a flexible baseline

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = 20 * rng.weibull(1.5, n) * np.exp(-0.03 * (age - 60) - 0.05 * (bmi - 27))
c = rng.uniform(5, 40, n)
df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c), "event": (t <= c).astype(float), "age": age, "bmi": bmi})

gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi + timewiggle(internal_knots=6)",
    survival_likelihood="transformation",
    baseline_target="weibull",
)
```

## Survival with parametric Gompertz baseline + frailty

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = 20 * rng.weibull(1.5, n) * np.exp(-0.03 * (age - 60) - 0.05 * (bmi - 27))
c = rng.uniform(5, 40, n)
df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c), "event": (t <= c).astype(float), "age": age, "bmi": bmi})

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
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = 20 * rng.weibull(1.5, n) * np.exp(-0.03 * (age - 60) - 0.05 * (bmi - 27))
c = rng.uniform(5, 40, n)
df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c), "event": (t <= c).astype(float), "age": age, "bmi": bmi})
test_df = df.head(5)

model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="transformation",
)
pred = model.predict(test_df)

S = pred.survival_at([1, 5, 10, 20])      # (n_rows, 4)
H = pred.cumulative_hazard_at([10])       # (n_rows, 1)
F = 1.0 - pred.survival_at([10])          # (n_rows, 1)
```

## Stream survival predictions to CSV

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
age, bmi = rng.uniform(40, 80, n), rng.normal(27, 4, n)
t = 20 * rng.weibull(1.5, n) * np.exp(-0.03 * (age - 60) - 0.05 * (bmi - 27))
c = rng.uniform(5, 40, n)
df = pd.DataFrame({"entry": 0.0, "exit": np.minimum(t, c), "event": (t <= c).astype(float), "age": age, "bmi": bmi})
test_df = df.head(5)

pred = gamfit.fit(df, "Surv(entry, exit, event) ~ s(age) + bmi").predict(test_df)
pred.write_survival_at_csv(
    "surv.csv",
    times=[1, 5, 10, 20, 50, 100],
    people_chunk=50_000,
    time_grid_chunk=64,
)
```

## Calibrated marginal-slope pipeline

Condition the score on covariates and fit the slope surface in one
cross-fitted, orthogonalized call by supplying a Stage-1 recipe with
`transformation_normal_stage1=`. No `z_column` is materialised or passed
by hand — the conditioned, cross-fitted score lives inside the fit.

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["pc1", "pc2", "pc3", "pc4"]).assign(age=rng.uniform(40, 70, n), family_id=np.arange(n) // 2)
df["PGS"] = 0.5 * df["pc1"] + rng.normal(size=n)
df["case"] = (rng.normal(size=n) < -0.3 + 0.8 * (df["PGS"] - 0.5 * df["pc1"]) + 0.03 * (df["age"] - 55)).astype(int)
test_df = df.drop(columns="case").head(5)

model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3, pc4, centers=20)",
    family="bernoulli-marginal-slope",
    slope_formula="matern(pc1, pc2, pc3, pc4, centers=20)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="PGS",
        covariates="matern(pc1, pc2, pc3, pc4, centers=20)",
        group_column="family_id", folds=2,
    ),
    scale_dimensions=True,
)
probs = model.predict(test_df, return_type="dict")["mean"]
```

## Survival marginal-slope

```python
import gamfit
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"bmi": rng.normal(27, 4, n), "hba1c": rng.normal(5.5, 0.6, n), "family_id": np.arange(n) // 2})
df["PGS"] = 0.1 * (df["bmi"] - 27) + rng.normal(size=n)
t, c = 15 * rng.weibull(1.5, n) * np.exp(-0.5 * df["PGS"] - 0.3 * (df["hba1c"] - 5.5)), rng.uniform(5, 30, n)
df = df.assign(entry=0.0, exit=np.minimum(t, c), event=(t <= c).astype(float))

gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi) + s(hba1c)",
    survival_likelihood="marginal-slope",
    slope_formula="s(bmi) + s(hba1c)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="PGS",
        covariates="s(bmi) + s(hba1c)",
        group_column="family_id", folds=2,
    ),
)
```

## Joint mean and variance (location-scale)

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x1, x2 = rng.uniform(0, 1, 400), rng.uniform(0, 1, 400)
df = {"x1": x1, "x2": x2, "y": np.sin(2 * np.pi * x1) + x2 ** 2 + rng.normal(0, 0.1 + 0.4 * x1)}

gamfit.fit(
    df,
    "y ~ s(x1) + s(x2)",
    noise_formula="s(x1)",     # smooth log-scale submodel
)
```

`noise_formula=` is a first-class `fit` keyword; `config=` refuses the same
key.

## Pass through an identifier column

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")

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
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")
test = {"x": [1.5, 5.0, 8.5]}

posterior = model.sample(train, seed=42)
bands = posterior.predict(test, level=0.95)
# columns: linear_predictor, linear_predictor_lower, linear_predictor_upper,
#          posterior_mean, posterior_mean_lower, posterior_mean_upper
```

## Posterior of a derived quantity

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
treatment, age = rng.integers(0, 2, 400), rng.uniform(30, 70, 400)
train = {"treatment": treatment, "age": age,
         "case": (rng.uniform(size=400) < 1 / (1 + np.exp(-(0.7 * treatment + (age - 50) / 10)))).astype(int)}
model = gamfit.fit(train, "case ~ treatment + s(age)", family="binomial")

posterior = model.sample(train, seed=42)
beta_t = posterior["beta_1"]              # (n_draws,)
or_draws = np.exp(beta_t)
print(f"OR = {or_draws.mean():.2f} "
      f"(95% CI {np.quantile(or_draws, 0.025):.2f}–"
      f"{np.quantile(or_draws, 0.975):.2f})")
```

## Save and reload

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")

model.save("model.gam")
m = gamfit.load("model.gam")
```

## Catch schema errors

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")


def safe_predict(model, data):
    check = model.check(data)
    if not check.ok:
        for issue in check.issues:
            print(issue.kind, issue.column, issue.message)
        check.raise_for_error()
    return model.predict(data)


safe_predict(model, {"x": [1.5, 2.5]})
```

## sklearn cross-validation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from gamfit.sklearn import GAMRegressor

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": rng.uniform(0, 10, 300)})
y = np.sin(X["x"]) + rng.normal(0, 0.3, 300)

scores = cross_val_score(
    GAMRegressor(formula="y ~ s(x)"),
    X, y, cv=5, scoring="r2",
)
```

## Choose between formulas

Fit each candidate and rank the fits with `gamfit.compare_models`, which
scores them by AIC corrected for smoothing-parameter selection. There is
nothing to grid-search: REML already chose every smoothing parameter inside
each fit, so the candidates differ only in structure.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
data = pd.DataFrame({"x": rng.uniform(0, 1, 400), "z": rng.uniform(0, 1, 400)})
data["y"] = np.sin(2 * np.pi * data["x"]) + 4 * (data["z"] - 0.5) ** 2 + rng.normal(0, 0.3, 400)

candidates = ["y ~ s(x)", "y ~ s(x) + z", "y ~ s(x) + s(z)"]
fits = [gamfit.fit(data, formula) for formula in candidates]
comparison = gamfit.compare_models(fits, names=candidates)

print("winner:", comparison["winner"])
for row in comparison["ranking"]:
    print(f"{row['name']:18s} delta={row['delta_aic']:7.2f}  edf={row['edf_corrected']:.2f}")
```

Here `z` bends symmetrically around 0.5, so a straight line in `z`
explains nothing: a bare numeric term is a penalized linear effect, REML
shrinks its slope to zero, and `y ~ s(x) + z` ties with `y ~ s(x)`. Only
`s(z)` captures the bend, and it wins by a wide margin.

The candidates must share a family; `compare_models` refuses to rank, say,
a Poisson fit against a negative binomial one. Basis size is not a
candidate either: a default `s(x)` sizes its own basis, and a fixed `k` is
an upper bound, so check it with `model.basis_check(data)` instead of
comparing `k=10` against `k=20` (see
[Choosing `k`](formulas.md#choosing-k)).

## Per-group trajectories (factor by smooth)

`y ~ fac + s(time, by=fac)` fits separate time trajectories by level; include the main `fac` effect for level offsets.

## Hierarchical / partial-pooling smooths (`bs="fs"`)

`y ~ s(time) + s(time, subject, bs="fs")` models a population curve plus shrinkage-stabilized subject-specific departures.

## Treatment vs control difference smooth (`bs="sz"`)

`y ~ s(time) + s(time, treatment, bs="sz")` estimates a population time effect and sum-to-zero treatment deviations.

## HTML report

```python
import gamfit
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
train = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}
model = gamfit.fit(train, "y ~ s(x)")

model.report("report.html")     # writes to disk
html = model.report()           # returns the string for inline display
```

## Example index

Focused demos live under `examples/`:

| Area | Examples |
| --- | --- |
| Manifold / SAE | `sae_manifold_demo.py`, `sae_manifold_uncertainty.py`, `sae_manifold_ordered_beta_bernoulli_topology_atoms.py`, `topology_selection_demo.py` |
| Torch | `torch_autograd_sae_training_demo.py` and the `gamfit.torch` tests for REML primitives, smooth APIs, and manifold SAE parity |
| Streaming / scale | `streaming_bspline_demo.py`, `streaming_matern_demo.py`, `streaming_arrow_schur_k100k_demo.py` |
| Response geometry / topology | `sphere_optimization_demo.py`, `product_torus_demo.py`, `grassmann_subspace_demo.py`, `spd_metric_learning_demo.py` |

Performance-oriented Rust examples live under `examples/perf/`.

