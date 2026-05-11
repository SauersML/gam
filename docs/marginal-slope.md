# Marginal-slope models

Marginal-slope is `gamfit`'s machinery for handling a **standardised risk
score** (e.g. a polygenic score) whose effect on the outcome varies across
covariate space. The defining feature: the baseline risk surface and the
score's effect are decoupled into separate formulas, so the baseline can't
absorb signal that belongs to the slope (or vice versa).

There are two flavours:

- **Bernoulli marginal-slope** for binary outcomes.
- **Survival marginal-slope** for time-to-event outcomes.

Both pair naturally with **transformation-normal** calibration of the
underlying score on ancestry/covariate PCs, so the input score is conditionally
N(0, 1) before it enters the marginal-slope fit.

## When you'd reach for this

You have:

1. A binary or survival outcome.
2. A continuous risk score, ideally already standardised (zero mean, unit
   variance, marginally), perhaps a polygenic score, perhaps something else.
3. A belief that the *score's* effect size varies across some covariate
   space — across age, across ancestry PCs, etc.

Standard logistic regression on `outcome ~ score + age + ...` forces a
*single* slope on `score`. Marginal-slope lets the slope itself be a smooth
function of the covariates while leaving the baseline as its own smooth.

## Stage 1: transformation-normal calibration

If your raw score isn't already conditionally N(0, 1), fit:

```python
calib = gamfit.fit(
    df,
    "raw_score ~ duchon(pc1, pc2, pc3, pc4, centers=20)",
    transformation_normal=True,
    scale_dimensions=True,
)
df["z"] = calib.predict(df)   # 1-D numpy array of conditional z-scores
```

`transformation_normal=True` fits `h(score | PCs) ~ N(0, 1)`. The
`predict()` of a transformation-normal model returns a **1-D numpy array of
z-scores**, not a table.

After calibration, `df["z"]` is approximately N(0, 1) conditional on the
PCs, which is what the marginal-slope likelihood expects in `z_column`.

## Stage 2a: Bernoulli marginal-slope

```python
model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="z",
    logslope_formula="matern(pc1, pc2, pc3)",
    scale_dimensions=True,
)

probs = model.predict(test_df, return_type="dict")["mean"]
```

- `family="bernoulli-marginal-slope"` selects the marginal-slope likelihood.
- `link="probit"` — the standard choice; `cloglog` also works. Probit pairs
  with the Gaussian z assumption.
- `z_column="z"` — name of the standardised score column in `df` and `test_df`.
- `logslope_formula="..."` — formula governing how `z`'s log-slope varies
  across covariates. Typically a smooth on the same covariates as the
  baseline.

The main formula (`case ~ s(age) + matern(pc1, pc2, pc3)`) controls the
**baseline risk landscape**. The `logslope_formula` controls how strongly
`z` modifies that risk at each point in covariate space.

## Stage 2b: Survival marginal-slope

```python
model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi) + s(hba1c)",
    survival_likelihood="marginal-slope",
    z_column="z",
    logslope_formula="s(bmi) + s(hba1c)",
)

pred = model.predict(test_df)
S = pred.survival_at([1, 5, 10])
```

Same idea, applied to survival data. The baseline hazard surface is
specified by the main formula; the score's log hazard ratio is a smooth
function of covariates given by `logslope_formula`.

## Frailty for marginal-slope survival

Marginal-slope survival pairs with frailty for unmeasured heterogeneity:

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age)",
    survival_likelihood="marginal-slope",
    z_column="z",
    logslope_formula="s(age)",
    frailty_kind="gaussian-shift",
    frailty_sd=0.3,
)
```

See [survival.md](survival.md#frailty) for the frailty options.

## Detecting marginal-slope models after loading

```python
model = gamfit.load("model.gam")
model.is_marginal_slope            # True if a marginal-slope family
model.is_survival                  # True if survival
model.is_transformation_normal     # True if Stage 1 calibration model
model.model_class                  # full class string
```

## Tips

- **Order matters.** Run Stage 1 (transformation-normal) before Stage 2 so
  that `z_column` is genuinely conditionally N(0, 1).
- **Scale your covariates** for Stage 1 — `scale_dimensions=True` is almost
  always the right call when fitting transformation-normal on a handful of
  PCs.
- **Sampling / posterior:** Bernoulli marginal-slope, survival marginal-slope,
  and transformation-normal models currently fall back to the Gaussian
  Laplace approximation for `Model.sample(...)` rather than exact NUTS. The
  posterior summary still works the same way — see
  [posterior-sampling.md](posterior-sampling.md).
- **Predict output:** Bernoulli marginal-slope returns a 1-D array of
  probabilities by default. Pass `id_column=` or `return_type="dict"` to
  get a table back. Same warning applies to transformation-normal output.

## Recipe: full two-stage pipeline

```python
import gamfit
import numpy as np
import pandas as pd

# Synthetic data: PGS to calibrate, then disease outcomes
n = 1000
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "PGS":  rng.normal(0, 1, n) + 0.3 * rng.normal(0, 1, n),
    "pc1":  rng.normal(0, 1, n),
    "pc2":  rng.normal(0, 1, n),
    "pc3":  rng.normal(0, 1, n),
})
df["disease"] = (rng.uniform(0, 1, n) < 0.25).astype(float)

# Stage 1: condition the score on PCs
calib = gamfit.fit(
    df, "PGS ~ matern(pc1, pc2, pc3, centers=20)",
    transformation_normal=True, scale_dimensions=True,
)
df["pgs_z"] = np.asarray(calib.predict(df), dtype=float)

# Stage 2: Bernoulli marginal-slope
model = gamfit.fit(
    df,
    "disease ~ matern(pc1, pc2, pc3, centers=20)",
    family="bernoulli-marginal-slope",
    link="probit",
    z_column="pgs_z",
    logslope_formula="matern(pc1, pc2, pc3, centers=20)",
    scale_dimensions=True,
)

# Predict probabilities
test = df.head(50).copy()
probs = model.predict(test, return_type="dict")["mean"]
```
