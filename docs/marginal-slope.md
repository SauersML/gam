# Marginal-slope models

A marginal-slope model fits a standardised risk score `z` whose effect on
the outcome varies across covariate space. The baseline risk surface and
the score's slope surface live in two separate formulas, so the
baseline does not absorb score-specific signal and vice versa.

![two-surface marginal-slope viz over a joint Duchon smooth](images/marginal_slope_3d.png)

The vertical gap between the two probability surfaces is the risk
difference for a unit contrast in `z`. The modelled score effect lives on
the probit/slope scale and varies smoothly with covariates.

Two families:

- Bernoulli marginal-slope for binary outcomes. In Python, pass
  `family="bernoulli-marginal-slope"` with `logslope_formula=`; in the
  CLI, `--z-column` and `--logslope-formula` route to this fit.
- Survival marginal-slope (`survival_likelihood="marginal-slope"`) for
  time-to-event outcomes.

Both are identified around a latent `z` scale that should be approximately
`N(0, 1)` conditional on the covariates. Pass
`transformation_normal_stage1=gamfit.CtnStage1(...)` to condition the score
on covariates inside the fit (the calibrated chain below), or a raw
`z_column=` when the score is already conditionally `N(0, 1)` from outside
this pipeline.

## When to use it

You have:

1. A binary or time-to-event outcome.
2. A continuous risk score that is, or can be made, conditionally
   `N(0, 1)`.
3. Reason to believe the score's effect size varies across covariates
   (e.g. across age, grouping PCs).

A single-coefficient logistic or Cox fit on `outcome ~ score + ...`
forces one slope on the score. Marginal-slope makes the slope itself a
smooth function of covariates while leaving the baseline as a separate
smooth.

## Calibrated marginal slope (CTN-conditioned score)

When the score must be conditioned on covariates to reach the latent
`N(0, 1)` scale, supply a Stage-1 transformation-normal recipe with
`transformation_normal_stage1=`. This is the single calibrated
marginal-slope entry: it fits the conditional transformation
`h(score | covariates) ~ N(0, 1)`, cross-fits it out-of-fold, and absorbs
the Stage-1 score-influence directions so the fitted slope surface
`β(x)` is insensitive to Stage-1 calibration error (Neyman-orthogonal
cross-fitting). You do not materialise or pass a `z_column` yourself — the
conditioned score is produced and cross-fitted inside the one fit.

```python
model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    logslope_formula="matern(pc1, pc2, pc3)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score",
        covariates="duchon(pc1, pc2, pc3, pc4, centers=20)",
    ),
    scale_dimensions=True,
)

probs = model.predict(test_df, return_type="dict")["mean"]
```

By default, Bernoulli marginal-slope prediction returns a 1-D NumPy
array of probabilities. Passing `return_type=` asks for a table. Passing
`interval=0.95` asks for the interval table with `linear_predictor`,
`mean`, `std_error`, `mean_lower`, and `mean_upper`; probability-scale
values are clipped to `[0, 1]`.

- `family="bernoulli-marginal-slope"` names the likelihood;
  `logslope_formula=` is the slope surface as a function of covariates.
- `transformation_normal_stage1=gamfit.CtnStage1(response=..., covariates=...)`
  is the Stage-1 recipe: `response` is the raw score column to condition,
  `covariates` is the covariate-side formula right-hand side used to fit
  `h(score | covariates) ~ N(0, 1)`. Supplying it *is* the request for the
  orthogonalized chain — there is no separate boolean.
- The base link is fixed to probit. The Python `link=` keyword is not
  needed for marginal-slope fits.

The main formula controls the baseline risk; `logslope_formula` controls
the strength of the score effect at each point in covariate space.

The same recipe drives the survival likelihood:

```python
model = gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(bmi) + s(hba1c)",
    survival_likelihood="marginal-slope",
    logslope_formula="s(bmi) + s(hba1c)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score",
        covariates="s(bmi) + s(hba1c)",
    ),
)

pred = model.predict(test_df)
S = pred.survival_at([1, 5, 10])
```

The main formula specifies the baseline survival surface; the score's
slope on the marginal-calibrated probit survival scale is a smooth
function of covariates given by `logslope_formula`. In the Python API,
omitting `logslope_formula` reuses the main covariate formula for the
slope surface.

## Externally calibrated score (raw `z_column`)

If the score is already conditionally `N(0, 1)` — for example a
standardised score produced outside this pipeline — pass it directly with
`z_column=` and omit the Stage-1 recipe. This raw-`z` path uses the
free-warp `score_warp` fallback (it can only defend the covariate-free
component of any residual miscalibration); prefer the calibrated chain
above when the score is conditioned on covariates here.

```python
model = gamfit.fit(
    df,
    "case ~ s(age) + matern(pc1, pc2, pc3)",
    family="bernoulli-marginal-slope",
    z_column="z",
    logslope_formula="matern(pc1, pc2, pc3)",
    scale_dimensions=True,
)
```

- `z_column="z"`: name of the conditional z-score column in both the
  training and prediction tables.

CLI equivalent:

```bash
gam fit data.csv 'case ~ s(age) + matern(pc1, pc2, pc3)' \
    --logslope-formula 'matern(pc1, pc2, pc3)' --z-column z \
    --scale-dimensions --out model.gam
```

Bernoulli marginal-slope currently consumes a single `z_column`.

## Frailty in marginal-slope survival

Survival marginal-slope supports no frailty, or
`frailty_kind="gaussian-shift"` with a fixed `frailty_sd`.
`"hazard-multiplier"` and a learnable gaussian-shift sigma are rejected
at fit time.

```python
gamfit.fit(df,
    "Surv(entry, exit, event) ~ s(age)",
    survival_likelihood="marginal-slope",
    logslope_formula="s(age)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="raw_score", covariates="s(age)",
    ),
    frailty_kind="gaussian-shift",
    frailty_sd=0.3,
)
```

## Detecting marginal-slope models after loading

```python
model = gamfit.load("model.gam")
model.is_marginal_slope            # True if a marginal-slope family
model.is_survival                  # True if survival
model.is_transformation_normal     # True if a Stage 1 calibration model
model.model_class                  # full class string
```

## Notes

- Supply `transformation_normal_stage1=` to condition the score on
  covariates inside the fit (the calibrated chain). Use a raw `z_column=`
  only for a score already conditionally `N(0, 1)` from outside this
  pipeline.
- Set `scale_dimensions=True` when calibrating on a handful of PCs so
  anisotropic length scales are learned per axis.
- Posterior sampling: Bernoulli marginal-slope and transformation-normal
  models use the Gaussian Laplace approximation in `Model.sample(...)`;
  survival marginal-slope uses NUTS over the joint coefficient vector.
  See [posterior-sampling.md](posterior-sampling.md).
- Predict output: Bernoulli marginal-slope returns a 1-D probability
  array by default. Pass `id_column=` or `return_type="dict"` for a
  table. The same applies to transformation-normal models.

## End-to-end example

```python
import gamfit
import numpy as np
import pandas as pd

n = 1000
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "PGS":  rng.normal(0, 1, n) + 0.3 * rng.normal(0, 1, n),
    "pc1":  rng.normal(0, 1, n),
    "pc2":  rng.normal(0, 1, n),
    "pc3":  rng.normal(0, 1, n),
})
df["disease"] = (rng.uniform(0, 1, n) < 0.25).astype(float)

# Condition the score on the PCs and fit the slope surface in one
# cross-fitted, orthogonalized call.
model = gamfit.fit(
    df,
    "disease ~ matern(pc1, pc2, pc3, centers=20)",
    family="bernoulli-marginal-slope",
    logslope_formula="matern(pc1, pc2, pc3, centers=20)",
    transformation_normal_stage1=gamfit.CtnStage1(
        response="PGS",
        covariates="matern(pc1, pc2, pc3, centers=20)",
    ),
    scale_dimensions=True,
)

test = df.head(50).copy()
probs = model.predict(test, return_type="dict")["mean"]
```
