# Location-scale models

A standard GAM models the conditional mean of the response. A
location-scale GAM additionally models a second parameter — the
log-scale of the response distribution — as its own smooth function of
covariates.

The engine implements two-parameter location-scale / dispersion fits for
Gaussian responses (mean and scale), Binomial responses (threshold and
scale), Gamma / Beta / negative-binomial / Tweedie responses (mean and
dispersion), and survival responses (time/threshold survival index and
scale). It is not a full arbitrary-parameter GAMLSS implementation: no
separate skewness or kurtosis submodels.

Use it when:

- The residual scale varies with covariates (heteroscedasticity).
- Prediction intervals need to widen or narrow with `x`.
- A survival model has a non-proportional shape that benefits from a
  covariate-dependent scale.

## Specifying the scale submodel

The scale formula is passed via the `noise_formula` key in `config`:

```python
gamfit.fit(
    df,
    "y ~ s(x1) + s(x2)",
    config={"noise_formula": "s(x1)"},
)
```

The main formula models the location/threshold. `noise_formula` models
the scale predictor. Pass only the right-hand side terms, not a full
`response ~ terms` formula. Both formulas refer to columns of the same
input table. The CLI flag is `--predict-noise`:

```bash
gam fit data.csv 'y ~ s(x1) + s(x2)' --predict-noise 's(x1)' --out model.gam
```

This is supported for:

- Gaussian location-scale: joint mean and scale.
- Binomial location-scale: joint threshold and scale. The default
  inverse link is logit; explicit links such as probit/cloglog use the
  same threshold-scale parameterization.
- Dispersion GAMLSS: Gamma shape, Beta precision, negative-binomial
  size, and Tweedie inverse-dispersion submodels.
- Survival location-scale: pair with `survival_likelihood="location-scale"`:

```python
gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="location-scale",
    config={"noise_formula": "s(age)"},
)
```

`--predict-noise` cannot be combined with `--logslope-formula` /
`--z-column`, with `--transformation-normal`, or with `--firth`.

## Prediction

A Gaussian location-scale fit returns the same Python prediction columns
as a standard Gaussian fit:

```python
preds = model.predict(test_df, interval=0.95)
# Columns: linear_predictor, mean, noise_scale, std_error, mean_lower, mean_upper
```

`noise_scale` is the fitted scale / dispersion channel on the response
model's natural scale. `std_error` is the response-scale delta-method
standard error. `mean_lower` / `mean_upper` are response-scale pointwise
Wald bands at the requested `interval`. Without `interval=`, standard
point-payload models return a 1-D NumPy array of response-scale point
predictions.

For survival location-scale, predictions return a
[`SurvivalPrediction`](predictions.md#survivalprediction). Pass any
`interval=...` to get delta-method standard errors on the survival
surface and linear predictor (issue #342 — the single `interval` knob
replaces the previous `with_uncertainty` boolean):

```python
pred = model.predict(test_df, interval=0.95)
S    = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])
```

## Posterior sampling

Location-scale models use the Gaussian Laplace approximation in
`Model.sample(...)` because the predictive linear predictor is non-linear
in the joint coefficient vector. The returned `PosteriorSamples` has
`method == "laplace"`, `rhat == 1.0`, and predictive bands flow through
the same `.predict(...)` interface. See
[posterior-sampling.md](posterior-sampling.md).

## Location-scale versus conditional transformation

For heteroscedastic but otherwise Gaussian-shaped residuals, use
location-scale. For residuals whose entire conditional distribution
deviates from Gaussian (skew, heavy tails), use
`transformation_normal=True` to fit a conditional transformation model
that maps the response to `N(0, 1)` given covariates; see
[marginal-slope.md](marginal-slope.md).

Guidance:

- Variance varies with `x`, residuals otherwise Gaussian:
  Gaussian location-scale.
- Mean and scale both depend on `x`: location-scale.
- The whole conditional distribution is non-Gaussian:
  transformation-normal, optionally combined with marginal-slope.
