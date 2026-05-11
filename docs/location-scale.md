# Location-scale models

A standard GAM models the **mean** of the response. A *location-scale* GAM
models the **mean and the variance/scale** jointly — each as its own smooth
function of covariates.

This is what `gamlss` calls "distributional regression". Use it when:

- Residual variance is itself a function of covariates (heteroscedasticity).
- You need prediction intervals that vary with `x`.
- Survival data has non-proportional shape and you want a flexible scale.

## Enabling location-scale

Pass a second formula for the noise / scale submodel via the `predict_noise`
key in `config`:

```python
gamfit.fit(
    df,
    "y ~ s(x1) + s(x2)",
    config={"predict_noise": "s(x1)"},
)
```

The main formula models the location (mean). The `predict_noise` formula
models log-scale. Both share the same covariate table.

This works for:

- **Gaussian location-scale** — joint mean and log-σ.
- **Binomial location-scale** — joint logit-p and a variance-modulating
  log-scale term.
- **Survival location-scale** — joint log-hazard and log-scale; pair with
  `survival_likelihood="location-scale"`:

```python
gamfit.fit(
    df,
    "Surv(entry, exit, event) ~ s(age) + bmi",
    survival_likelihood="location-scale",
    config={"predict_noise": "s(age)"},
)
```

## Prediction output

A Gaussian location-scale fit returns `sigma` alongside `eta` and `mean`:

```python
preds = model.predict(test_df, interval=0.95)
# Columns: eta, mean, sigma, mean_lower, mean_upper
```

The `sigma` column is the per-row predicted standard deviation.

For survival location-scale, predictions are a
[`SurvivalPrediction`](predictions.md#survivalprediction) object. Pass
`with_uncertainty=True` to get delta-method standard errors on the survival
surface and linear predictor:

```python
pred = model.predict(test_df, with_uncertainty=True)
S    = pred.survival_at([1, 5, 10])
se_S = pred.survival_se_at([1, 5, 10])  # populated for location-scale
```

## Posterior sampling

Location-scale models fall back to the Gaussian Laplace approximation
rather than NUTS, because the predictive linear predictor is non-linear in
the joint coefficient vector. The returned `PosteriorSamples` object has
`posterior.method == "laplace_gaussian"`, `rhat == 1.0`, and predictive
bands flow through the same `.predict(...)` interface. See
[posterior-sampling.md](posterior-sampling.md).

## Location-scale vs transformation

For lopsided / skewed residuals, location-scale models the
heteroscedasticity, but a *conditional transformation* model
(`transformation_normal=True`) transforms the response so the residual
structure becomes N(0, 1) conditionally. See
[marginal-slope.md](marginal-slope.md) for the transformation-normal flow.

Rule of thumb:

- **Pure variance heteroscedasticity** → Gaussian location-scale.
- **Both location *and* scale matter** → location-scale.
- **The whole conditional distribution is non-Gaussian (skew, heavy tails)**
  → transformation-normal, then build a marginal-slope model on top.
