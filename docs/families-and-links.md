# Families and link functions

`gamfit` supports Gaussian, binomial, Poisson, and Gamma GLMs, plus survival
models (covered in [survival.md](survival.md)) and conditional
transformation-normal models (covered in [marginal-slope.md](marginal-slope.md)).
The family is **auto-detected from the response** unless you override it.

## Auto-detection

| Response column looks like... | Inferred family | Default link |
| --- | --- | --- |
| Binary `{0, 1}` only | binomial | logit |
| Continuous numeric | Gaussian | identity |
| `Surv(entry, exit, event)` | survival | depends on likelihood mode |

For count data (Poisson) and positive-continuous data (Gamma), set the link
explicitly with `link(type=log)` and pass `family=` to `fit()`:

```python
gamfit.fit(df, "y ~ s(x) + link(type=log)", family="poisson")
gamfit.fit(df, "y ~ s(x) + link(type=log)", family="gamma")
```

## Overriding family or link

Two equivalent ways to set the link:

```python
# Formula-level
gamfit.fit(df, "case ~ s(age) + link(type=probit)")

# Python kwarg
gamfit.fit(df, "case ~ s(age)", link="probit")
```

If both are set, the formula-level option wins.

## Link functions in detail

### Identity — `link(type=identity)`

`g⁻¹(η) = η`. The default for continuous responses. Use when you have no
reason to constrain the response scale.

### Logit — `link(type=logit)`

`g⁻¹(η) = 1 / (1 + e^{-η})`. The default for binary `{0, 1}` responses.
Symmetric and standard; the workhorse link for binomial regression.

### Probit — `link(type=probit)`

`g⁻¹(η) = Φ(η)` (standard normal CDF). Has slightly lighter tails than the
logit. Common in econometrics and when a Gaussian latent-variable
interpretation is natural. Required by Bernoulli marginal-slope models — see
[marginal-slope.md](marginal-slope.md).

### Complementary log-log — `link(type=cloglog)`

`g⁻¹(η) = 1 − e^{−e^η}`. Asymmetric (left-skewed). Useful when events at
high covariate values cluster toward 1 — e.g. interval-censored survival in
discrete time, or rare events.

### Log — `link(type=log)`

`g⁻¹(η) = e^η`. For positive responses. Use with Poisson (counts) and Gamma
(positive continuous):

```python
gamfit.fit(df, "count ~ s(time) + offset_log_exposure + link(type=log)",
           family="poisson", offset="offset_log_exposure")
```

### SAS (sinh-arcsinh) — `link(type=sas)`

A flexible link with skewness/tail-weight parameters that the model learns.
Use when you suspect the link is misspecified in a structured way and want
to estimate the correction. *Cannot* be combined with `linkwiggle(...)`.

### Beta-logistic — `link(type=beta-logistic)`

A bounded link with learnable shape parameters. *Cannot* be combined with
`linkwiggle(...)`.

### Blended mixture — `link(type=blended(a, b, …))`

Mixture of two or more component inverse links, with mixing weights learned
from the data:

```
link(type=blended(logit, probit))
link(type=blended(logit, cloglog))
link(type=blended(logit, probit, cloglog))
```

Component options: `logit`, `probit`, `cloglog`, `loglog`, `cauchit`.
At least two required. *Cannot* be combined with `linkwiggle(...)`.

### Flexible — `link(type=flexible(base))`

Adds a spline offset to a base link, so the data can correct for link
misspecification while keeping the base as a sensible default:

```
link(type=flexible(probit))
link(type=flexible(logit))
link(type=flexible(cloglog))
link(type=flexible(identity))
link(type=flexible(log))
```

Automatically enables `linkwiggle`. You can tune the offset spline directly:

```
y ~ s(x) + link(type=flexible(probit)) + linkwiggle(internal_knots=8, penalty_order=all)
```

See [formulas.md](formulas.md#linkwiggle-flexible-link-offset) for
`linkwiggle` options.

## Choosing a link

| You want... | Use... |
| --- | --- |
| Binary outcome, no strong prior | `logit` (default). |
| Binary outcome, Gaussian latent interpretation | `probit`. |
| Rare events / interval-censored discrete-time survival | `cloglog`. |
| Count / positive continuous | `log` (with `family="poisson"` or `"gamma"`). |
| Believe the base link is approximately right but want a safety net | `flexible(base)` + `linkwiggle`. |
| Believe two link shapes both have support | `blended(a, b)`. |
| Strong skewness signal in the residuals | `sas`. |

## Firth bias reduction

For separable or near-separable logistic problems, enable Firth's bias-reduced
estimator with `firth=True`:

```python
gamfit.fit(df, "rare_event ~ s(x)", family="binomial", firth=True)
```

This shrinks the MLE toward the Jeffreys prior, stabilising estimates when
classes are imbalanced and a coefficient would otherwise diverge.

## Offsets and weights

Both are referenced by column name:

```python
gamfit.fit(df,
    "count ~ s(age) + link(type=log)",
    family="poisson",
    offset="log_exposure",   # column in df
    weights="freq",          # column in df
)
```

- **`offset`**: an additive term on the linear predictor that is *not*
  estimated. For Poisson rate models, pass `log(exposure)` here.
- **`weights`**: per-observation weight applied to the likelihood. Use for
  frequency-weighted or inverse-variance-weighted data.
