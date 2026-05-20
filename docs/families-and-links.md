# Families and link functions

`gamfit` supports Gaussian, binomial, Poisson, and Gamma GLMs, plus survival
models ([survival.md](survival.md)) and conditional transformation-normal
models ([marginal-slope.md](marginal-slope.md)). The family is detected from
the response unless overridden.

## Auto-detection

| Response column looks like... | Inferred family | Default link |
| --- | --- | --- |
| Binary `{0, 1}` only | binomial | logit |
| Continuous numeric | Gaussian | identity |
| `Surv(entry, exit, event)` | survival | depends on likelihood mode |

For counts (Poisson) and positive-continuous data (Gamma), set the link with
`link(type=log)` and pass `family=` to `fit()`:

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

If both are set, the formula wins.

## Link functions in detail

### Identity — `link(type=identity)`

Inverse link: `eta`. Default for continuous responses.

### Logit — `link(type=logit)`

Inverse link: `1 / (1 + exp(-eta))`. Default for binary `{0, 1}` responses.
Symmetric.

### Probit — `link(type=probit)`

Inverse link: `Phi(eta)` (standard normal CDF). Lighter tails than logit. Common when
a Gaussian latent-variable interpretation applies. Required by Bernoulli
marginal-slope models — see [marginal-slope.md](marginal-slope.md).

### Complementary log-log — `link(type=cloglog)`

Inverse link: `1 - exp(-exp(eta))`. Left-skewed. Use for grouped discrete-time
hazards or rare events.

### Log — `link(type=log)`

Inverse link: `exp(eta)`. For positive responses. Use with Poisson (counts) and Gamma
(positive continuous):

```python
gamfit.fit(df, "count ~ s(time) + link(type=log)",
           family="poisson", offset="offset_log_exposure")
```

The offset column is supplied via `offset=`; do not also place it on the
formula RHS (a bare RHS identifier would be parsed as a linear term with
its own coefficient).

### SAS (sinh-arcsinh) — `link(type=sas)`

Link with learned skewness and tail-weight parameters. Cannot be combined
with `linkwiggle(...)`.

### Beta-logistic — `link(type=beta-logistic)`

Bounded link with learned shape parameters. Cannot be combined with
`linkwiggle(...)`.

### Blended mixture — `link(type=blended(a, b, …))`

Mixture of two or more component inverse links with learned mixing weights:

```
link(type=blended(logit, probit))
link(type=blended(logit, cloglog))
link(type=blended(logit, probit, cloglog))
```

Component options: `logit`, `probit`, `cloglog`, `loglog`, `cauchit`. At
least two required. Cannot be combined with `linkwiggle(...)`.

### Flexible — `link(type=flexible(base))`

Adds a spline offset to a base link. The data corrects for link
misspecification; the base is the default:

```
link(type=flexible(probit))
link(type=flexible(logit))
link(type=flexible(cloglog))
link(type=flexible(identity))
link(type=flexible(log))
```

Enables `linkwiggle`. Tune the offset spline directly:

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
| Rare events / grouped discrete-time hazards | `cloglog`. |
| Count / positive continuous | `log` (with `family="poisson"` or `"gamma"`). |
| Base link approximately right, want a safety net | `flexible(base)` + `linkwiggle`. |
| Two link shapes both plausible | `blended(a, b)`. |
| Skewness in the residuals | `sas`. |

## Firth bias reduction

For separable or near-separable logistic problems, enable Firth's
bias-reduced estimator with `firth=True`:

```python
gamfit.fit(df, "rare_event ~ s(x)", family="binomial", firth=True)
```

Shrinks the MLE toward the Jeffreys prior, stabilising estimates when
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

- **`offset`**: additive term on the linear predictor; not estimated. For
  Poisson rate models, pass `log(exposure)`.
- **`weights`**: per-observation weight on the likelihood. Use for
  frequency-weighted or inverse-variance-weighted data.
