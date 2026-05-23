# Families and link functions

`gamfit` supports Gaussian, binomial, Poisson, Gamma, and Royston-Parmar
GLM-style likelihoods, plus survival ([survival.md](survival.md)),
conditional transformation-normal ([marginal-slope.md](marginal-slope.md)),
location-scale ([location-scale.md](location-scale.md)) and
marginal-slope families. The family is inferred from the response unless
overridden via `family=`.

## Auto-detection

| Response column | Inferred family | Default link |
| --- | --- | --- |
| Binary `{0, 1}` | binomial | logit |
| Continuous numeric | Gaussian | identity |
| Non-negative integer with `link(type=log)` | Poisson | log |
| Positive non-integer with `link(type=log)` | Gamma | log |
| `Surv(entry, exit, event)` | survival | depends on `survival_likelihood` |

When `link(type=log)` is set, Poisson vs Gamma is chosen automatically
by whether the response is integer-valued — `family=` is optional:

```python
gamfit.fit(df, "y ~ s(x) + link(type=log)")     # auto-routes Poisson/Gamma
gamfit.fit(df, "y ~ s(x) + link(type=log)", family="poisson")  # explicit
```

## Setting family and link

The `family=` kwarg accepts `"gaussian"`, `"binomial"` (aliases
`"binomial-logit"`, `"binomial-probit"`, `"binomial-cloglog"`),
`"latent-cloglog-binomial"`, `"poisson"`, and `"gamma"`. Omitting
`family=` triggers auto-detection. Survival, transformation-normal,
and Bernoulli marginal-slope families are selected via `Surv(...)` or
dedicated flags (`--transformation-normal`, `--z-column`/`--logslope-formula`),
not via `family=`. The link can be set in the formula via
`link(type=...)` or with the `link=` kwarg. If both are set, the formula
specification takes precedence.

```python
# Formula
gamfit.fit(df, "case ~ s(age) + link(type=probit)")

# Kwarg
gamfit.fit(df, "case ~ s(age)", link="probit")
```

## Link functions

The engine recognises the following link types in `link(type=...)` and
`--link`:

### `identity`

Inverse link `eta`. Default for continuous Gaussian responses.

### `logit`

Inverse link `1 / (1 + exp(-eta))`. Default for binary `{0, 1}` responses.

### `probit`

Inverse link `Phi(eta)`, the standard normal CDF. Required for the
Bernoulli marginal-slope family (see [marginal-slope.md](marginal-slope.md)).

### `cloglog`

Inverse link `1 - exp(-exp(eta))`. Used for grouped discrete-time hazards
and rare-event Bernoulli data.

### `log`

Inverse link `exp(eta)`. Pair with `family="poisson"` for counts and
`family="gamma"` for positive continuous responses.

```python
gamfit.fit(df, "count ~ s(time) + link(type=log)",
           family="poisson", offset="log_exposure")
```

Pass the offset column via `offset=`; do not include it on the formula RHS.

### `sas`

Sinh-arcsinh inverse link with learned skewness (`epsilon`) and
tail-weight (`delta`) parameters. Cannot be combined with `linkwiggle(...)`
or with blended/mixture links.

### `beta-logistic`

Bounded inverse link with two learned shape parameters. Cannot be
combined with `linkwiggle(...)` or with blended/mixture links.

### `blended(a, b, ...)` / `mixture(a, b, ...)`

Convex combination of two or more inverse links with learned mixing
weights:

```
link(type=blended(logit, probit))
link(type=blended(logit, cloglog))
link(type=blended(logit, probit, cloglog))
```

Component options: `logit`, `probit`, `cloglog`, `loglog`, `cauchit`. At
least two components are required. Cannot be combined with `linkwiggle(...)`
or with `flexible(...)`.

### `flexible(base)`

Adds a jointly fit anchored spline offset to a base link. Accepted base
links: `identity`, `log`, `logit`, `probit`, `cloglog`. The `sas`,
`beta-logistic`, `blended(...)`, and `mixture(...)` types are not
supported as a `flexible(...)` base.

```
link(type=flexible(probit))
link(type=flexible(logit))
link(type=flexible(cloglog))
```

`flexible(...)` enables `linkwiggle(...)` for tuning the offset spline:

```
y ~ s(x) + link(type=flexible(probit)) + linkwiggle(internal_knots=8, penalty_order=all)
```

See [formulas.md](formulas.md#linkwiggle-flexible-link-offset) for
`linkwiggle` options.

## Firth bias reduction

`firth=True` activates Firth's bias-reduced estimator. The shared
implementation is available only for the standard logit binomial family
(`LikelihoodFamily::BinomialLogit`); other families reject `firth=True`.
Firth is not compatible with survival models, location-scale fitting, or
the Bernoulli marginal-slope family.

```python
gamfit.fit(df, "rare_event ~ s(x)", family="binomial-logit", firth=True)
```

## Offsets and weights

```python
gamfit.fit(df,
    "count ~ s(age) + link(type=log)",
    family="poisson",
    offset="log_exposure",
    weights="freq",
)
```

- `offset`: column added to the linear predictor and not estimated. For
  Poisson rate models pass `log(exposure)`.
- `weights`: per-observation likelihood weight. Use for frequency or
  inverse-variance weighting.
