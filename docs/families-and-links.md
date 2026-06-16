# Families and link functions

`gamfit` supports Gaussian, binomial, Poisson, negative-binomial, beta,
Gamma, Tweedie, multinomial-logit, and Royston-Parmar likelihoods, plus
survival ([survival.md](survival.md)), conditional transformation-normal,
location-scale / dispersion ([location-scale.md](location-scale.md)) and
marginal-slope families. The family is inferred from the response unless
overridden via `family=`.

## Auto-detection

| Response column | Inferred family | Default link |
| --- | --- | --- |
| Binary `{0, 1}` | binomial | logit |
| Continuous numeric | Gaussian | identity |
| Non-negative integer with log link | Poisson | log |
| Otherwise, with log link | Gamma | log |
| `Surv(entry, exit, event)` | survival | depends on `survival_likelihood` |

When the log link is set, Poisson vs Gamma is chosen automatically
by whether the response is integer-valued — `family=` is optional:

```python
gamfit.fit(df, "y ~ s(x)", link="log")     # auto-routes Poisson/Gamma
gamfit.fit(df, "y ~ s(x)", family="poisson", link="log")  # explicit
```

## Setting family and link

The `family=` kwarg accepts `"gaussian"`, `"binomial"` (aliases
`"binomial-logit"`, `"binomial-probit"`, `"binomial-cloglog"`),
`"latent-cloglog-binomial"`, `"poisson"`, `"negative-binomial"`,
`"beta"`, `"gamma"`, `"tweedie"`, `"royston-parmar"`, and
`"multinomial"` / `"softmax"`. Omitting
`family=` triggers auto-detection. Survival, transformation-normal,
and Bernoulli marginal-slope families are selected via `Surv(...)` or
dedicated fit options (Python: `transformation_normal=True`,
`z_column=`/`logslope_formula=`; CLI: `--transformation-normal`,
`--z-column`/`--logslope-formula`). In `gamfit`, set standard-family
links with the `link=` kwarg. In the CLI, set them in the formula via
`link(type=...)`.

```python
gamfit.fit(df, "case ~ s(age)", link="probit")
```

## Link functions

The engine recognises the following link types in formula `link(type=...)`
and the Python `link=` kwarg. There is no top-level `gam fit --link`
flag; CLI fits use formula-level `link(...)`.

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
gamfit.fit(df, "count ~ s(time)",
           family="poisson", link="log", offset="log_exposure")
```

Pass the offset column via `offset=`; do not include it on the formula RHS.

### Dispersion families

Gamma, Beta, negative-binomial, and Tweedie can be fit as ordinary
mean models or as two-submodel dispersion fits with `noise_formula=`.
For the dispersion path, the secondary formula models Gamma shape, Beta
precision, negative-binomial size, or Tweedie inverse dispersion.

```python
gamfit.fit(df, "rate ~ s(age)", family="negative-binomial", link="log")
gamfit.fit(df, "prop ~ s(x)", family="beta", noise_formula="s(x)")
gamfit.fit(df, "claim ~ te(age, year)", family="tweedie", link="log")
```

`negative_binomial_theta` / `--negative-binomial-theta` fixes the
negative-binomial size parameter when a constant-size model is desired.

### Multinomial

Use `family="multinomial"` (aliases `"multinomial-logit"`,
`"categorical"`, `"categorical-logit"`, `"softmax"`) for a vector
softmax model. The Python API returns a `MultinomialModel`; scalar
`Model.predict` details do not apply to that class.

`gamfit.fit(..., family="multinomial")` dispatches to the dedicated
multinomial formula path. `gamfit.validate_formula(...)` uses the scalar
materialization preflight, so it is not a multinomial validator.

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

`firth=True` activates Firth's bias-reduced estimator. It is available for
any binomial family — i.e. a binomial response with any binomial inverse
link that carries a Fisher-weight jet (`logit`, `probit`, `cloglog`,
`latent-cloglog`, `sas`, `beta-logistic`, and blended/mixture links). The
eligibility gate is `LikelihoodSpec::supports_firth()`; non-binomial
families reject `firth=True` with an error.
Firth is not compatible with survival models, location-scale fitting, or
the Bernoulli marginal-slope family.

```python
gamfit.fit(df, "rare_event ~ s(x)", family="binomial-logit", firth=True)
```

## Offsets and weights

```python
gamfit.fit(df,
    "count ~ s(age)",
    family="poisson",
    link="log",
    offset="log_exposure",
    weights="freq",
)
```

- `offset`: column added to the linear predictor and not estimated. For
  Poisson rate models pass `log(exposure)`.
- `weights`: per-observation likelihood weight. Use for frequency or
  inverse-variance weighting.
