# Families and link functions

`gamfit` supports Gaussian, binomial, Poisson, negative-binomial, beta,
Gamma, inverse Gaussian, Tweedie, multinomial-logit, and Royston-Parmar
likelihoods, plus
survival ([survival.md](survival.md)), conditional transformation-normal,
location-scale / dispersion ([location-scale.md](location-scale.md)) and
marginal-slope families. The family is inferred from the response unless
overridden via `family=`.

## Auto-detection

| Response column | Inferred family | Default link |
| --- | --- | --- |
| Binary `{0, 1}` | binomial | logit |
| Non-negative integer counts (auto) | Poisson | log |
| Other continuous numeric | Gaussian | identity |
| `Surv(entry, exit, event)` | survival | depends on `survival_likelihood` |

A numeric response auto-routes to **Poisson/log on the bare `y ~ s(x)` path**
(no `family=`, no `link=`) when it has the **count signature**: every value is
finite, `>= 0`, and exactly integer-valued, with at least one value `>= 2`. Any
other numeric response is Gaussian/identity. (A `{0, 1}` column is binary, not a
count; the `>= 2` requirement is what separates counts from a binary column.)

> **Ordinal caveat.** An integer rating column such as `0..5` matches the count
> signature and therefore auto-routes to Poisson/log. If you want it treated as
> a continuous/Gaussian response, pass `family="gaussian"` explicitly.

A link that several families admit does not choose the family. `log` is legal for
Poisson, Tweedie, negative binomial, Gamma and inverse Gaussian, and `inverse` for
Gaussian and Gamma. Pinning either link without a `family=` is an error that lists
those families. A variance function is a modelling choice, so gamfit does not read
it off whether `y` happens to be integer-valued:

```python
gamfit.fit(df, "count ~ s(x)")                 # integer counts -> Poisson/log (auto)
gamfit.fit(df, "count ~ s(x)", family="poisson", link="log")  # explicit
gamfit.fit(df, "cost ~ s(x)", family="gamma", link="log")     # explicit
gamfit.fit(df, "cost ~ s(x)", link="log")      # error: name one with family=
```

## Setting family and link

The `family=` kwarg accepts `"gaussian"`, `"binomial"` (aliases
`"binomial-logit"`, `"binomial-probit"`, `"binomial-cloglog"`),
`"latent-cloglog-binomial"`, `"poisson"`, `"negative-binomial"`,
`"beta"`, `"gamma"`, `"inverse-gaussian"`, `"tweedie"`, `"royston-parmar"`, and
`"multinomial"` / `"softmax"`. Omitting
`family=` triggers auto-detection. Survival, transformation-normal,
and Bernoulli marginal-slope families are selected via `Surv(...)` or
dedicated fit options (Python: `transformation_normal=True`,
`z_column=`/`slope_formula=`; CLI: `--transformation-normal`,
`--z-column`/`--slope-formula`). In `gamfit`, set standard-family
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
`family="gamma"` or `family="inverse-gaussian"` for positive continuous
responses.

```python
gamfit.fit(df, "count ~ s(time)",
           family="poisson", link="log", offset="log_exposure")
```

Pass the offset column via `offset=`; do not include it on the formula RHS.

### `inverse`

Inverse link `1 / eta` (alias `1/mu`), canonical for the Gamma family and
also legal for the Gaussian family. The mean is only defined on `eta > 0`.
There is no hand-supplied bound: an inner Newton/PIRLS step or an outer
trial point that would put any weighted row at `eta <= 0` is reported as an
inverse-link domain violation, and the step is halved until every row is
feasible again (the same retriable refusal every bounded link uses). A fit
therefore only ever certifies at a mean that is positive at every observed
row. Prediction away from the data can still produce `eta <= 0`; such points
have no mean under this link and are refused rather than clipped.

```python
gamfit.fit(df, "y ~ s(x)", family="gamma", link="inverse")
```

### `inverse-squared`

Inverse link `eta^(-1/2)` (`eta = 1 / mu^2`, aliases `inv-squared` and
`1/mu^2`), canonical for the inverse Gaussian family and legal only there.
The `eta > 0` domain is handled exactly as for `inverse`.

### Link legality

Each family admits a fixed set of links. An illegal pairing is refused with
the family's legal links spelled out, generated from the same table the
engine checks, e.g.

```
illegal likelihood cell: response `gamma` does not admit inverse link `identity`;
legal links for `gamma`: log|inverse
```

| family | legal links |
| --- | --- |
| gaussian | identity, inverse |
| gamma | log, inverse |
| inverse-gaussian | log, inverse-squared |
| poisson, negative-binomial, tweedie | log |

An unknown link name is refused with the whole vocabulary. Link names are
case-insensitive and `_` is read as `-`, so `inverse_squared` and
`Inverse-Squared` are the same link.

### Inverse Gaussian

`family="inverse-gaussian"` fits `y > 0` with `Var(y) = phi * mu^3`, using
its canonical `inverse-squared` link by default or `link="log"`. `phi` is
the dispersion itself, not its square root. It is estimated exactly like the
Gaussian non-identity-link dispersion: the maximum-likelihood value
`sum(w (y - mu)^2 / (y mu^2)) / sum(w)` at the converged mean, refreshed until
it is stationary and then held while REML/LAML selects the smoothing
parameters. Responses must be strictly positive; a
zero or negative weighted response is refused with its row. `predict`
returns the posterior mean of `mu` (the expectation of the inverse link under the
Gaussian posterior of `eta`, not the plug-in `mu(eta_hat)`), and prediction intervals use the
inverse Gaussian law with the estimated `phi`.

```python
gamfit.fit(df, "y ~ s(x)", family="inverse-gaussian")              # 1/mu^2
gamfit.fit(df, "y ~ s(x)", family="inverse-gaussian", link="log")
```

### Dispersion families

Gamma, Beta, negative-binomial, and Tweedie can be fit as ordinary
mean models or as two-submodel dispersion fits with `noise_formula=`.
For the dispersion path, the secondary formula models Gamma shape, Beta
precision, negative-binomial size, or Tweedie inverse dispersion.

```python
gamfit.fit(df, "rate ~ s(age)", family="negative-binomial", link="log")
gamfit.fit(df, "prop ~ s(x)", family="beta", noise_formula="s(x)")
gamfit.fit(df, "claim ~ te(age, year)", family="tweedie(p=1.5)", link="log")
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

At `epsilon = 0`, `delta = 1` the SAS link is the probit, not the logit.
`epsilon` and `delta` describe the link's tails, so they are identified only
when the fitted means reach those tails. When every mean stays near 1/2, as in
`tests/sas_link_logistic_data_regression_test.py`, where the means lie within
about [0.27, 0.73], the two shape parameters are weakly identified. The
REML/LAML path then tends to drift toward small `delta`. On many such draws it
converges and the fitted mean is as accurate as the `logit` fit's. On others it
reaches an inner-mode fold, a point where the penalized likelihood's softest
curvature vanishes and past which there is no inner mode. There the Laplace
normalizer breaks down, and the fit is refused with "did not certify a
stationary optimum" or "all 1 seed candidates failed" rather than returned
uncertified. If the data do not reach the tails, use `logit` or `probit`, or
`beta-logistic` (whose shape parameters leave logit's location and scale
fixed).

### `beta-logistic`

Bounded inverse link with two learned shape parameters. The link is the
CDF of `logit(U)`, `U ~ Beta(a, b)`, standardized to logit's location and
scale, so its `(epsilon, log_delta)` move only skew and tails. Cannot be
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

Adds a jointly fit anchored spline offset to a base link for binomial
responses. Accepted base links: `identity`, `log`, `logit`, `probit`,
`cloglog`. The `sas`,
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

For Gaussian, Poisson, Gamma, and other non-binomial responses, use the
plain base link. Non-binomial `flexible(...)` links are rejected because
there is no fitted offset-spline solver for those likelihoods.

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
