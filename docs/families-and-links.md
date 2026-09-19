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

When `link="log"` is pinned *without* a `family=`, Poisson vs Gamma is chosen
automatically by whether the response is integer-valued — `family=` is optional:

```python
gamfit.fit(df, "count ~ s(x)")                 # integer counts -> Poisson/log (auto)
gamfit.fit(df, "count ~ s(x)", link="log")     # log pinned: Poisson (integer) / Gamma (else)
gamfit.fit(df, "count ~ s(x)", family="poisson", link="log")  # explicit
```

## Setting family and link

The `family=` kwarg accepts `"gaussian"`, `"binomial"` (aliases
`"binomial-logit"`, `"binomial-probit"`, `"binomial-cloglog"`),
`"latent-cloglog-binomial"`, `"poisson"`, `"negative-binomial"`,
`"beta"`, `"gamma"`, `"tweedie"`, `"student-t"` (see
[Student-t](#student-t)), `"royston-parmar"`,
`"expectile"` (see [Expectile regression](#expectile-regression)), and
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
gamfit.fit(df, "claim ~ te(age, year)", family="tweedie(p=1.5)", link="log")
```

`negative_binomial_theta` / `--negative-binomial-theta` fixes the
negative-binomial size parameter when a constant-size model is desired.

### Student-t

`family="student-t"` (aliases `"student_t"`, `"t"`) is a heavy-tailed
alternative to the Gaussian for a continuous response with outliers. The
link is the identity. The scale `σ` and the degrees of freedom `ν` are
estimated by LAML jointly with the smoothing parameters. The fitted model
reports them as `model.student_t_sigma` and `model.student_t_nu`, which are
`None` for any other family.

```python
import numpy as np
import pandas as pd
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
y = np.sin(2 * np.pi * x) + rng.standard_t(3, x.size) * 0.3
model = gamfit.fit(pd.DataFrame({"x": x, "y": y}), "y ~ s(x)", family="student-t")
print(model.student_t_sigma, model.student_t_nu)
```

### Multinomial

Use `family="multinomial"` (aliases `"multinomial-logit"`,
`"categorical"`, `"categorical-logit"`, `"softmax"`) for a vector
softmax model. The Python API returns a `MultinomialModel`; scalar
`Model.predict` details do not apply to that class.

`gamfit.fit(..., family="multinomial")` dispatches to the dedicated
multinomial formula path. `gamfit.validate_formula(...)` uses the scalar
materialization preflight, so it is not a multinomial validator.

### Expectile regression

`family="expectile"` fits the conditional `tau`-expectile `e_tau(x)`: the
minimiser of the asymmetric squared loss `sum_i |tau - 1[y_i < f(x_i)]| (y_i -
f(x_i))^2` plus the usual smoothing penalties. `tau = 0.5` is the conditional
mean; `tau` near 0 or 1 tracks the lower or upper tail.

```python
gamfit.fit(df, "y ~ s(x)", family="expectile", expectile_tau=0.9)
gamfit.fit(df, "y ~ s(x)", family="expectile(0.9)")      # same fit
```

CLI: `gam fit data.csv 'y ~ s(x)' --family expectile --expectile-tau 0.9`.

The fit is least asymmetrically weighted squares (LAWS): each inner solve is a
penalized weighted least-squares fit with weights `|tau - 1[r_i < 0]|`, and the
smoothing parameters are re-selected by REML at every weight update. A fit is
returned only once the weight pattern is a fixed point certified by the KKT
residual of the penalized asymmetric loss, so the published coefficients are
the exact penalized expectile, not an approximation.

**Uncertainty.** Expectile regression has no likelihood, so the Gaussian
working-model covariance `phi * H^-1` of the last weighted least-squares solve
is not the variance of the estimator: it ignores that the noise scale varies
with `x` and that the asymmetric weights inflate the score variance in the tail.
The published coefficient covariance is the penalized Newey–Powell sandwich at
the certified fixed point,

```
H = X' W X + S_lambda,   meat = n/(n - edf) * X' diag(w_i^2 r_i^2) X
V = H^-1 (meat + phi * S_lambda) H^-1
```

It reduces to the Newey–Powell asymptotic covariance as the penalty vanishes
and to `phi * H^-1` when the working model is exact; the `phi * S_lambda` term
keeps the penalty as a prior (a Bayesian band, as for every other family) and
`n/(n - edf)` is the HC1 degrees-of-freedom correction. The same covariance
feeds `predict(..., interval=...)` bands, `sample_posterior`, `summary()`
standard errors and the CLI, and the smoothing-parameter uncertainty correction
is added on top of it. Under heteroscedastic noise the Gaussian working
covariance under-covers badly in the tails (about 0.66 for a nominal-95% band
at `tau = 0.95` where the noise is largest); the sandwich brings that to about
0.87 there and to 0.91–0.975 over the whole range for `tau` in
{0.05, 0.5, 0.9, 0.95} (the calibration gate is
`tests/quality/misc/quality_expectile_band_coverage_heteroscedastic.rs`).

`model.family_name` and the summary header report the estimator, e.g.
`Expectile(tau=0.9)`, not the Gaussian working family used internally.

**Conditional quantiles.** An expectile is not a quantile. For conditional
quantiles use the transformation-normal family (`transformation_normal=True`;
CLI `--transformation-normal`): it models the whole conditional distribution
`F(y | x) = Phi(h(y | x))` with `h` strictly increasing in `y`, so every
quantile `h^-1(Phi^-1(p) | x)` is analytic and quantiles at different `p` can
never cross. They are returned by `predict(..., observation_interval=True)`; see
[predictions.md](predictions.md#transformation-normal-observation-intervals).
gamfit deliberately has no pyGAM-style `fit_quantile`, which searches for the
expectile `tau` whose in-sample coverage matches a target quantile: that search
is a per-quantile bisection with no joint model, and the resulting curves can
cross.

### `sas`

Sinh-arcsinh inverse link with learned skewness (`epsilon`) and
tail-weight (`delta`) parameters. Cannot be combined with `linkwiggle(...)`
or with blended/mixture links.

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
