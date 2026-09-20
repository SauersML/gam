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

A link that several families admit does not choose the family. `log`, `sqrt`,
`inverse` and `inverse-squared` are each legal for several response families
(see [Link legality](#link-legality)), so pinning one of them without a
`family=` is an error that says the link does not determine a response family
and asks for one with `family=`. A variance function is a modelling choice, so
gamfit does not read it off whether `y` happens to be integer-valued:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
df = {"x": x, "count": rng.poisson(np.exp(1 + np.sin(2 * np.pi * x))),
      "prop": rng.gamma(5.0, np.exp(0.5 * np.cos(2 * np.pi * x)) / 5.0)}

gamfit.fit(df, "count ~ s(x)")                 # integer counts -> Poisson/log (auto)
gamfit.fit(df, "count ~ s(x)", family="poisson", link="log")  # explicit
gamfit.fit(df, "prop ~ s(x)", family="gamma", link="log")     # explicit
try:
    gamfit.fit(df, "prop ~ s(x)", link="log")
except gamfit.errors.FormulaError as err:      # name one with family=
    print(err)
```

## Setting family and link

The `family=` kwarg accepts `"gaussian"`, `"binomial"` (aliases
`"binomial-logit"`, `"binomial-probit"`, `"binomial-cloglog"`),
`"latent-cloglog-binomial"`, `"poisson"`, `"negative-binomial"`,
`"beta"`, `"gamma"`, `"inverse-gaussian"`, `"tweedie"`, `"student-t"` (see
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
import numpy as np
import gamfit
from scipy.stats import norm

rng = np.random.default_rng(0)
age = rng.uniform(30, 80, 400)
df = {"age": age, "case": (rng.uniform(size=age.size) < norm.cdf((age - 55) / 12)).astype(float)}

gamfit.fit(df, "case ~ s(age)", link="probit")
```

## Link functions

The engine recognises the following link types in formula `link(type=...)`
and the Python `link=` kwarg. There is no top-level `gam fit --link`
flag; CLI fits use formula-level `link(...)`.

### `identity`

Inverse link `eta`. Default for continuous Gaussian responses.

### `logit`

Inverse link `1 / (1 + exp(-eta))` (alias `binomial-logit`). Default for
binary `{0, 1}` responses.

### `probit`

Inverse link `Phi(eta)` (alias `binomial-probit`), the standard normal CDF.
Required for the Bernoulli marginal-slope family (see
[marginal-slope.md](marginal-slope.md)).

### `cloglog`

Inverse link `1 - exp(-exp(eta))` (alias `binomial-cloglog`). Used for
grouped discrete-time hazards and rare-event Bernoulli data.

### `log`

Inverse link `exp(eta)`. Pair with `family="poisson"` for counts and
`family="gamma"` or `family="inverse-gaussian"` for positive continuous
responses. With `family="gaussian"` it is a log-mean model with additive
normal noise; with `family="binomial"` it is the relative-risk model, whose
mean `exp(eta)` is a probability only on `eta < 0` (see
[Feasibility sets](#feasibility-sets)).

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
time, exposure = rng.uniform(0, 10, 400), rng.uniform(0.5, 2.0, 400)
df = {"time": time, "log_exposure": np.log(exposure),
      "count": rng.poisson(exposure * np.exp(0.5 + 0.5 * np.sin(time)))}

gamfit.fit(df, "count ~ s(time)",
           family="poisson", link="log", offset="log_exposure")
gamfit.fit(df, "case ~ s(age)", family="binomial", link="log")  # relative risk
```

Pass the offset column via `offset=`; do not include it on the formula RHS.

### `sqrt`

Inverse link `eta^2` (`eta = sqrt(mu)`), the variance-stabilising link for
counts. The mean map is only one-to-one on `eta > 0`, so that is its
feasibility set in every family.

### `inverse`

Inverse link `1 / eta`, canonical for the Gamma family. The mean is only
defined on `eta > 0`. The other spelling `1/mu` is refused with an error
naming `inverse`.

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
mu = 1.0 / (0.9 + 0.3 * np.sin(2 * np.pi * x))   # eta = 1/mu > 0 everywhere
df = {"x": x, "y": rng.gamma(5.0, mu / 5.0)}

gamfit.fit(df, "y ~ s(x)", family="gamma", link="inverse")
```

### `inverse-squared`

Inverse link `eta^(-1/2)` (`eta = 1 / mu^2`), canonical for the inverse
Gaussian family. The mean is only defined on `eta > 0`. Other spellings
(`1/mu^2`, `inv-squared`, `inv_squared`) are refused with an error naming
`inverse-squared`.

### Link legality

Legality is decided by support: a link is legal for a family when the link's
mean range meets the family's mean domain. The Gaussian, Poisson, Gamma and
inverse Gaussian families all take the whole power/log ladder; binomial takes
every probability link plus `log`. An illegal pairing is refused with the
family's legal links spelled out, generated from the same table the engine
checks, e.g.

```
illegal likelihood cell: response `negative-binomial` does not admit inverse link `identity`;
legal links for `negative-binomial`: log
```

| family | legal links |
| --- | --- |
| gaussian, poisson, gamma, inverse-gaussian | identity, log, sqrt, inverse, inverse-squared |
| binomial | logit, probit, cloglog, loglog, cauchit, log |
| negative-binomial, tweedie | log |
| beta | logit |

`family="binomial", link="identity"` stays refused: a linear probability GAM
is a Gaussian model and should be spelled as one (`family="gaussian"`).

An unknown link name is refused with the whole vocabulary. Link names are
case-insensitive and `_` is read as `-`, so `inverse_squared` and
`Inverse-Squared` are the same link.

### How non-canonical cells are fitted

The canonical cells (Gaussian-identity, Poisson-log, Gamma-log,
binomial-logit/probit/cloglog, …) keep their hand-written row kernels. Every
other cell is fitted by one generic exponential-dispersion kernel composed
from the family's variance function `V(mu)` and its derivatives and the
link's inverse `mu(eta)` and its derivatives. It supplies the exact
log-likelihood, score, observed information and the third and fourth
eta-derivatives that REML/LAML needs, so a non-canonical cell gets the same
exact outer gradient and Hessian as a canonical one. The hand-written kernels
are the oracles it is tested against. These cells are always evaluated on the
CPU.

### Feasibility sets

When a link's mean range is larger than the family's mean domain, the cell
is legal but the linear predictor is restricted to a feasibility set:

| cell | feasible `eta` |
| --- | --- |
| `sqrt`, `inverse`, `inverse-squared` (any family) | `eta > 0` |
| `identity` for poisson, gamma, inverse-gaussian | `eta > 0` |
| `log` for binomial | `eta < 0` |

There is no hand-supplied bound, clamp or jitter. The starting linear
predictor is built inside the set. An inner P-IRLS step or an outer trial
point that would put any weighted row outside the set is reported as an
inverse-link domain violation, and the step is halved until every row is
feasible again. A fit therefore only ever certifies at a mean that lies in
the family's domain at every observed row.

When the likelihood's maximum lies on the boundary of the set, as in
identity-Poisson with a group whose counts are all zero or log-binomial with
a group whose outcomes are all one, there is no interior maximum. The fit
fails with a typed error saying the maximum lies on the boundary of the
link's feasibility set. It does not return the last feasible iterate. Use
the family's canonical link, or remove the predictor or rows that force the
mean to the edge.

Prediction away from the data can still produce an infeasible `eta`. Such
points have no mean under this link and are refused rather than clipped.

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
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
mu = np.exp(0.5 + 0.4 * np.sin(2 * np.pi * x))
df = {"x": x, "y": rng.wald(mu, 20.0)}             # inverse Gaussian, mean mu

gamfit.fit(df, "y ~ s(x)", family="inverse-gaussian")              # 1/mu^2
gamfit.fit(df, "y ~ s(x)", family="inverse-gaussian", link="log")
```

### Dispersion families

Gamma, Beta, negative-binomial, and Tweedie can be fit as ordinary
mean models or as two-submodel dispersion fits with `noise_formula=`.
For the dispersion path, the secondary formula models Gamma shape, Beta
precision, negative-binomial size, or Tweedie inverse dispersion.

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
n = 400
x, age, year = rng.uniform(0, 1, n), rng.uniform(20, 70, n), rng.integers(2000, 2012, n).astype(float)
mean_rate, mean_prop = np.exp(1 + 0.02 * (age - 45)), 1 / (1 + np.exp(-np.sin(2 * np.pi * x)))
df = {"x": x, "age": age, "year": year,
      "rate": rng.negative_binomial(5, 5 / (5 + mean_rate)).astype(float),
      "prop": rng.beta(20 * mean_prop, 20 * (1 - mean_prop)),
      "claim": np.where(rng.uniform(size=n) < 0.3, 0.0,
                        rng.gamma(2.0, np.exp(0.02 * (age - 45) + 0.05 * (year - 2005)) / 2))}

gamfit.fit(df, "rate ~ s(age)", family="negative-binomial", link="log")
gamfit.fit(df, "prop ~ s(x)", family="beta", noise_formula="s(x)")
gamfit.fit(df, "claim ~ te(age, year)", family="tweedie(p=1.5)", link="log")
```

`negative_binomial_theta` / `--negative-binomial-theta` fixes the
negative-binomial size parameter when a constant-size model is desired.

### Student-t

`family="student-t"` is a heavy-tailed
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
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
df = {"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.2 + 0.3 * x)}

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
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 400)
df = {"x": x, "rare_event": (rng.uniform(size=x.size) < 0.1 / (1 + np.exp(-4 * (x - 0.5)))).astype(float)}

gamfit.fit(df, "rare_event ~ s(x)", family="binomial-logit", firth=True)
```

## Offsets and weights

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
age, exposure = rng.uniform(20, 80, 400), rng.uniform(0.5, 2.0, 400)
df = {"age": age, "log_exposure": np.log(exposure), "freq": rng.integers(1, 4, 400).astype(float),
      "count": rng.poisson(exposure * np.exp(0.3 + 0.03 * (age - 50)))}

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
