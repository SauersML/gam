# Formula DSL reference

Every model in `gamfit` uses a Wilkinson-style formula:

```
response ~ term + term + ... + option(...)
```

Terms are joined with `+`. `*` and `:` are not supported; interactions
come from multivariate smooths (`s(x1, x2)`, `te(x1, x2)`,
`matern(...)`, `duchon(...)`).

This page lists each right-hand-side term, its options, and the
formula-level configuration terms (`link(...)`, `linkwiggle(...)`,
`timewiggle(...)`, `survmodel(...)`).

## Response (left of `~`)

| Response | Default behaviour |
| --- | --- |
| `y` continuous | Gaussian family, identity link. |
| `y` binary `{0, 1}` | Binomial family, logit link. |
| `y` non-negative integer, with `link(type=log)` | Poisson. |
| `y` positive continuous, with `link(type=log)` | Gamma. |
| `Surv(entry, exit, event)` | Survival model. See [survival.md](survival.md). |

The family is inferred from the response. When `link(type=log)` is set,
Poisson vs Gamma is chosen by whether `y` is integer-valued — `family=`
is optional in that case. `family=` accepts `gaussian`, `binomial`
(aliases `binomial-logit`, `binomial-probit`, `binomial-cloglog`),
`latent-cloglog-binomial`, `poisson`, and `gamma`. Survival,
transformation-normal, and Bernoulli marginal-slope families are
selected through `Surv(...)` or dedicated CLI flags rather than
`family=`.

## Linear and constrained coefficients

```
y ~ x                                # implicit penalized linear
y ~ linear(x)                        # explicit linear
y ~ linear(x, min=0)                 # box-constrained coefficient >= 0
y ~ linear(x, min=-1, max=1)         # box-constrained coefficient
y ~ nonnegative(x)                   # sugar for linear(x, min=0)
y ~ nonpositive(x)                   # sugar for linear(x, max=0)
y ~ bounded(x, min=0, max=1)         # exact interval transform on x
```

`linear(x, min=..., max=...)` keeps a penalized linear term and
projects the coefficient into `[min, max]`. Accepted aliases for the
same function: `linear`, `constrain`, `constraint`, `box`. Each accepts
`min`/`lower` and `max`/`upper`. `constrain()`/`constraint()`/`box()`
requires at least one of those four to be set.

`bounded(x, min, max)` applies an exact interval transform to `x`. It
is a distinct term type from `linear`, not a constrained linear.
Required options: `min` and `max` (finite, `min < max`).

### bounded() priors

`bounded()` accepts one of `prior=`, `target=`+`strength=`, or no
prior:

```
bounded(x, min=0, max=1, prior=uniform)
bounded(x, min=0, max=1, prior=center)
bounded(x, min=0, max=1, target=0.5, strength=3)
```

`prior=` values:

- `none` — flat on the transformed scale, no penalty.
- `uniform` (aliases `log-jacobian`, `log_jacobian`, `jacobian`) — flat
  on the original scale, applied as a log-Jacobian correction.
- `center` — `Beta(2, 2)` toward the midpoint.

`target` plus `strength` is shorthand for a Beta prior:
`a = 1 + strength * z`, `b = 1 + strength * (1 - z)` with
`z = (target - min) / (max - min)`. `target` must lie strictly between
`min` and `max`; `strength` must be positive.

`prior=`, `target`/`strength`, and the (legacy) `pull=` shorthand are
mutually exclusive.

## Random effects and factor smooths

```
y ~ x + group(site)
y ~ x + re(site)                         # alias of group()
y ~ s(x, by=site) + site                 # separate smooth per factor level
y ~ s(x, by=z)                           # numeric varying-coefficient smooth
y ~ s(x, site, bs="fs")                  # partial-pooling random smooths
y ~ fs(x, site)                          # alias for bs="fs"
y ~ s(site, x, bs="sz") + s(x)           # sum-to-zero deviations
y ~ sz(site, x)                          # alias for bs="sz"
y ~ s(site, x, bs="re") + group(site)    # random slopes + random intercepts
y ~ s(x, site, bs="re") + group(site)    # column order interchangeable
```

`group(name)` and `re(name)` add a random intercept per level of a
categorical column. The single argument must be a column name.

A `by=` option on a smooth multiplies the smooth basis by a column.
For a categorical `by=`, one smooth is built per kept level; include
the factor main effect (e.g. `+ site` or `+ group(site)`) when level
offsets should be identifiable. For a numeric `by=`, the result is a
varying-coefficient smooth.

`bs="fs"` builds factor smooths: each group gets its own curve,
including penalized null-space components (intercept and linear
trend). `m` sets the order of the single null-space shrinkage penalty
(default `m=2`, second-order). New groups at prediction time
contribute zero for the term.

`bs="sz"` builds sum-to-zero factor-smooth deviations; deviations sum
to zero across factor levels at each spline coefficient. Use with a
population smooth such as `s(x)`.

`bs="re"` with one grouping column and one numeric column builds
random slopes as a factor-by-linear basis with an identity ridge
penalty.

The `fs`/`sz`/`re` factor-smooth path accepts `k`, `basis_dim`,
`knots`, `degree`, `penalty_order`, `m` (fs only), and
`double_penalty`. It requires exactly two variables: one categorical
or binary grouping column and one continuous (or binary) numeric
column.

## Univariate smooths

```
y ~ s(x)                    # P-spline (B-spline + difference penalty)
y ~ smooth(x)               # alias of s()
y ~ s(x, k=15)              # basis dimension 15
y ~ s(x, knots=10)          # 10 interior knots
y ~ s(x, degree=3, penalty_order=2)
y ~ s(x, type=ps)           # explicit P-spline
y ~ s(x, double_penalty=true)
```

For a single covariate, `s(x)` defaults to a cubic P-spline with a
second-order difference penalty.

| Option | Default | Meaning |
| --- | --- | --- |
| `k` (`basis_dim`) | from data | Total basis dimension. |
| `knots` | from data | Number of interior knots. Cannot combine with `k`. |
| `degree` | 3 | Polynomial degree of the B-spline. |
| `penalty_order` | 2 | Order of the difference penalty. |
| `type` | B-spline (1-D), `tps` (≥2-D) | One of `tps`, `matern`, `duchon`, `sphere`, `cyclic`. `ps` and the bare default both produce the 1-D B-spline; there is no separate `ps` dispatch. |
| `double_penalty` | `true` | Add a ridge penalty alongside the difference penalty. |
| `bc` | `free` | Same boundary condition on both ends: `free`, `clamped`, or `anchored`. |
| `bc_left`, `bc_right` | `free` | Per-endpoint boundary condition. Aliases: `left_bc`/`right_bc`, `start_bc`/`end_bc`. |
| `anchor`, `anchor_left`, `anchor_right` | 0 | Anchor value for `anchored`. Aliases: `anchor_value`, `value`, `left_anchor`, `right_anchor`. |
| `by` | none | Column for varying-coefficient or by-factor smooths. |

The 1-D B-spline path accepts these options plus `periodic`, `period`,
`periods`, `period_start`, `period_end`, `origin`, `identifiability`.

Default interior knots:
`clamp(unique_values / 4, 4, max(20, cbrt(unique_values)))`. The basis
dimension is then `k = internal_knots + degree + 1`. Passing both `k`
and `knots` is an error.

### Boundary-conditioned 1-D smooths {#boundary-conditioned-1d-smooths}

Boundary-condition values: `free`/`none`/`open`,
`clamped`/`zero_derivative`, `anchored`/`zero`/`zero_value`. `clamped`
forces zero first derivative at the endpoint; `anchored` pins the
endpoint value (anchor defaults to 0, currently the only supported
anchor value).

```
y ~ s(x, bc=clamped)                       # zero slope at both endpoints
y ~ s(x, bc_left=clamped)                  # zero slope at the start, free at end
y ~ s(x, bc_left=anchored, anchor_left=0)  # endpoint value pinned to 0
y ~ s(x, start_bc=clamped, end_bc=anchored, end_anchor=0)
```

`bc(x)` (aliases `boundary(x)`, `boundary_conditioned(x)`) is sugar for
`s(x, bc=clamped)`; if any anchor option is set, it defaults to
`bc=anchored`. Per-side overrides are read directly by the smooth
builder.

## Multivariate smooths

```
y ~ s(x1, x2)                # thin-plate (default for >=2 args)
y ~ tps(x1, x2)              # alias of thin-plate
y ~ thinplate(x1, x2)        # alias
y ~ thin_plate(x1, x2)       # alias
y ~ matern(x1, x2, x3)
y ~ duchon(x1, x2, x3)
y ~ te(x, z)                 # tensor product
y ~ tensor(x, z)             # alias
y ~ interaction(x, z)        # alias of te()
```

### Thin-plate (`tps`, `thinplate`, multivariate `s(...)`)

Radial-basis surface smooth with thin-plate kernel.

| Option | Default | Meaning |
| --- | --- | --- |
| `centers` (`k`, `basis_dim`) | auto | Number of radial centres. |
| `length_scale` | auto (data-derived) | Global length-scale init. |
| `double_penalty` | `true` | Ridge + main penalty. |
| `scale_dims` | `false` | Per-axis input standardisation. |
| `include_intercept` | `false` | Append a constant column. |
| `by`, `identifiability` | — | See common options above. |

### Matérn (`matern`)

Radial basis with Matérn covariance kernel.

| Option | Default | Meaning |
| --- | --- | --- |
| `centers` (`k`, `basis_dim`) | auto | Number of centres. |
| `length_scale` | auto | Global length-scale init. |
| `nu` | `5/2` | Smoothness, one of `1/2`, `3/2`, `5/2`, `7/2`, `9/2`. |
| `include_intercept` | `false` | Append a constant column. |
| `double_penalty` | `true` | Ridge + main penalty. |
| `scale_dims` | `false` | Per-axis anisotropy (learns per-axis log-scales). |

Higher `nu` gives smoother sample paths. `nu=1/2` is rejected for
`d >= 2` because the exponential kernel's Laplacian is singular at
zero, which makes the operator-collocation penalty non-invertible.

### Duchon (`duchon`)

Radial basis with triple-operator regularization (mass + tension +
stiffness). Scale-free unless `length_scale` is given.

| Option | Default | Meaning |
| --- | --- | --- |
| `order` (`nullspace_order`) | auto | Polynomial nullspace order `p`. Polynomial block has `C(d + p, d)` columns (`p=0` → constant only, `p=1` (Linear) → `d+1` columns, `p=2` → `(d+1)(d+2)/2`). |
| `power` (`p`) | auto | Riesz fractional smoothness `s`. Auto-resolved against `d` and the active operator penalties. |
| `centers` (`k`, `basis_dim`) | auto | Number of centres. |
| `length_scale` | none (scale-free) | Optional global scale. Without it, the kernel is pure polyharmonic; with it, the kernel is the hybrid Duchon-Matérn (κ = 1/length_scale). |
| `scale_dims` | `false` | Per-axis contrasts. |
| `periodic`, `period`, `period_start`, `period_end` | — | 1-D cyclic Duchon (see below). |

`duchon()` rejects `double_penalty`; the three operator penalties
(mass, tension, stiffness) each get their own smoothing parameter
under REML.

Duchon is not TPS. TPS forces `m = ⌊d/2⌋+1`, so its polynomial
nullspace `C(d+m-1, d)` grows combinatorially with `d` (e.g. 735,471
columns at `d=16`). Duchon keeps `p` small (typically `Linear`, giving
`d+1` polynomial columns) and grows the Riesz power `s` instead, so
the polynomial block stays linear in `d` at any ambient dimension.

### Tensor product (`te`, `tensor`, `interaction`)

Kronecker product of univariate B-spline bases. Each axis has its own
smoothing parameter. Use when axes have different units (space × time).

| Option | Default | Meaning |
| --- | --- | --- |
| `k` (`basis_dim`) | auto, per margin | Basis dim per margin. List form `k=[k1, k2]` accepted. |
| `knots` | auto, per margin | Interior knots per margin. List form accepted. |
| `degree` | 3 | Polynomial degree (all margins). |
| `penalty_order` | 2 | Difference-penalty order. |
| `double_penalty` | `false` | Ridge alongside per-margin penalties. |
| `bc` | none | Per-margin boundary conditions (list form). |
| `periodic`, `period`, `periods`, `origin`, `origins` | — | Per-margin periodicity (see below). |
| `by`, `identifiability` | — | See common options above. |

`k` and `knots` cannot both be set. Margins requested as a single
value are broadcast across all margins.

### Picking the right smooth

| Situation | Term |
| --- | --- |
| One covariate | `s(x)` |
| Two coordinates, same units | `s(x, y)` or `matern(x, y)` |
| Coordinates in different units | `te(x, t)` |
| Three or more coordinates | `duchon(...)` with `scale_dims=true`, or `matern(...)` |
| Direct control of wiggliness | `matern(..., nu=...)` |
| Scale-free behaviour | `duchon(...)` without `length_scale` |

![four smooth families on the same dataset](images/smooth_zoo.png)

### Periodic and cyclic smooths {#periodic-cyclic-smooths}

```
y ~ s(theta, periodic=true, period=6.283)             # cyclic 1-D B-spline
y ~ cyclic(theta, period_start=0, period_end=6.283)   # alias of s(..., periodic=true)
y ~ periodic(theta, period=6.283)                     # alias
y ~ cp(theta, period=6.283)                           # alias
y ~ cc(day_of_week, period=7)                         # mgcv-style alias

# Tensor with one or more periodic margins
y ~ te(theta, h,  periodic=[0], period=[2*pi, None])
y ~ te(theta, phi, periodic=[0, 1], period=[2*pi, 2*pi])
y ~ te(day, hour, bc=['periodic','periodic'], periods=[7, 24], origins=[0, 0])
```

`cyclic`, `periodic`, `cc`, `cp` are aliases of `s(..., type=cyclic)`.

`periodic=[axes]` lists zero-based axis indices that wrap. `period=` or
`periods=` holds one positive period per margin (`None` for
non-periodic margins). `origin=` / `origins=` fixes the half-open
cyclic domain start when the sample does not contain the boundary.
`period_start`/`period_end` (1-D) override the data range when the
angular domain is wider than the observed sample.

Period values can be plain numbers (e.g. `6.283185307`) or the
symbolic constants `pi` / `PI` / `tau` / `TAU` (case-insensitive),
optionally multiplied by a single numeric literal: `pi`, `2*pi`,
`pi*2`, `.5*pi`, and `tau` are all accepted. `pi/2` and `2*3.14`
(number × number) are not. The same rule applies inside list syntax
(`period=[2*pi, None]`).

`duchon()` accepts `periodic` only for 1-D inputs. Formula-level
multivariate `duchon(..., periodic=true)` is rejected.

### Intrinsic S² (sphere) smooth

```
y ~ sphere(lat, lon)                                    # Wahba kernel, m=2
y ~ sos(lat, lon)                                       # alias
y ~ spherical(lat, lon)                                 # alias
y ~ s(lat, lon, type=sphere)                            # equivalent
y ~ s(lat, lon, bs=sos)                                 # mgcv-style alias
y ~ sphere(lat, lon, m=3, radians=true, k=64)
y ~ sphere(lat, lon, method=harmonic, max_degree=6)
```

`sphere()`, `sos()`, and `spherical()` require exactly two columns
(latitude, longitude).

| Option | Default | Meaning |
| --- | --- | --- |
| `method` | `wahba` | `wahba` (reproducing kernel) or `harmonic` (spherical harmonics). Aliases: `wahba_sobolev`, `wahba_pseudo`/`mgcv`/`sos`, `sh`. |
| `m` (`order`, `penalty_order`) | 2 | Wahba pseudo-spline order, integer in `{1, 2, 3, 4}`. |
| `max_degree` (`max_l`, `harmonic_degree`, `l`) | auto | Maximum harmonic degree L; basis dim = `L(L+2)`. |
| `centers` (`k`, `basis_dim`) | auto | Wahba centre count. |
| `radians` / `units=radians` | `false` | Treat lat/lon as radians (default: degrees). |
| `double_penalty` | `true` | Add a ridge-like null-space shrinkage penalty. |
| `kernel` (`wahba_kernel`) | `sobolev` | `sobolev` (default) or `pseudo`/`mgcv`/`sos`. |

Both Wahba and harmonic constructions are rotation-invariant. The
harmonic basis is fixed-rank with a diagonal Laplace-Beltrami squared
penalty; Wahba uses centres via a closed-form reproducing kernel.

## Adaptive anisotropy

Multi-dimensional smooths that support `scale_dims=true` learn
per-axis shrinkage. Setting `scale_dimensions=True` on `fit()`
enables it globally across compatible spatial smooths.

```python
gamfit.fit(df, "y ~ matern(pc1, pc2, pc3, pc4)", scale_dimensions=True)
```

- Matérn and hybrid Duchon (with `length_scale`): a global scale plus
  centered per-axis contrasts.
- Pure Duchon (no `length_scale`): centered per-axis contrasts only;
  remains scale-free.
- Thin-plate: per-axis input standardisation.
- Tensor: each margin has its own smoothing parameter without this
  option.

## Link function

```
y ~ x + link(type=identity)
y ~ x + link(type=logit)
y ~ x + link(type=probit)
y ~ x + link(type=cloglog)
y ~ x + link(type=log)
y ~ x + link(type=sas)
y ~ x + link(type=beta-logistic)
y ~ x + link(type=blended(logit, probit))
y ~ x + link(type=flexible(probit))
```

| `link(type=...)` | Inverse link |
| --- | --- |
| `identity` | `eta`. Default for Gaussian. |
| `logit` (alias `binomial-logit`) | `1 / (1 + exp(-eta))`. Default for binomial. |
| `probit` (alias `binomial-probit`) | `Phi(eta)`. |
| `cloglog` (alias `binomial-cloglog`) | `1 - exp(-exp(eta))`. |
| `log` | `exp(eta)`. For counts and positive-continuous. |
| `sas` | Sinh-arcsinh skewed link. Not compatible with `linkwiggle`. |
| `beta-logistic` | Bounded link. Not compatible with `linkwiggle`. |
| `blended(a, b, ...)` / `mixture(a, b, ...)` | Mixture of component inverse links from `logit`, `probit`, `cloglog`, `loglog`, `cauchit`. |
| `flexible(base)` | Spline offset from a base link; enables `linkwiggle`. |

`link(type=...)` in the formula and `link=` on `fit()` are equivalent.
The formula value wins if both are set.

```python
gamfit.fit(df, "case ~ s(age)", link="logit")
```

## `linkwiggle` — flexible link offset

```
y ~ s(x) + link(type=flexible(probit))
y ~ s(x) + linkwiggle(internal_knots=10)
y ~ s(x) + linkwiggle(degree=2, internal_knots=8, penalty_order=all)
```

Adds a spline offset to a base link. The base link is the prior; the
data can correct for link misspecification.

| Option | Default | Meaning |
| --- | --- | --- |
| `internal_knots` | ~10 | Interior knots for the offset spline (must be > 0). |
| `degree` | 3 | Polynomial degree (>= 1). |
| `penalty_order` | `all` (1, 2, 3) | Which derivatives to penalise. Comma-separated `slope`, `curvature`, `curvature-change` (or `1`, `2`, `3`), or `all`. |
| `double_penalty` | `true` | Ridge + main penalty. |

Compatible base links: `identity`, `log`, `logit`, `probit`,
`cloglog`. Not `sas`, `beta-logistic`, or `blended(...)`.

`linkwiggle()` takes named options only; positional arguments are
rejected.

## `timewiggle` — survival baseline offset

```
Surv(entry, exit, event) ~ age + timewiggle(internal_knots=8)
```

Same options as `linkwiggle`. Adds a spline offset to the survival
time basis so the baseline hazard can deviate from a parametric form.
Survival formulas only, with a non-linear scalar `baseline_target`
such as `weibull`, `gompertz`, or `gompertz-makeham`. See
[survival.md](survival.md).

## `survmodel` — survival configuration

```
Surv(entry, exit, event) ~ age + survmodel(distribution=gaussian)
```

| Option | Meaning |
| --- | --- |
| `distribution` | Residual distribution. Case-insensitive. Accepted: `gaussian`/`probit`, `gumbel`/`cloglog`, `logistic`/`logit`. |

Survival likelihood and baseline target are selected via CLI flags
(`--survival-likelihood`, `--baseline-target`), not through `spec=` on
`survmodel(...)`.

`survmodel()` requires at least one named option and takes named
arguments only. Only one `survmodel(...)` term is allowed per formula.
Pair it with `survival_likelihood=` on `fit()`. See
[survival.md](survival.md).

## Examples

```python
# GAM with a smooth and a linear term
"y ~ s(bmi) + age"

# Spatial smooth with per-axis anisotropy
"z ~ matern(lat, lon, scale_dims=true)"

# Tensor of space x time
"y ~ te(x_coord, time, k=8)"

# 4-D scale-free Duchon
"y ~ duchon(pc1, pc2, pc3, pc4, centers=50)"

# Constrained linear + bounded proportion + linear age
"y ~ nonnegative(cost) + bounded(prop, min=0, max=1, target=0.5, strength=2) + age"

# Logistic with flexible link
"case ~ s(age) + link(type=flexible(probit)) + linkwiggle(internal_knots=6)"

# Survival with smooth covariate
"Surv(entry, exit, event) ~ s(age) + bmi"

# Random intercept per site
"y ~ smooth(x) + group(site)"
```

* [Difference smooths](difference-smooths.md)
