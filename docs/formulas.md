# Formula DSL reference

Every model in `gamfit` is specified as a Wilkinson-style formula:

```
response ~ term + term + ... + option(...)
```

Terms are joined with `+`. Multiplication and `:` are *not* used — interaction
structure comes from multivariate smooths (`s(x1, x2)`, `te(x1, x2)`,
`matern(...)`, `duchon(...)`) rather than `x1*x2`.

This page is a reference for every term you can put on the right-hand side,
every option each term accepts, and the formula-level options (`link(...)`,
`linkwiggle(...)`, `timewiggle(...)`, `survmodel(...)`).

## The response (left of `~`)

| Response | What it triggers |
| --- | --- |
| `y` (continuous) | Gaussian family, identity link (default). |
| `y` (binary `{0, 1}`) | Binomial family, logit link (default). |
| `y` (count, with `link(type=log)` and explicit family) | Poisson. |
| `y` (positive continuous, with `link(type=log)`) | Gamma. |
| `Surv(entry, exit, event)` | Survival model. See [survival.md](survival.md). |

The family is auto-detected from the response column. Use `family=` or
`link()` to override.

## Linear and constrained coefficients

```
y ~ x                                # implicit penalized linear
y ~ linear(x)                        # same, explicit
y ~ linear(x, min=0)                 # box-constrained ≥ 0
y ~ linear(x, min=-1, max=1)         # box-constrained
y ~ nonnegative(x)                   # sugar for linear(x, min=0)
y ~ nonpositive(x)                   # sugar for linear(x, max=0)
y ~ bounded(x, min=0, max=1)         # exact interval transform (no ridge)
```

`linear(x, min=…, max=…)` keeps the ordinary penalized linear term but
projects the coefficient into `[min, max]`. `bounded(...)` instead applies
an exact interval transform — useful when you want hard bounds without the
ridge term distorting your prior toward zero.

`bounded(...)` supports a prior on the unit-scaled interior:

```
bounded(x, min=0, max=1, prior=uniform)
bounded(x, min=0, max=1, prior=center)               # Beta(2, 2)
bounded(x, min=0, max=1, target=0.5, strength=3)     # Beta from (target, strength)
bounded(x, min=0, max=1, beta_a=2.5, beta_b=2.5)     # explicit Beta(a, b)
```

`prior=` options:

- `none` — flat on the transformed scale, no penalty (default if `prior` is
  omitted with no other prior args).
- `uniform` / `log-jacobian` — flat on the original scale (log-Jacobian
  correction).
- `center` — Beta(2, 2) pulling toward the midpoint.

`target` ∈ `(min, max)` with `strength > 0` is the convenient way to write
"pull toward target with this much strength": it sets
`a = 1 + strength·z`, `b = 1 + strength·(1−z)`, where
`z = (target − min) / (max − min)`.

Aliases for box constraints: `linear`, `constrain`, `constraint`, `box`,
`bounded` (the latter with the transform rather than ridge).

## Random effects

```
y ~ x + group(site)
y ~ x + re(site)            # alias
```

Adds a random intercept per level of the grouping column. The column may be
string- or integer-valued. Only random intercepts are supported — no random
slopes.

## Univariate smooths

```
y ~ s(x)                    # P-spline (B-spline + difference penalty)
y ~ smooth(x)               # alias
y ~ s(x, k=15)              # 15-dim basis
y ~ s(x, knots=10)          # 10 interior knots
y ~ s(x, degree=3, penalty_order=2)
y ~ s(x, type=ps)           # explicit P-spline
y ~ s(x, double_penalty=true)
```

Default `s(x)` for a single covariate is a cubic P-spline with a second-order
difference penalty. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `k` (or `basis_dim`) | auto from data | Total basis dimension. |
| `knots` | auto | Number of interior knots. Cannot combine with `k`. |
| `degree` | 3 | Polynomial degree of the B-spline. |
| `penalty_order` | 2 | Derivative order penalised (1 = slope, 2 = curvature). |
| `type` | `ps` (1-D), `tps` (2+D) | `ps`, `tps`, `matern`, `duchon`. |
| `double_penalty` | `true` | Add a ridge penalty alongside the difference penalty. |

Default `k`: `clamp(unique_values / 4, 4, max(20, cbrt(unique_values)))`. You
rarely need to touch this.

## Multivariate smooths

```
y ~ s(x1, x2)                       # thin-plate (default for ≥ 2 args)
y ~ tps(x1, x2)                     # alias
y ~ thinplate(x1, x2)               # alias
y ~ matern(x1, x2, x3)
y ~ duchon(x1, x2, x3)
y ~ te(x, z)                        # tensor product
y ~ tensor(x, z)                    # alias
```

### Thin-plate spline (`tps`, `thinplate`, `s(x1, x2)`)

Radial-basis surface smooth with thin-plate kernel.

| Option | Default | Meaning |
| --- | --- | --- |
| `centers` (or `k`) | auto | Number of radial basis centres. |
| `length_scale` | 1.0 | Global length scale. |
| `double_penalty` | `true` | Ridge + main penalty. |
| `scale_dims` | `false` | Standardize inputs per-axis before kernel eval. |

### Matérn (`matern`)

Radial basis with Matérn covariance kernel.

| Option | Default | Meaning |
| --- | --- | --- |
| `centers` (or `k`) | auto | Number of centres. |
| `length_scale` | 1.0 | Global length scale. |
| `nu` | `5/2` | Smoothness. Options: `1/2`, `3/2`, `5/2`, `7/2`, `9/2`. |
| `include_intercept` | `false` | Append a constant column. |
| `double_penalty` | `true` | Ridge + main penalty. |
| `scale_dims` | `false` | Per-axis anisotropy (learns axis contrasts). |

Higher `nu` gives smoother sample paths; `5/2` is the standard "smooth but
not analytic" choice.

### Duchon (`duchon`)

Radial basis with triple-operator regularization (mass + tension +
stiffness). Scale-free unless `length_scale` is given.

| Option | Default | Meaning |
| --- | --- | --- |
| `order` | auto | Polynomial nullspace order. `0`, `Linear`, or `Degree(d)`. |
| `power` | auto | Kernel power. |
| `centers` (or `k`) | auto | Number of centres. |
| `length_scale` | none | Optional global scale (hybrid mode). |
| `scale_dims` | `false` | Per-axis shape contrasts. Scale-free by default. |

Three independent penalties (mass, tension, stiffness) get their own
smoothing parameters under REML — this is the three-part penalty structure
that distinguishes `gamfit` from libraries with a single curvature penalty.

### Tensor product (`te`, `tensor`)

Kronecker product of univariate B-spline bases. Lets each axis have its own
smoothing parameter — appropriate when axes have different units (e.g. space
× time).

| Option | Default | Meaning |
| --- | --- | --- |
| `k` | auto, per margin | Basis dim per margin. |
| `knots` | auto, per margin | Interior knots per margin. |
| `degree` | 3 | Polynomial degree (all margins). |
| `double_penalty` | `true` | Ridge + main penalty. |

### Picking the right smooth

| You have... | Use... |
| --- | --- |
| One covariate | `s(x)` (P-spline). |
| Two coordinates in the same units (lat, lon) | `s(x, y)` (thin-plate) or `matern(x, y)`. |
| Coordinates in different units (space × time) | `te(x, t)`. |
| 3+ coordinates, especially in different units | `duchon(...)` with `scale_dims=true`, or `matern(...)`. |
| You want to control wiggliness directly | `matern(...)` with `nu`. |
| You want scale-free behaviour | `duchon(...)` without `length_scale`. |

## Adaptive anisotropy

For any multi-d smooth that supports it, add `scale_dims=true` (or set
`scale_dimensions=True` on `fit()`) to let the model learn how much to
shrink each axis independently:

```python
gamfit.fit(
    df,
    "y ~ matern(pc1, pc2, pc3, pc4)",
    scale_dimensions=True,
)
```

- **Matérn / hybrid Duchon (with `length_scale`):** optimizes a global scale
  plus centered per-axis contrasts.
- **Pure Duchon (no `length_scale`):** optimizes only the centered per-axis
  contrasts — stays scale-free.
- **Thin-plate / tensor:** standardizes inputs per axis before evaluating
  the kernel.

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

| `link(type=...)` | What you get |
| --- | --- |
| `identity` | `g⁻¹(η) = η`. Default for Gaussian. |
| `logit` | `g⁻¹(η) = 1/(1+e^{-η})`. Default for binomial. |
| `probit` | `g⁻¹(η) = Φ(η)`. |
| `cloglog` | `g⁻¹(η) = 1 − e^{−e^{η}}`. |
| `log` | `g⁻¹(η) = e^η`. For counts and positive-continuous. |
| `sas` | Sinh-arcsinh skewed link. Don't combine with `linkwiggle`. |
| `beta-logistic` | Bounded link. Don't combine with `linkwiggle`. |
| `blended(a, b, …)` | Mixture of two or more component inverse links (e.g. `logit`, `probit`, `cloglog`, `loglog`, `cauchit`). |
| `flexible(base)` | A spline offset from a base link. Automatically enables `linkwiggle`. |

You can also set it from the Python side with `link=` on `fit()`:

```python
gamfit.fit(df, "case ~ s(age)", link="logit")
```

`link(type=...)` in the formula and `link=` on `fit()` are equivalent; if
both are set, the formula-level option wins.

## `linkwiggle` — flexible link offset

```
y ~ s(x) + link(type=flexible(probit))
y ~ s(x) + linkwiggle(internal_knots=10)
y ~ s(x) + linkwiggle(degree=2, internal_knots=8, penalty_order=all)
```

Adds a spline offset to a base link, letting the data correct for link
misspecification while encoding the belief that the base link is
approximately right.

| Option | Default | Meaning |
| --- | --- | --- |
| `internal_knots` | ~10 | Interior knots for the offset spline. |
| `degree` | 3 | Polynomial degree (≥ 1). |
| `penalty_order` | `all` (= 1, 2, 3) | Which derivatives to penalise. Comma-separated `slope`, `curvature`, `curvature-change` (or `1, 2, 3`), or `all`. |
| `double_penalty` | `true` | Ridge + main penalty. |

Available with `identity`, `log`, `logit`, `probit`, `cloglog` base links.
Not with `sas`, `beta-logistic`, or `blended(...)`.

## `timewiggle` — survival-only baseline offset

```
Surv(entry, exit, event) ~ age + timewiggle(internal_knots=8)
```

Same options as `linkwiggle`. Adds a spline offset to the survival time
basis, so the baseline hazard can flex away from a parametric form. Only
valid in survival formulas. See [survival.md](survival.md).

## `survmodel` — survival configuration

```
Surv(entry, exit, event) ~ age + survmodel(spec=net, distribution=gaussian)
```

| Option | Meaning |
| --- | --- |
| `spec` | Baseline specification, e.g. `net`. |
| `distribution` | Distributional assumption, e.g. `gaussian`. |

These typically pair with `survival_likelihood=` on `fit()` and the
`--predict-noise` route. See [survival.md](survival.md) for full details.

## A few full examples

```python
# Simple GAM with a smooth and a linear term
"y ~ s(bmi) + age"

# Spatial smooth with per-axis anisotropy
"z ~ matern(lat, lon, scale_dims=true)"

# Tensor of space × time
"y ~ te(x_coord, time, k=8)"

# 4-D scale-free Duchon
"y ~ duchon(pc1, pc2, pc3, pc4, centers=50)"

# Constrained linear + bounded proportion + linear age
"y ~ nonnegative(cost) + bounded(prop, min=0, max=1, target=0.5, strength=2) + age"

# Logistic with flexible link
"case ~ s(age) + link(type=flexible(probit)) + linkwiggle(internal_knots=6)"

# Survival with transformation likelihood + smooth covariate
"Surv(entry, exit, event) ~ s(age) + bmi"

# Random intercept per site
"y ~ smooth(x) + group(site)"
```
