# Inspecting terms

A fitted GAM is a sum of terms, and most questions about it concern one term at
a time: what shape did `s(age)` take, how far apart are the levels of
`education`, and where the surface `te(lon, lat)` rises. This page covers the
tools that answer them.

| Tool | Answers |
| --- | --- |
| `model.partial_dependence(term)` | A term's curve, surface or per-level effect, with pointwise intervals and a simultaneous band. |
| `model.plot_terms()` | The same numbers drawn with matplotlib, one panel per term. |
| `gam partial-effect MODEL --term TERM` | The same numbers from the CLI, as JSON or CSV. |
| `model.term_blocks` | Each term's name, kind and coefficient columns. |
| `model.variance_share(data)` | How much of the fitted predictor's variance each term carries. |
| `model.smooth_significance()` | Per-smooth Wald tests. |
| `model.difference_smooth(...)` | The difference between two levels' smooths, with bands. See [difference-smooths.md](difference-smooths.md). |

## Partial effects

```python
import gamfit

model = gamfit.fit(train, "wage ~ s(year) + s(age) + education")
effect = model.partial_dependence("s(age)")

effect.x                      # the grid: 100 ages over the training range
effect.fit, effect.se         # the curve and its standard error
effect.lower, effect.upper    # 95% pointwise intervals
effect.simultaneous_lower     # 95% simultaneous band over the whole grid
effect.simultaneous_upper
```

The result is a [`gamfit.results.PartialEffect`](api-reference.md#gamfit.results.PartialEffect).
The term names are those of `model.term_blocks`.

The curve is the term's own contribution to the linear predictor,
`f_t(x) = X_t(x) β_t`. Each term's basis carries the identifiability constraint
the fit applied, so the curve is the centred effect the model estimated. Only
the term's own columns enter it, which is why the result needs no reference
data and does not depend on the other covariates' values.

Under a non-identity link the curve stays on the link scale
(`effect.scale == "linear_predictor"`). It is not an average over the data on
the response scale. For a response-scale quantity at chosen rows, use
`model.predict(..., interval=...)`.

### Which covariance

The standard error is `sqrt(diag(X_t V_t X_tᵀ))`. `V_t` is the term's block of
the coefficient covariance the fit publishes. When the fit carries it, this is
the covariance corrected for smoothing-parameter uncertainty. `effect.covariance_source`
names it: `"smoothing-corrected"` or `"conditional"`.

### Pointwise intervals and simultaneous bands

`lower`/`upper` are `fit ∓ z · se`, where `z` is the two-sided normal quantile
for `level` (`effect.pointwise_critical`). They cover the true value at each
grid point separately. Across the curve, about `level` of the points are
covered, which is Nychka's across-the-function property of the Bayesian
intervals.

`simultaneous_lower`/`simultaneous_upper` are `fit ∓ c · se`. They cover the
whole curve over the grid at once with probability `level`. The critical value
`c` (`effect.simultaneous_critical`) is the `level` quantile of the maximum
absolute standardized error over the grid. It comes from `effect.simulations`
draws of the posterior curve, seeded with `effect.seed`, so a repeated call
returns the same band.

The number of draws is not fixed. It is the smallest count for which the
Monte-Carlo standard error of the band's attained coverage is at most 5% of its
nominal miss rate. That is 7,600 draws at `level=0.95` and 39,600 at `0.99`.
The count depends only on `level`, whatever the grid or the term. The attained
coverage of a band built from `N` simulated maxima follows a
`Beta(⌈level·N⌉, N + 1 − ⌈level·N⌉)` distribution.

To test whether a term is flat or zero everywhere, use the simultaneous band.
The pointwise intervals are for reading the curve one point at a time.

### Factor terms

A factor term has one effect per level:

```python
edu = model.partial_dependence("education")
edu.labels()                  # ["1. < HS Grad", "2. HS Grad", ...]
edu.fit, edu.lower, edu.upper
```

`grid` holds level codes. `axis_levels[0]` maps each code to its label.
`plot_terms` draws one point per level, with both intervals as error bars.

### Two-axis terms

A `te(x, z)`, `ti(x, z)` or two-dimensional thin-plate term sweeps a product
grid of `n_points` values per axis over the training box. `surface()`
reshapes any series onto that grid:

```python
surf = model.partial_dependence("te(lon, lat)", n_points=40)
lon, lat = surf.axis_values          # the swept values of each axis
fit = surf.surface("fit")            # fit[i, j] is at (lon[i], lat[j])
se = surf.surface("se")
```

A factor-by smooth `s(x, by=g)` whose block spans every level sweeps `x` and
`g` together. A per-level block holds its own level, which `effect.held` names.

### Numeric-by terms

For `s(x, by=z)` the curve is the coefficient function `f(x)`. The term enters
the predictor as `z · f(x)`, so `effect.quantity == "coefficient_function"`,
`effect.contribution == "z * f(x)"` and `effect.held == {"z": 1.0}`.

### Your own grid

```python
effect = model.partial_dependence("s(age)", grid=np.linspace(20, 60, 41))
```

For a multi-axis term, pass a `(n, d)` array with columns in `effect.axes`
order, and factor axes as level codes. `surface()` needs the default product
grid, so it refuses a caller grid.

## Plotting

```python
axes = model.plot_terms()                    # every non-intercept term
model.plot_terms(["s(age)", "education"], level=0.99)
```

- A one-axis numeric term is drawn as a line with its pointwise interval and
  its simultaneous band shaded.
- A factor term is drawn as one point per level, with error bars.
- A numeric-by-factor term is drawn as one curve per level.
- A two-axis term is drawn as a filled contour of the surface, with dashed
  standard-error contours.

The helper only draws. Every number comes from `partial_dependence`. Pass `axes=`
(one matplotlib axes per term) to draw into an existing figure. It needs
matplotlib (`pip install gamfit[plot]`).

`model.plot(data, kind="prediction")` is not a term plot. It draws the full
model's prediction against the data's one feature column. That is a curve only
for a one-feature model, so it refuses data with more feature columns and
points to `plot_terms`.

## CLI

```bash
gam partial-effect model.gam --term 's(age)'                    # JSON to stdout
gam partial-effect model.gam --term 's(age)' --level 0.99 --out age.csv
gam partial-effect model.gam --term education --grid levels.csv --out edu.json
```

| Option | Meaning |
| --- | --- |
| `--term TERM` | Term name, as in `model.term_blocks`. |
| `--level VALUE` | Coverage of the pointwise intervals and the simultaneous band; default `0.95`. |
| `--n-points N` | Values per numeric axis on the default grid; default `100`. |
| `--grid FILE` | A CSV grid whose columns are exactly the term's axes, in any order. It has numbers for numeric axes and level labels for factor axes. |
| `--out FILE` | Write `.json` (the full record, as below) or `.csv` (one row per grid point: the axes, then `fit,se,lower,upper,simultaneous_lower,simultaneous_upper`). Without it, JSON goes to stdout. |

The JSON record has the same fields as `gamfit.results.PartialEffect`. Both come from one
Rust function, `gam_predict::partial_effect::partial_effect`, so the CLI and
Python return the same numbers.

## Comparison with pyGAM

| pyGAM | gamfit |
| --- | --- |
| `gam.generate_X_grid(term=i)` | Built in: `partial_dependence(term)` sweeps the training range. |
| `gam.generate_X_grid(term=i, meshgrid=True)` | `effect.axis_values`, `effect.surface(...)`. |
| `gam.partial_dependence(term=i, X=XX, width=0.95)` | `effect.lower`, `effect.upper`. |
| (none) | `effect.simultaneous_lower`, `effect.simultaneous_upper`. |
| Terms by index | Terms by name, as in `model.term_blocks`. |
