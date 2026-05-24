# Difference smooths

Difference smooths answer questions like “how does group B’s trajectory differ
from group A’s over `x`?”  The library supports the same core idioms used in
mgcv workflows and exposes a single covariance-aware post-fit API:

```python
model.difference_smooth(view="Time", group="Group", simultaneous=True)
```

The result is tidy: one row per grid point and pair, with `diff`, `se`, `lower`,
`upper`, the interval `level`, whether the band is `simultaneous`, and the
critical value used.

## Which parameterisation to use

### 1. Unordered by-factor smooths

```python
model = gamfit.fit(df, "Y ~ s(Time, by=Group)")
```

Use this when groups are scientifically distinct and each level should have its
own independently penalised curve.  Internally the formula expands to one
smooth block per level, with rows outside that level zeroed.  Post-hoc pairwise
contrasts are obtained with `difference_smooth()`.

### 2. Ordered-factor / reference difference smooths

The canonical Soskuthy setup is a reference smooth plus deviations from that
reference:

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Time, by=OrderedGroup)")
```

When the `by` column is represented with treatment-coded ordered levels, the
non-reference by-smooths encode deviations from the reference trajectory.  This
is useful for a pre-specified baseline comparison, but it privileges one level.
For three or more symmetric groups prefer the sum-to-zero construction below.

### 3. Binary numeric by-smooths

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Time, by=is_treated)")
```

If `is_treated` is numeric 0/1, the by-smooth is multiplied by that column and
is not active for the 0 rows.  This is a compact two-group contrast when the
vertical offset and shape difference are meant to be interpreted together.

### 4. Sum-to-zero factor smooths (`bs="sz"`)

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Group, Time, bs='sz')")
```

This is the recommended default when no group should be the reference.  Each
level gets a deviation smooth, and equivalent coefficients are constrained to
sum to zero across levels.  The main `s(Time)` captures the population smooth;
the factor smooth captures level-specific deviations around that population
curve.  This mirrors mgcv’s `bs="sz"` design motivation: it is more symmetric
than ordered-factor contrasts.

## Pointwise vs simultaneous bands

`difference_smooth(..., simultaneous=False)` returns pointwise credible bands.
At each grid value the interval uses the joint coefficient covariance:

```text
se(x) = sqrt(diag(X_delta V X_delta^T))
```

Do **not** compare two independently plotted smooth bands for overlap; that is
not a valid difference-band calculation.

`difference_smooth(..., simultaneous=True)` simulates coefficient draws from the
posterior covariance, computes the maximum standardized deviation over the grid
for each draw, and uses the requested quantile as a simultaneous critical value.
Use simultaneous bands for regional claims such as “the groups differ from Time
4 to Time 8,” because that claim implicitly scans many x-values.

## Random effects and population-level contrasts

By default `marginalise_random=True`, so random-effect coefficient columns are
zeroed in the contrast matrix before computing the difference.  This reports the
population-level contrast rather than a contrast conditional on one subject’s
random deviation.  Set `marginalise_random=False` when a conditional contrast is
intended.

## Including group means

`group_means=True` is the substantive default: the contrast is the full fitted
trajectory difference.  Set `group_means=False` to suppress group-offset blocks
where the fitted model exposes them separately from smooth shape.

## References

* Soskuthy (2017), *Generalised additive mixed models for dynamic analysis in linguistics*.
* Soskuthy (2021), tutorial notes on ordered-factor difference smooths.
* Wieling (2018), tutorials on GAMMs for time-course data.
* Simpson, gratia documentation and papers on simultaneous intervals for GAM smooths.
* Wood (2017), *Generalized Additive Models: An Introduction with R*, especially smooth construction and inference chapters.
* mgcv documentation for factor `by` smooths and `bs="sz"` sum-to-zero factor smooth interactions.
