# Difference smooths

`Model.difference_smooth()` returns a pairwise contrast between group-level
smooths, evaluated on a grid of the view variable, with bands computed from
the fitted joint coefficient covariance.

```python
model.difference_smooth(view="Time", group="Group", simultaneous=True)
```

The result has one row per grid point and pair, with columns `diff`, `se`,
`lower`, `upper`, `level`, `simultaneous`, and `critical`.

## Parameterisations

### Unordered by-factor smooths

```python
model = gamfit.fit(df, "Y ~ s(Time, by=Group)")
```

Expands to one smooth block per level of `Group`, each independently
penalised, with rows outside that level zeroed. Pairwise contrasts are
computed by `difference_smooth()`.

### Ordered-factor / reference difference smooths

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Time, by=OrderedGroup)")
```

When `OrderedGroup` is treatment-coded, the non-reference by-smooths
encode deviations from the reference trajectory. Use this when there is
a pre-specified baseline; for three or more symmetric groups, the
sum-to-zero construction below is more appropriate.

### Binary numeric by-smooths

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Time, by=is_treated)")
```

When `is_treated` is numeric 0/1, the by-smooth is multiplied by that
column and contributes nothing on the 0 rows.

### Sum-to-zero factor smooths (`bs="sz"`)

```python
model = gamfit.fit(df, "Y ~ s(Time) + s(Group, Time, bs='sz')")
```

Each level gets a deviation smooth, with coefficients constrained to sum
to zero across levels. `s(Time)` is the population smooth; the factor
smooth captures level-specific deviations.

## Pointwise vs simultaneous bands

`simultaneous=False` (default) returns pointwise bands. At each grid
value the standard error uses the joint coefficient covariance:

```text
se(x) = sqrt(diag(X_delta V X_delta^T))
```

Comparing two independently plotted smooth bands for overlap is not a
valid difference-band calculation.

`simultaneous=True` draws coefficient samples from the posterior
covariance (default `n_sim=10000`), computes the maximum standardised
deviation over the grid for each draw, and uses the requested quantile
as the critical value. Use this for claims about regions, such as "the
groups differ from Time 4 to Time 8".

## Random effects and population-level contrasts

`marginalise_random=True` (default) zeroes random-effect coefficient
columns in the contrast design matrix before evaluating the difference.
Set `marginalise_random=False` for a contrast conditional on a specific
random deviation.

## Group means

`group_means=True` (default) includes the full fitted trajectory
difference. `group_means=False` zeroes the random-effect columns
(categorical offsets are stored as random-effect blocks in this package).

## References

- Soskuthy (2017), *Generalised additive mixed models for dynamic
  analysis in linguistics*.
- Soskuthy (2021), tutorial notes on ordered-factor difference smooths.
- Wieling (2018), tutorials on GAMMs for time-course data.
- Simpson, gratia documentation on simultaneous intervals for GAM
  smooths.
- Wood (2017), *Generalized Additive Models: An Introduction with R*.
- mgcv documentation for factor `by` smooths and `bs="sz"` sum-to-zero
  factor smooth interactions.
