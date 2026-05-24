# Difference smooths

Difference smooths compare trajectories across groups using the fitted model's joint coefficient covariance. The safe workflow is to fit one of the supported group-smooth parameterisations, then call `Model.difference_smooth(...)` instead of eyeballing overlap between two separately plotted smooth bands.

## Parameterisations

| Idiom | Formula shape | Use when |
| --- | --- | --- |
| Unordered by-factor smooths | `y ~ Group + s(Time, by=Group)` | Groups are scientifically distinct and each level should have its own smoothness. |
| Ordered-factor/reference differences | `y ~ Group + s(Time) + s(Time, by=Group)` | A reference level is meaningful and non-reference rows should be interpreted as deviations from it. |
| Binary numeric by-smooths | `y ~ s(Time) + s(Time, by=treated)` | `treated` is a numeric 0/1 column and the contrast may include both level and shape. |
| Sum-to-zero factor deviations | `y ~ s(Time) + s(Group, Time, bs="sz")` | Three or more groups have no privileged baseline; this is the symmetric default to prefer for new analyses. |

The Rust term builder now recognises `by=` on `s(...)`. Categorical `by` columns are expanded to one row-gated smooth per observed level; numeric/binary `by` columns multiply the smooth by the numeric `by` value. The `bs="sz"` alias is accepted for `s(factor, x, bs="sz")` and exposes the symmetric factor-deviation formula idiom.

## Inference API

```python
contr = model.difference_smooth(
    data=train,
    view="Time",
    group="Group",
    pair=("A", "B"),
    level=0.95,
    simultaneous=True,
)
```

The returned tidy table contains the evaluation coordinate, `level_1`, `level_2`, `diff`, `se`, `lower`, `upper`, `critical`, and `band`. Pointwise bands use

```text
Xd = X_level2 - X_level1
se = sqrt(diag(Xd V Xd'))
```

where `V` is the joint coefficient covariance saved with the model. Simultaneous bands simulate from the coefficient posterior and use the empirical quantile of the maximum absolute standardized deviation over the grid. Use simultaneous bands for regional claims such as “the curves differ between Time = 4 and Time = 8”; pointwise bands do not have whole-curve coverage.

## Options

- `group_means=True` includes parametric group offsets in the contrast. Set it to `False` for a shape-only contrast.
- `marginalise_random=True` is the population-level default. The current public design-matrix export does not yet label random-effect columns, so the flag is accepted for API stability while fixed-effect/smooth covariance contrasts are already computed jointly.

## References

This page follows the distinction drawn in Soskuthy (2017, 2021), Wieling (2018), Wood (2017), and Simpson's gratia/difference-smooth work: parameterisation controls what the model encodes, while post-fit contrast inference must use the joint covariance and simultaneous intervals for regional claims. The mgcv `factor.smooth` documentation motivates the `bs="sz"` construction because no group is privileged as the reference.
