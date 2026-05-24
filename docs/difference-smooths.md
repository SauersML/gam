# Difference smooths

Difference smooths compare trajectories between groups without asking users to do coefficient-covariance algebra by hand.

## Parameterisations

* **Unordered by-factor smooths**: `y ~ s(x, by=group)` expands to one centred smooth per categorical level. The implementation also adds an unpenalized treatment-coded factor main effect, because centred level smooths cannot absorb vertical offsets.
* **Ordered/reference difference smooths**: fit a reference smooth plus a categorical-by smooth (for example `y ~ s(x) + s(x, by=group)`) when a scientific baseline is meaningful. Post-fit contrasts use the same covariance-aware API.
* **Binary numeric by-smooths**: `y ~ s(x) + s(x, by=treated)` multiplies the smooth basis by a numeric 0/1 column. This term is not level-centred, matching the mgcv idiom for a combined level-and-shape treatment contrast.
* **Sum-to-zero factor smooths**: `y ~ s(x) + s(group, x, bs=sz)` estimates coefficient-wise group deviations that sum to zero across levels. This is the recommended symmetric default when no level should be privileged as the reference.

## Inference API

`Model.difference_smooth(data, group="group", view="x")` builds two design matrices on a grid, forms `X_B - X_A`, and computes the contrast standard error from the joint coefficient covariance. `simultaneous=True` uses posterior simulation of the maximum standardized deviation over the grid to return a simultaneous band for regional claims.

By default the returned contrast includes the parametric group offset (`group_means=True`), because that is usually the substantive full-trajectory difference. The current public method raises a clear error for `group_means=False` until term-role metadata is exposed in the Python FFI.

## Choosing among them

Use unordered by-factor smooths when groups are scientifically independent and each group should get its own smoothing parameter. Use a reference/difference setup when a baseline is central to the question. Use binary by-smooths for two-group treatment indicators when level and shape differences need not be separated. Use `bs=sz` for three or more peer groups because it estimates deviations around a population mean without privileging any level.

## References

This page follows the difference-smooth guidance in Soskuthy (2017, 2021), Wieling (2018), Wood (2017, *Generalized Additive Models*), and Simpson's gratia work on covariance-aware difference smooths and simultaneous intervals. The sum-to-zero recommendation mirrors mgcv's `bs="sz"` documentation.
