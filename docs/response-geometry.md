# Response geometry

`gamfit.fit` accepts manifold-valued responses via `response_geometry=`
and `response_columns=`. Each training response row is mapped to the
tangent space at the intrinsic Fréchet mean of the training responses.
A shared-smoothing Gaussian REML fit is then run on the tangent
coordinates, and predictions are mapped back to the response manifold
with the exponential map.

Supported geometries:

| `response_geometry` | Response space | Tangent coordinates |
| --- | --- | --- |
| `"spherical"` | unit sphere | ambient tangent coordinates at the Karcher/Fréchet mean |
| `"simplex"` / `"clr"` | strictly positive simplex | centered log-ratio |
| `"alr"` | strictly positive simplex | additive log-ratio |

For simplex responses, the base point is the Aitchison Fréchet mean
(the closed componentwise geometric mean). For spherical responses, the
base point is computed by Karcher iteration on the sphere.

```python
import gamfit

model = gamfit.fit(
    train,
    "composition ~ s(x)",          # LHS is a label; the RHS is reused
    response_geometry="simplex",
    response_columns=["sand", "silt", "clay"],
)

pred = model.predict(test)          # columns: sand, silt, clay; rows sum to one
summary = model.summary()           # includes base_point and per-coordinate summaries
```

Pass `response_coordinates="alr"` (or `response_geometry="alr"`) to fit
a `D − 1` dimensional ALR chart instead of the default `D`-column CLR
representation. `response_reference=` selects the ALR denominator
component and defaults to the last response column.
