# Response geometry

`gamfit.fit` can treat manifold-valued responses as a first-class target via
`response_geometry=` and `response_columns=`. The implementation maps each
training response to the tangent space at the **intrinsic Fréchet mean** of the
training responses, fits one ordinary Gaussian GAM per tangent coordinate, and
maps predictions back to the response manifold.

Supported geometries:

| `response_geometry` | Response space | Coordinates |
| --- | --- | --- |
| `"spherical"` | unit sphere | ambient tangent coordinates at the Karcher/Fréchet mean |
| `"simplex"` / `"clr"` | strictly positive simplex | centered log-ratio tangent coordinates |
| `"alr"` | strictly positive simplex | additive log-ratio tangent coordinates |

For simplex responses, the base point is the Aitchison Fréchet mean (the closed
componentwise geometric mean), not the arithmetic component mean. For spherical
responses, the base point is computed by Karcher iteration on the sphere, not by
using the unnormalized extrinsic mean as the model origin.

```python
import gamfit

model = gamfit.fit(
    train,
    "composition ~ s(x)",          # the left side is only a label; RHS is reused
    response_geometry="simplex",
    response_columns=["sand", "silt", "clay"],
)

pred = model.predict(test)          # columns: sand, silt, clay; rows sum to one
summary = model.summary()           # includes base_point and per-coordinate GAM summaries
```

Use `response_coordinates="alr"` (or `response_geometry="alr"`) to fit a
`D - 1` dimensional ALR chart instead of the default `D`-column CLR tangent
representation. `response_reference=` chooses the ALR denominator component and
defaults to the last response column.
