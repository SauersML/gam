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
| `"spherical"` / `"sphere"` | unit sphere | geodesic log-map at the Karcher/Fréchet mean (lives in the ambient tangent plane; norm equals geodesic distance) |
| `"simplex"` / `"clr"` | strictly positive simplex | centered log-ratio |
| `"alr"` | strictly positive simplex | additive log-ratio, with the Aitchison Gram installed automatically unless `fisher_rao_w=` is supplied |
| `"spd"` | symmetric positive-definite matrices | SPD log map; response columns are a flattened square matrix |
| `"grassmann(k=...)"` | k-dimensional subspaces | Grassmann log map at the intrinsic mean; `n` is inferred from the column count unless supplied |
| `"stiefel(k=...)"` | orthonormal k-frames | Stiefel tangent chart; `n` is inferred from the column count unless supplied |
| `"poincare"` / `"poincare(curvature=-0.5)"` | hyperbolic ball with fixed negative curvature | Poincare log map; default curvature is `-1.0` |
| `"constant_curvature"` | learned constant-curvature family | REML/evidence estimates curvature and reports the spherical / flat / hyperbolic verdict |

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
summary = model.summary()           # includes base_point, coordinates, and shared_fit
```

Pass `response_coordinates="alr"` (or `response_geometry="alr"`) to fit
a `D − 1` dimensional ALR chart instead of the default `D`-column CLR
representation. `response_reference=` selects the ALR denominator by
integer component index and defaults to the last response-column position.
Euclidean ALR is not Aitchison-isometric, so an ALR fit runs in the raw ALR
frame and its predictions depend on the (arbitrary) reference component unless
you pass an explicit `fisher_rao_w=` residual metric. For a reference-free
simplex fit, prefer the default CLR representation (or ILR): both are already
Aitchison-isometric (`G = I`), so the fit is invariant to the coordinate choice.

`response_geometry="constant_curvature"` first estimates `kappa_hat` from the
responses, then fits and predicts on `constant_curvature(dim=D,kappa=kappa_hat)`.
`model.summary()["curvature"]` carries `kappa_hat`, the profile CI, the
flatness LR test, and the verdict.

For curved matrix / subspace responses, pass response columns containing
the flattened representation expected by the corresponding geometry
helper. The returned `ResponseGeometryModel` keeps the scalar coordinate
fits, base point, coordinate chart, and geometry metadata in its summary.

## Fisher-Rao weights

`fisher_rao_w=` supplies behavioral precision blocks for response
geometry fits. It accepts:

- a length-`N` vector of scalar row weights;
- one broadcast `(p, p)` positive-semidefinite matrix;
- dense `(N, p, p)` row-specific precision blocks.

The blocks must be finite, symmetric, and have non-negative diagonal
entries. They weight the tangent-coordinate shared REML fit without
changing the response manifold.
