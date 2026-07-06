# Qwen Geometric-Wall Closure Test

This run uses existing Qwen activation arrays and numpy-only local fits.
Each layer is deterministically subsampled, centered, and peeled along its top PCA sink direction.

Flat uses held-out local PCA with rank `q + q(q+1)/2`.
Curved uses rank-`q` tangent PCA plus all centered quadratic tangent products.
Both lanes therefore have the same output-parameter budget per local neighborhood.

## Settings

- sample rows per layer: 30000
- neighborhoods per layer: 24
- neighborhood size: 240
- tangent rank q: 10
- matched flat rank: 65
- train fraction: 0.75
- ridge scale: 1e-06

## Layer Results

| layer | sink frac | flat floor | curved floor | drop | curvature proxy | curvature/drop r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_8b_L18 | 0.991806 | 0.680735 | 0.898458 | -0.217724 | 0.391442 | -0.529363 |
| qwen3_8b_L30 | 0.837339 | 0.716330 | 0.907891 | -0.191561 | 0.399611 | -0.886797 |
| qwen36_35b_L17 | 0.040939 | 0.625247 | 0.855434 | -0.230187 | 0.428243 | -0.424542 |

## Interpretation

- Floors are held-out residual-energy fractions in post-sink local neighborhoods.
- Drop is `flat_floor_mean - curved_floor_mean`; positive values mean the curved chart lowers the floor.
- The curvature proxy is the RMS quadratic correction energy divided by held-out local energy.
- The reported correlation is across neighborhoods within each layer.
- This run is a null/negative result for the wall-closure hypothesis under this matched local quadratic protocol: curved floors are higher than flat floors in all three layers.
