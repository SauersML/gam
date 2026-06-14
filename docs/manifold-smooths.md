# Manifold and geometric smooths

`gamfit` supports smooths whose predictor space is a manifold other than
flat Euclidean space: a circle, a cylinder, a torus, a sphere, or a
tensor product with periodic margins. Boundary-conditioned 1-D smooths
(clamped and anchored) are also available. Reference for formula
options is in the [Formula DSL reference](formulas.md).

## Demo: six geometric examples

The image below is the output of
[`scripts/geometric_shapes_demo.py`](https://github.com/SauersML/gam/blob/main/scripts/geometric_shapes_demo.py).
For each shape, noisy 3-D points `(x, y, z)` are sampled along the
manifold and one geometric smooth is fit per output coordinate:

```python
gamfit.fit(df, "x ~ <geometric-smooth>(latent_params)")
gamfit.fit(df, "y ~ <geometric-smooth>(latent_params)")
gamfit.fit(df, "z ~ <geometric-smooth>(latent_params)")
```

Predicting all three coordinates on a dense grid in the latent space
gives the reconstructed surface. Left panel of each pair: noisy
observations. Right panel: the fitted surface.

![rotating-shape demo of six geometric smooths recovered from noisy 3-D
point clouds](images/geometric_shapes_demo.gif){ width="100%" }

The script writes a full-length MP4 alongside the GIF and PNG. See
[reproduction recipe](#reproducing-the-demo) below.

## Shapes and formulas

| Shape | Latent params | Formula |
| --- | --- | --- |
| Trefoil knot (closed curve in ℝ³) | `t` ∈ [0, 2π) | `x ~ s(t, periodic=true, period=2*pi, k=24)` |
| Wiggly loop (closed curve) | `t` ∈ [0, 2π) | `x ~ s(t, periodic=true, period=2*pi, k=18)` |
| Wobbly cylinder (one periodic axis, one open axis) | `θ` ∈ [0, 2π), `h` ∈ [0, 1] | `x ~ te(theta, h, periodic=[0], period=[2*pi, None], k=[26,12])` |
| Lumpy sphere (intrinsic S²) | `lat`, `lon` (radians) | `x ~ sphere(lat, lon, radians=true, k=100)` |
| Bumpy torus (two periodic axes) | `u`, `v` ∈ [0, 2π) | `x ~ te(u, v, periodic=[0,1], period=[2*pi, 2*pi], k=[20,16])` |
| Möbius embedding (4π double-cover) | `u` ∈ [0, 4π), `v` ∈ [−0.8, 0.8] | `x ~ te(u, v, periodic=[0], period=[4*pi, None], k=[32,10])` |

The three coordinate fits per shape are independent: there is no shared
parameter and no joint loss. Surface continuity at the seams is a
consequence of the basis and penalty on the latent manifold.

### Notes

- Wiggly loop. The demo samples and fits the observed latent parameter
  `t`; there is no latent-coordinate inference step. The cyclic
  boundary removes the seam.
- Sphere. The intrinsic `sphere(lat, lon)` smooth uses Wahba's
  reproducing kernel on S² (rotation-invariant, no pole artefacts). A
  spherical-harmonic alternative is available as
  `method=harmonic, max_degree=L`. Both are documented in the
  [formula reference](formulas.md#intrinsic-s2-sphere-smooth).
- Möbius embedding (4π double-cover). The demo uses
  `F(u, v) = ((1 + ½v cos(u/2)) cos u, (1 + ½v cos(u/2)) sin u,
  ½v sin(u/2))`, which satisfies `F(u+2π, v) = F(u, −v)` and
  `F(u+4π, v) = F(u, v)`. The smoother is given the latter, ordinary
  periodicity (period `4π` in `u`), so the predictor manifold is the
  orientable cylinder `S¹ × [−0.8, 0.8]`. The fitted surface in ℝ³ looks
  Möbius because the embedding is Möbius; the basis and penalty do not
  encode the twisted identification `(u, v) ∼ (u+2π, −v)`. A true
  Möbius basis is not exposed by the formula DSL.

## Why use a geometric smooth

`te(theta, h)` on a cylinder without `periodic=[0]` produces a visible
seam at `θ = 0`. Same for a torus (two seams) or sphere (a longitude
seam plus pole crowding). The geometric smooths build the wrap topology
into both basis and penalty:

- Predictions at `θ = 0` and `θ = 2π` agree.
- The penalty integrates the squared derivative around the loop, so
  the wiggliness budget is allocated correctly.
- For the sphere, the kernel is isotropic and applies uniformly at
  the poles and the equator.

## Reproducing the demo

```bash
cargo build --release
uv run --with numpy --with pyvista --with matplotlib \
       --with imageio --with imageio-ffmpeg --with pillow \
       python3 scripts/geometric_shapes_demo.py
```

The first run generates noisy CSVs under
`scripts/geometric_shapes_demo_data/`, fits 18 small models via the
`gam` CLI (about 15 s on a laptop), and writes PNG, MP4, and GIF
outputs. Re-runs reuse the cache; `--regen` starts fresh, and
`--still / --mp4 / --gif` selects a single output.

A smaller Python-only entry point (a tilted 3-D circle with a localized
radial spike, fit via `gamfit.fit` directly) is in
[`scripts/circle_3d_cyclic_demo.py`](https://github.com/SauersML/gam/blob/main/scripts/circle_3d_cyclic_demo.py).

## Torch-side: `gamfit.torch.ManifoldSAE`

`gamfit.torch.ManifoldSAE(cfg)` is a trainable `nn.Module` mirror of the
closed-form SAE manifold primitive. Atom *i* is a 1-D parametric curve in
ambient `R^D`, parameterized by a manifold point and a decoder block; a shared
encoder produces per-atom on-manifold coordinates and amplitudes from each
input batch. The closed-form primitive itself — the objective, the typed
topologies, posterior shape bands, and coordinate ranges — is documented in
the [Manifold SAE dictionary guide](manifold-sae.md).

```python
import torch
from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

cfg = ManifoldSAEConfig(
    input_dim=D, n_atoms=F, intrinsic_rank=1,
    atom_manifold="circle",
    atom_basis="fourier", basis_order=2, n_basis_per_atom=K,
    sparsity={"kind": "ibp_gumbel", "init_alpha": 0.1,
              "tau_schedule": "linear:4.0->1.0"},
    decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
    reml={"enabled": True, "select": ["lambda", "tau"]},
)
sae = ManifoldSAE(cfg)
out = sae(x)               # ManifoldSAEOutput(z, x_hat, positions, ...)
sae.fit(x)                 # delegates to gamfit.sae_manifold_fit
sae.lock_snapshot()        # freeze REML-selected hypers
curves = sae.extract_feature_curves(grid_size=128)   # {atom_idx -> Tensor(grid, D)}
```

**Parity contract.** `ManifoldSAE.fit(...)` and `gamfit.sae_manifold_fit(...)`
share a Rust kernel. Given equivalent configurations they return identical
numerics on synthetic data for supported closed-form configurations (see
`tests/torch/test_manifold_sae_parity.py`). Closed-form `.fit()` rejects
nonzero decoder orthogonality / monotonicity penalties and cylinder atoms
because those objectives are not represented by the Rust SAE primitive.

Supported atom manifolds: `circle`, `cylinder`, `sphere`, `product`. Supported
bases-on-manifold: `duchon`, `bspline`, `fourier`. Sparsity layers:
`ibp_gumbel`, `softmax_topk`, `jumprelu`. In the closed-form `.fit()` path,
`circle` maps to a periodic atom, `sphere` maps to a sphere atom, and `product`
maps to a Duchon Euclidean patch; `bspline` remains a torch-training basis.

## Related reading

- [Formula DSL — periodic / cyclic smooths](formulas.md#periodic-cyclic-smooths)
- [Formula DSL — boundary-conditioned 1-D smooths](formulas.md#boundary-conditioned-1d-smooths)
- [Formula DSL — intrinsic S² (sphere) smooth](formulas.md#intrinsic-s2-sphere-smooth)
- [Response geometry](response-geometry.md) — manifold-valued
  responses (the predictor side is ordinary).
