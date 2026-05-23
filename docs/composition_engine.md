# Composition Engine

`gamfit` is a three-tier penalized-likelihood engine. The familiar GAM API is
one configuration of that engine, not the boundary of the abstraction.

The tiers are:

| Tier | Meaning | Main owners |
| --- | --- | --- |
| β | Decoder / smooth coefficients solved by the penalized Newton inner loop. | `src/solver/pirls.rs`, `src/solver/latent_inner.rs` |
| ψ / ext-coords | Design-moving coordinates transported through β by IFT. `ψ` is reserved for kernel-shape state such as `SpatialLogKappaCoords`; per-row latents are `t`. | `src/terms/smooth.rs::SpatialLogKappaCoords`, `src/terms/latent_coord.rs::LatentCoordValues`, `src/solver/reml/runtime.rs::evaluate_unified_with_psi_ext` |
| ρ | Penalty strengths and structural hyperparameters selected by REML / LAML. | `src/solver/reml/`, `src/terms/analytic_penalties.rs` |

The engine accepts a design block, a penalty block, derivatives of both with
respect to design-moving ext-coordinates and penalty strengths, and a weighted
likelihood. It returns β and ρ, with the implicit-function-theorem correction
through the inner optimum.

## Methodspace

| Configuration | β | ext-coords | ρ | Main priors |
| --- | --- | --- | --- | --- |
| GAM | Smooth coefficients | None; covariates are observed | Smoothness | `S(ρ)` smoothness |
| Anisotropic Matern / Duchon | Smooth coefficients | Kernel-shape `ψ` (`SpatialLogKappaCoords`) | Smoothness + per-axis scale | Smoothness, scale priors |
| LatentCoord / GP-LVM | Decoder coefficients | Per-row latent `t_i` (`LatentCoordValues`) | Smoothness + optional ARD | Aux prior or isometry, topology via basis |
| Linear SAE | Dictionary / code blocks | None in the current public wrapper | Sparsity strength + atom count | `SparsityPenalty`, `compare_models` |
| SAE-manifold | Per-atom decoder coefficients | Soft assignment `a_i` plus on-atom `t_ik` | Sparsity, smoothness, per-atom ARD | Sparsity, isometry or aux prior |
| Fisher-Rao manifold | Decoder coefficients | Optional `t_i` | Smoothness + geometry priors | Working weight `W` as output Fisher information |

Related design notes:

- [`proposals/composition_engine.md`](../proposals/composition_engine.md)
- [`proposals/latent_coord.md`](../proposals/latent_coord.md)
- [`proposals/sae_manifold.md`](../proposals/sae_manifold.md)

## Minimal Configurations

Standard GAM:

```python
model = gamfit.fit(df, "y ~ s(x) + duchon(u, v, centers=64)")
```

Topology comparison:

```python
from gamfit import topology

candidates = [
    ("circle", topology.Circle(name="theta")),
    ("torus", topology.Torus(centers=torus_grid, name="theta_phi")),
    ("patch", topology.EuclideanPatch(d=2, centers=patch_grid, name="uv")),
]
fits = [fit_topology(name, basis) for name, basis in candidates]
gamfit.compare_models(fits, names=[name for name, _ in candidates])
```

LatentCoord / GP-LVM shape:

```python
t = gamfit.LatentCoord(
    n=N,
    d=4,
    init="pca",
    aux_prior={"u": environment, "family": "ridge", "strength": "auto"},
    dim_selection=True,
)
iso = gamfit.IsometryPenalty(target="t", strength="auto")
```

Low-level Gaussian latent fit:

```python
out = gamfit.gaussian_reml_fit_latent(
    t0.reshape(-1),
    Y,
    n_obs=N,
    latent_dim=4,
    centers=centers,
    penalty=omega,
    aux_u=environment,
)
```

SAE-manifold configuration:

```python
latents = {
    "a": gamfit.LatentCoord(n=N, d=K, init="kmeans", name="a"),
    "t": gamfit.LatentCoord(n=N, d=K * d_atom, init="pca", dim_selection=True, name="t"),
}
penalties = [
    gamfit.SparsityPenalty(target="a", strength="auto"),
    gamfit.IsometryPenalty(target="t", strength="auto"),
]
```

`gamfit.sae_manifold_fit` exists as a Python assembly wrapper, and
`src/terms/sae_manifold.rs` now names the first-class Rust term shape for the
joint Arrow-Schur row-block assembly.

## Identifiability Rules

Bare `LatentCoord` is gauge-unfixed: a smooth reparameterization of `t` can be
absorbed by refitting β. Use at least one gauge-breaking prior:

- `aux_prior`, with the log-strength normalizer present, `h(u)` at least C1,
  and conditional precision positive-definite on the anchored subspace.
- `IsometryPenalty`, which pins the decoder pullback metric to a reference.

`ARDPenalty` / `dim_selection=True` is a companion, not a gauge fix. It prunes
axes only after an aux prior or isometry has pinned the coordinate system.

The arrow-Schur latent Hessian is cheap at the cost level, not because every
REML derivative is independent. The outer Occam gradient uses one shared
Schur-inverse setup per outer iteration plus row-local rank updates.

## LLM Controllability

LLM experiments have two advantages over generic unsupervised learning:

- The auxiliary variable is often free. Prompt sweeps, interventions, or
  synthetic labels can supply `u`, turning nonlinear-ICA-style impossibility
  into a controlled identifiability assumption.
- Fisher-Rao geometry is one configuration flag away once the working model
  exposes behavioral Fisher information as `W`. Then one unit in `t` means one
  unit of behavioral change, not one unit of Euclidean activation distance.

That is the maintainers' mental model: pick a decoder, choose which quantities
live in β / ext-coords / ρ, attach explicit priors, and let REML compare the
resulting sentences in the same engine.
