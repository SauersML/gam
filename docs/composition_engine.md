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

- [`proposals/latent_coord.md`](https://github.com/SauersML/gam/blob/main/proposals/latent_coord.md)
- [`proposals/sae_manifold.md`](https://github.com/SauersML/gam/blob/main/proposals/sae_manifold.md)
- [`proposals/per_point_hessian.md`](https://github.com/SauersML/gam/blob/main/proposals/per_point_hessian.md)
- [`proposals/arrow_schur_evidence.md`](https://github.com/SauersML/gam/blob/main/proposals/arrow_schur_evidence.md)
- [`proposals/riemannian_per_point.md`](https://github.com/SauersML/gam/blob/main/proposals/riemannian_per_point.md)

## Analytic Primitives

| Name | Target | Math form | Role | One-line motivation |
| --- | --- | --- | --- | --- |
| `IsometryPenalty` | Decoder pullback metric over latent `t` | `½ w ‖g(t) - g_ref(t)‖²_F` | Gauge fix | Pins the latent chart to a reference geometry so IFT sees an isolated optimum. |
| `ARDPenalty` | Latent axes `t_{.,k}` | `½ Σ_k α_k ‖t_{.,k}‖²` plus REML normalizers | Dimension selection | Prunes axes only after another prior fixes the coordinate gauge. |
| `SparsityPenalty` | Code, assignment, or decoder block | `λ Σ_j sqrt(x_j² + ε²)` | Mechanism sparsity | Encourages small active sets for SAE-style codes and block mechanisms. |
| `OrthogonalityPenalty` | Latent coordinate matrix `T` | `½ w ‖T^T T - I‖²_F` | Gauge fix paired with ARD | Fixes rotation and scale so ARD can make axis-wise pruning identifiable. |
| `TotalVariationPenalty` | `ForwardDiff1D` or `GraphEdges` differences of a latent block | `λ Σ_e sqrt(‖D_e x‖² + ε²)` | Piecewise structure | Promotes piecewise-constant atom maps over sequences or arbitrary adjacency graphs. |

## Minimal Configurations

Standard GAM:

```python
model = gamfit.fit(df, "y ~ s(x) + duchon(u, v, centers=64)")
```

Topology comparison:

```python
gamfit.select_topology(df, "y", score_scale="per_observation")
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

## Sparsity / Piecewise Structure

`SparsityPenalty` handles sparse amplitudes and assignment blocks.
`TotalVariationPenalty` adds smoothed-L1 structure on atom maps: use
`ForwardDiff1D` for ordered sequences and `GraphEdges` for arbitrary adjacency.
Both variants expose analytic HVPs; large graph log-determinants use the shared
Hutchinson / SLQ helper through the analytic-penalty operator path.

## Identifiability Rules

Bare `LatentCoord` is gauge-unfixed: a smooth reparameterization of `t` can be
absorbed by refitting β. Use at least one gauge-breaking prior:

- `aux_prior`, with the log-strength normalizer present, `h(u)` at least C1,
  and conditional precision positive-definite on the anchored subspace.
- `IsometryPenalty`, which pins the decoder pullback metric to a reference.

Topology evidence uses the Tierney-Kadane Laplace normalizer
`F + 0.5 log|H| - 0.5 log|S| - 0.5(dim(H) - rank(S)) log(2*pi)`.
`gamfit.select_topology(..., score_scale="per_observation")` is the default
because it keeps cross-topology REML comparisons on the observed-data scale;
`project_gumbel_anneal_population_sparsity_falsified` records the historical
BIC/REML disagreement when basis dimensions were compared raw.

`ARDPenalty` / `dim_selection=True` is a companion, not a gauge fix. It prunes
axes only after an aux prior or isometry has pinned the coordinate system.
The `auto_exp_21` finding showed the failure mode directly: ARD alone stayed
rotation-invariant and kept all tested auxiliary dimensions. The established
axis-selection composition is `OrthogonalityPenalty` plus `ARDPenalty`, as in
`examples/orthogonality_plus_ard_demo.py`, because orthogonality fixes the
rotation/scale gauge before ARD applies axis-wise evidence pressure.

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
