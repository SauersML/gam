# Manifold SAE: a sparse dictionary of typed shapes

`gamfit.sae_manifold_fit(...)` fits a **sparse manifold dictionary**. The
data matrix `Z` (shape `(N, p)`, one ambient vector per token) is
reconstructed as a sparse mixture of `K` *atoms*. Each atom is a small,
low-dimensional **typed shape** — a line, circle, sphere, torus, cylinder,
hyperbolic patch, or flat Euclidean patch — carrying its own ambient
embedding (a decoder block) and a
per-token coordinate on that shape. A token's reconstruction is the sum of
the atoms it activates, evaluated at their per-token coordinates.

Each atom answers **where it lives** (a typed manifold embedded in ambient
space), **what shape it is** (the topology and its fitted curve), and **how
confident** that shape is (a posterior band).

```python
import numpy as np
import gamfit

Z = ...  # (N, p) activations / embeddings to decompose

fit = gamfit.sae_manifold_fit(
    X=Z,
    K=16,                       # dictionary size
    d_atom=1,                   # intrinsic dim per atom (default 2; int, or per-atom list)
    atom_topology="circle",     # default "circle"; see the topology table below
    assignment="ibp_map",       # IBP-MAP sparsity (default)
)

print(fit)              # ManifoldSAE(K=16, d_atom=1, atom_topology='circle', ...)
print(fit.summary())    # K, active dims, avg active atoms, reconstruction R², ...
recon = fit.predict(Z)  # (N, p) reconstruction; == fit.fitted on training Z
```

`sae_manifold_fit` returns a [`ManifoldSAE`](#the-manifoldsae-result). Both
`gamfit.sae_manifold_fit` and `gamfit.ManifoldSAE` are top-level exports.

### `sae_manifold_fit` parameters

The full signature, with defaults (keyword-only arguments follow the `*`):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `X` | `None` | `(N, p)` data to decompose |
| `K` | `None` | dictionary size (number of atoms) |
| `d_atom` | `2` | intrinsic dim per atom (int, or per-atom list) |
| `atom_topology` | `"circle"` | global topology string (see table) |
| `assignment` | `"ibp_map"` | gate kind: `ibp_map` / `softmax` / `jumprelu` |
| `schedule` | `None` | `GumbelTemperatureSchedule` for annealed gates |
| `isometry_weight` | `0.0` | unit-speed gauge penalty (off by default) |
| `ard_per_atom` | `True` | ARD pruning of unused coordinate axes |
| `decoder_feature_sparsity_groups` | `None` | output-feature partition for decoder group-lasso |
| `n_iter` | `50` | joint-solve iterations |
| `sparsity_weight` | `1.0` | strength of the gate-sparsity penalty |
| `gate_sparsity` | `"scad"` | `scad` / `mcp` / `l1` |
| `scad_mcp_gamma` | `None` | SCAD/MCP concavity (defaults SCAD 3.7, MCP 2.5) |
| `smoothness_weight` | `1.0` | roughness penalty strength |
| `alpha` | `1.0` | ARD/precision seed (`float` or a string policy) |
| `learning_rate` | `None` | optional step size override |
| `random_state` | `0` | RNG seed |
| `block_orthogonality_weight` | `0.0` | orthogonalize latent axes (needs `d_atom >= 2`) |
| `nuclear_norm_weight` | `1.0` | embedding-rank selection penalty |
| `nuclear_norm_max_rank` | `None` | optional cap on the embedding rank |
| `decoder_incoherence_weight` | `1.0` | cross-atom incoherence (separability lever) |
| `top_k` | `None` | cap on per-token active atoms, applied as a **post-fit** hard projection (#1409). Under `softmax` assignment the inner solve still optimises all `K` atoms; `top_k` truncates each token's assignments afterward. It is a sparsity cap on the reported support, **not** a reduction of the fit's per-token compute. (Gate-style modes — `jumprelu` / IBP-MAP — engage the compact active-set layout during the fit.) |
| `t_init` | `None` | coordinate warm start `(K, N, D_max)` |
| `a_init` | `None` | assignment-logit warm start `(N, K)` |
| `tau` | `None` | Gumbel-softmax temperature |
| `jumprelu_threshold` | `0.0` | hard-threshold cutoff for `jumprelu` |
| `atom_basis` | `None` | per-atom topology list (overrides `atom_topology`) |
| `fisher_factors` | `None` | per-token Fisher reweighting factors |
| `weights` | `None` | per-row observation weights |

## Topology types

Each atom carries a topology, set globally by `atom_topology=` or per atom by
passing a list to `atom_basis=`:

The string is resolved by `SaeAtomBasisKind` in the Rust core. The full set
of typed shapes:

| `atom_topology` / `atom_basis` | Intrinsic dim `d_atom` | Underlying basis | Shape |
| --- | --- | --- | --- |
| `linear` | 1 | affine rank-1 line | true linear atom `gamma(t)=b0+t*b1` |
| `circle` (aliases `periodic`, `periodic_spline`) | 1 (each axis if `d>1`) | periodic Fourier `cos/sin(2π·h·t)` | closed loop, seamless at the wrap |
| `euclidean` (alias `euclidean_patch`) | any | monomial / polynomial patch | open Euclidean patch (a "line" at `d_atom=1`) |
| `duchon` | any | Duchon thin-plate RKHS | open Euclidean patch with the thin-plate roughness Gram |
| `sphere` | 2 | intrinsic S² (lat, lon) chart | sphere, no pole artefacts |
| `torus` | 2 (each axis) | doubly-periodic Fourier | torus, seamless on both axes |
| `cylinder` | 2 | periodic circle axis ⊗ flat line axis | cylinder `S¹ × ℝ`, periodic in axis 0, open in axis 1 |
| `poincare` (aliases `hyperbolic`, `poincare_patch`) | any | monomial patch, hyperbolic roughness metric | Poincaré-ball tangent patch at curvature `c = −1` (wiggle measured in hyperbolic arc length) |
| any other string | any | caller-supplied | `Precomputed`: a precomputed basis you attach yourself |

`duchon` and `euclidean` share the same flat-`ℝᵈ` latent manifold and monomial
decoder; they differ only in the roughness penalty (`duchon` uses the
thin-plate RKHS Gram, `euclidean` the polynomial-patch Gram). `poincare`
likewise reuses the Euclidean tangent chart and monomial decoder but penalizes
roughness in hyperbolic arc length via the Poincaré conformal factor, so it
differs from the flat patch only where feature density grows toward the ball
boundary (tree-/hierarchy-like structure). String matching is
case-insensitive and treats `-` and `_` interchangeably.

A `d_atom=1` linear atom is the true rank-1 **line** primitive. A
`d_atom=1` Euclidean atom is a stronger polynomial patch, not the pure-linear
baseline used for #1026 reconstruction parity. A `d_atom=1` circle is the
**loop**. Mixing topologies across atoms is supported — pass an
`atom_basis=[...]` list and read the resolved per-atom labels off
`fit.atom_topologies` (the scalar `fit.atom_topology` collapses to `"mixed"`
when atoms disagree).

## The objective

The fit minimizes reconstruction error plus a stack of penalties, with
smoothing weights selected by REML. Each piece plays a distinct role
(default state in parentheses):

- **Reconstruction.** Squared error between `Z` and the sparse sum of
  per-atom decoded points. Reported as `fit.reconstruction_r2`.

- **Gate sparsity (`assignment=`, canonical).** The per-token, per-atom gate
  is selected by the assignment prior. The three supported kinds are
  `"ibp_map"` (default), `"softmax"`, and `"jumprelu"`. `"ibp_map"`
  (Indian Buffet Process MAP) adapts the *number* of active atoms per token
  via a sigmoid gate scaled by a stick-breaking geometric weight. The gate
  is `σ(ℓ/τ)·π_k(α)`, which is strictly positive at every finite logit, so it
  produces a **sharply decaying, sparsity-inducing** gate rather than a soft
  simplex — but not exact zeros on its own; **hard zeros require the separate
  `top_k=` truncation** (#1421). `"softmax"` is a
  dense, simplex-normalized gate; `"jumprelu"` is a hard threshold (its
  cutoff is `jumprelu_threshold=`, default `0.0`). `top_k=` optionally caps
  the per-token active set, and `tau=` sets the Gumbel-softmax temperature.

  **Gumbel temperature schedules.** For the annealed gates, pass `schedule=`
  (a `GumbelTemperatureSchedule` or a mapping). Three constructors are
  top-level exports:

  ```python
  from gamfit import (gumbel_geometric_schedule, gumbel_linear_schedule,
                      gumbel_reciprocal_iter_schedule)
  # geometric decay τ_start → τ_min at the given multiplicative rate
  sched = gumbel_geometric_schedule(tau_start=4.0, tau_min=1.0, rate=0.9)
  # linear ramp over `steps` iterations
  sched = gumbel_linear_schedule(tau_start=4.0, tau_min=1.0, steps=50)
  # reciprocal-in-iteration decay τ(i) = τ_start / (1 + i)
  sched = gumbel_reciprocal_iter_schedule(tau_start=4.0, tau_min=1.0)
  fit = gamfit.sae_manifold_fit(X=Z, K=16, assignment="softmax", schedule=sched)
  ```

- **Cross-atom decoder incoherence** (`decoder_incoherence_weight=1.0`, **on
  by default**). The separability lever. For `K >= 2` it
  penalizes the squared output-space cross-Gram `||B_j B_k^T||_F^2` between
  the `(M_k, p)` decoder blocks of *co-activating* atom pairs, weighted by
  their empirical co-activation `mean_n gate_j·gate_k`. This drives co-firing
  atoms toward perpendicular ambient subspaces, conditioning the joint solve
  and making the decomposition identifiable. Set the weight to `0.0` to
  disable.

- **Nuclear-norm embedding-rank selection** (`nuclear_norm_weight=1.0`, **on
  by default**; `nuclear_norm_max_rank=` optional cap). Adds a
  smoothed sum-of-singular-values penalty on each atom's `(M_k, p)` decoder matrix,
  shrinking its singular spectrum so the **embedding dimension** — how many
  ambient output directions the atom spends — is *selected* rather than
  fixed. Routed to the decoder (`"beta"`) block, complementary to the
  intrinsic-dim and topology selection below.

- **ARD intrinsic-dim pruning** (`ard_per_atom=True`, **on by default**). An
  automatic-relevance-determination penalty on each atom's latent coordinate
  block prunes unused coordinate axes, so the *intrinsic* dimension actually
  used is driven below `d_atom` when the data does not fill it. The surviving
  count per atom is `fit.atoms[k].active_dim` (also `fit.summary()
  ["active_dims"]`).

- **Isometry gauge** (`isometry_weight=0.0`, **off by default**).
  `IsometryPenalty` drives the pulled-back metric
  `g = J^T J` toward a unit-average-speed chart, making `t` easier to read as
  near arc length when the penalty is enabled. It is no longer required for
  topology evidence: the Rust core reparameterizes decoder roughness by the
  pulled-back metric, so `fit.reml_score` is gauge-invariant with
  `isometry_weight=0.0`.

- **Smoothness** (`smoothness_weight=1.0`). Roughness penalty on each atom's
  decoded curve, a fixed finite-/cyclic-difference Gram in the latent
  coordinate.

- **Gate sparsity penalty** (`gate_sparsity="scad"` default; `"l1"` /
  `"mcp"` alternatives). `scad` and `mcp` emit a row-block `ScadMcpPenalty` on
  the latent block, using `sparsity_weight=` as its strength and
  `scad_mcp_gamma=` as the concavity/taper (defaults SCAD `3.7`, MCP `2.5`).
  `l1` keeps the historical assignment-prior sparsity path.

Two more knobs: `decoder_feature_sparsity_groups=` group-lassoes the decoder
over a partition of the `p` output features (encouraging each basis function
to load on a single feature cluster), and
`block_orthogonality_weight=` orthogonalizes the latent coordinate axes
(requires `d_atom >= 2`).

## Uncertainty and range: where each manifold lives, what shape, how confident

A fresh fit reports, per atom, a **posterior shape band** and a **typical
coordinate range**. Together they answer: *where does this manifold live in
ambient space, what shape does it trace, and how confident is that shape?*

### Posterior shape uncertainty

The decoded ambient point at coordinate `t` is `m_k(t) = Φ_k(t)·B_k` — linear
in the decoder coefficients `β_k`. The fit carries the φ-scaled Laplace
posterior covariance of those coefficients,
`Cov(β_k) = φ · S_β^{-1}[block]`, and pushes it forward in closed form (no
sampling) to a per-channel ambient band:

```
Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]
```

`fit.shape_uncertainty(k)` returns the mean curve and its `±sd` envelope for atom
`k`, evaluated along the atom's own coordinates (so the band reports
uncertainty exactly where the data lives):

```python
band = fit.shape_uncertainty(0, n_sd=1.96)   # pointwise 95% band
band["coords"]   # (G, d_k)  the coordinates the band is evaluated at
band["mean"]     # (G, p)    fitted ambient point m_k(t)
band["sd"]       # (G, p)    posterior sd per channel
band["lower"]    # (G, p)    mean - n_sd * sd
band["upper"]    # (G, p)    mean + n_sd * sd
```

The raw covariance is also on each atom for custom grids:

```python
atom = fit.atoms[0]
atom.decoder_covariance   # (M_k*p, M_k*p), row-major (basis, channel) flat layout
atom.shape_band_coords    # (G, d_k)
atom.shape_band_mean      # (G, p)
atom.shape_band_sd        # (G, p)
```

This is an epistemic posterior on the manifold, not the per-observation data
scatter: it shrinks as `~1/sqrt(N)` with more data, scales with the
reconstruction dispersion `fit.dispersion`, and widens for a poorly-identified
atom (a near-singular Schur block fans the band out). With many tokens the
manifold is pinned far more tightly than any single noisy observation.

!!! note
    The uncertainty arrays are populated only on a **freshly-fit** model.
    A model round-tripped through `save` / `load` (or `to_dict` /
    `from_dict`) drops them, and `shape_uncertainty` then raises `ValueError`.

### Typical coordinate range

Each atom's per-token coordinate lives in `fit.coords[k]` (shape `(N, d_k)`;
equivalently `fit.atoms[k].coords`). The **observed extent** — where on the
shape the tokens actually land — is read directly off it. Combined with the
shape band, you get the curve *across the range the atom is used over*:

```python
coords_k = fit.coords[0]                       # (N, d_k) per-token coordinate
lo, hi   = coords_k.min(0), coords_k.max(0)    # full observed extent per axis
p5, p95  = np.percentile(coords_k, [5, 95], 0) # robust central range

band = fit.shape_uncertainty(0)
# The band coordinates are an evenly-strided subset of the per-token
# coordinates, so band["mean"] already traces the shape across exactly the
# range the atom occupies. To inspect a sub-range, mask on band["coords"]:
in_range = (band["coords"][:, 0] >= p5[0]) & (band["coords"][:, 0] <= p95[0])
typical_curve = band["mean"][in_range]         # shape over the central 90% of use
typical_sd    = band["sd"][in_range]
```

For out-of-sample tokens, `fit.per_atom_latent_for(X)` returns the per-atom
coordinates under the frozen decoder, so the same extent / percentile read
applies to new data.

### Per-atom curvature report

Fresh SAE fits also carry a first-class per-atom curvature report:

```python
curv = fit.curvature()
curv[0]["kappa_hat"]          # fitted empirical curvature estimate for atom 0
```

`kappa_hat` is the empirical curvature bound used by the curved-dictionary
certificate. SAE dictionaries do not currently expose a profile-likelihood CI
or flatness LR test, so the per-atom rows are intentionally just
`{"atom", "kappa_hat"}`.

## The `ManifoldSAE` result

Key attributes (see the [API reference](api-reference.md#manifold-sae) for
the full surface):

| Attribute | Meaning |
| --- | --- |
| `atoms` | list of `SaeManifoldAtomFit` (decoder, coords, `active_dim`, uncertainty) |
| `atom_topology` / `atom_topologies` | scalar (`"mixed"` if heterogeneous) / per-atom topology |
| `assignment` / `assignment_label` | canonical gate kind / the string you passed |
| `coords` | `[ (N, d_k) ]` per-atom per-token coordinates |
| `decoder_blocks` | `[ (M_k, p) ]` per-atom decoder matrices |
| `fitted` / `assignments` | `(N, p)` training reconstruction / `(N, K)` gates |
| `reconstruction_r2` / `reml_score` / `dispersion` | fit-quality / evidence / noise scale |

Methods: `predict` / `reconstruct(X)`, `encode(X)` (out-of-sample gates),
`project(X, k)`, `per_atom_latent_for(X)` and `featurize(X)` (coordinates),
`per_atom_active_set(X)`, `shape_uncertainty(k)`, `coordinate_range(k)`,
`typical_shape(k)`, `curvature()` / `atom_curvature(k)`, `summary()`,
`get_decoder()` / `get_anchors()`, and `to_dict` / `from_dict` / `save` /
`load`.

### Out-of-sample and encoder distillation

`X=` is the data to reconstruct — it is **not** a warm start.
To seed the joint solve from an amortized encoder's per-token prediction,
pass `a_init` (assignment logits `(N, K)`) and/or `t_init` (coordinates
`(K, N, D_max)`) and a small `n_iter` for a bounded refinement; read the
converged supervision targets back with `fit.converged_latents(X)`.

The out-of-sample surface:

```python
fit.predict(X)               # (N, p) reconstruction under the frozen decoder
fit.reconstruct(X, t_init=None, a_init=None)   # same, with optional warm starts
gates = fit.encode(X)        # (N, K) out-of-sample gates
gates, stats = fit.encode(X, return_stats=True)
coords = fit.featurize(X)    # [ (N, d_k) ] per-atom coords (alias of per_atom_latent_for)
coords = fit.per_atom_latent_for(X)
active = fit.per_atom_active_set(X, threshold=None)  # (N, K) bool active mask
proj_k = fit.project(X, atom_k=0)        # (N, p) single-atom reconstruction
latents = fit.converged_latents(X)       # exact solver targets for distillation
encoder = fit.distill_encoder(X)         # amortized torch encoder from exact solves
```

This enables the "encoder predicts → solver refines → distill the gap" loop:
`distill_encoder` trains a post-hoc torch MLP from the exact out-of-sample
latent solves so future inference is a single forward pass.

## Steering and causal intervention

`fit.steer(atom_k, t_from, t_to)` builds a **causal intervention plan** that
moves one atom's latent coordinate from `t_from` to `t_to` and reports the
resulting ambient delta with output dosimetry:

```python
plan = fit.steer(atom_k=3, t_from=0.1, t_to=0.4)
plan["delta"]             # (p,) activation-space move a·(g_k(t_to) − g_k(t_from))
plan["predicted_nats"]    # path-integrated output-Fisher KL dose (nats), or None
plan["validity_radius"]   # latent step length the linearization is trusted to, or None
plan["off_manifold_norm"] # component of the move off the atom's local tangents (≈0 on-manifold)
plan["metric_provenance"] # "OutputFisher" if a Fisher metric was installed, else "Euclidean"
```

It answers "if I push token feature *k* along its manifold from here to there,
what happens downstream, and how far can I trust that?" — the move stays on the
atom's fitted shape, and `predicted_nats` quantifies the intervention strength
in output-distribution terms. The KL dose and validity radius require an
output-Fisher metric (`fisher_factors=` supplied to `sae_manifold_fit`);
without it the geometry (`delta`, `off_manifold_norm`) is still returned but
the dose degrades to `None`, not zero.

## Certified structure (anytime-valid)

The dictionary's structural claims — *does this atom exist*, *which atoms bind
together*, *what geometry kind is each* — are adjudicated by an anytime-valid
e-BH (Benjamini-Hochberg on e-values) certificate, so the false-discovery
control holds under optional stopping:

```python
cert = fit.structure_certificate(alpha=0.05)   # confirmed/contested claims + e-values
contested = fit.contested_claims(alpha=0.05)    # only the unconfirmed claims
probes = fit.contested_probe_report(alpha=0.05) # what evidence would settle them
```

The same machinery is exposed at top level for working with raw claim ledgers:
`gamfit.e_bh_dictionary_certificate(...)` and
`gamfit.plan_probe_for_contested_claim(...)` (probe design for a contested
claim). `contested_probe_report` pairs a contested claim with the probe that
would most cheaply confirm or refute it — the design loop for active
interpretability.

## Trust, diagnostics, and curvature

Per-atom trust scores fold tangent conditioning, activation frequency, and
support into a single `[0, 1]` score:

```python
fit.atom_trust(0)            # scalar trust in [0, 1] for atom 0
fit.atom_diagnostics(0)      # full per-atom diagnostic dict (tangent condition, ...)
fit.coordinate_range(0)      # observed min/max/p05/p50/p95 of the atom's coords
fit.typical_shape(0, quantile_range=(5.0, 95.0), n_sd=1.0)  # central-range curve+band
```

The top-level helpers `gamfit.sae_trust_diagnostics(payload)` and
`gamfit.atom_trust_scores(diagnostics)` compute the same quantities from a raw
fit payload / diagnostics mapping (for batch or offline analysis).

## Supervised SAE

`gamfit.sae_supervised` fits a manifold dictionary jointly with a supervised
GLM head, so the learned atoms are predictive of a label on the rows where one
is available (semi-supervised: `supervised_mask` selects them):

```python
fit = gamfit.sae_supervised(
    X, Y, supervised_mask,        # (N, p) data, (N,) labels, (N,) bool mask
    K=16, d_atom=2,
    atom_topology="circle",       # default
    family="auto",                # GLM family for the head; "auto" infers it
    head_formula=None,            # optional formula over the latent coords
    sae_kwargs=None,              # extra kwargs forwarded to sae_manifold_fit
    fit_kwargs=None,              # extra kwargs forwarded to the head fit
)
```

It returns a `SaeSupervisedFit` carrying `sae` (the fitted `ManifoldSAE`),
`model` (the GLM head), `supervised_mask`, `latent_names`, `response_name`,
`n_train`, and `n_supervised`. Methods: `fit.report()` (a metrics dict) and
`fit.predict(X)` (label predictions through the frozen atoms + head).

## Checkpoint dynamics and layer transport

These track how the *same* dictionary atom moves as you sweep a third axis —
training checkpoints or model layers — the OLMo-trajectory capability.

`gamfit.sae_checkpoint_dynamics` takes a grid of per-checkpoint decoder
evaluations and reports how each atom drifts across checkpoints with an
anytime-valid stability test:

```python
dyn = gamfit.sae_checkpoint_dynamics(
    decoder_grid,                 # decoder evaluations across checkpoints
    checkpoint_ids=["step10k", "step20k", ...],
    atom_names=["atom0", "atom1", ...],
    latent_grid,                  # shared latent grid the atoms are evaluated on
    alpha=0.05,
)
```

`gamfit.layer_transport_fit` aligns one atom's coordinates between two layers,
and `gamfit.layer_transport_ladder` chains the pairwise transports across a
sequence of layers:

```python
t = gamfit.layer_transport_fit(coords_from, coords_to,
                               topology_from="circle", topology_to="circle",
                               layer_from=0, layer_to=1)
ladder = gamfit.layer_transport_ladder(coords, topology="circle", layers=None)
```

## Cross-fit alignment

`gamfit.align(fit_a, fit_b)` matches atoms between two independent fits (e.g.
two seeds, or two checkpoints) so the same latent shape is identified across
runs — the basis for reproducibility and trajectory analysis.

## Visualization and benchmarking

`gamfit.plot_atom(fit, k, ax=None)` draws one atom's fitted curve and band;
`gamfit.plot_fit(fit)` draws the whole dictionary. `gamfit.sae_benchmark(...)`
runs a single synthetic recovery benchmark and `gamfit.sweep_sae_benchmark()`
sweeps the coherence/coverage/K/topology grid, returning a list of result rows.

## Torch-native SAE: `gamfit.torch.ManifoldSAE`

A trainable `nn.Module` (differentiable, GPU-capable) mirror of the closed-form
primitive lives in `gamfit.torch`. Each atom is a parametric curve in ambient
`ℝ^D`; a shared encoder produces per-atom on-manifold coordinates and
amplitudes from each input batch.

```python
import torch
from gamfit.torch import ManifoldSAE, ManifoldSAEConfig

cfg = ManifoldSAEConfig(
    input_dim=D, n_atoms=F, intrinsic_rank=2,    # intrinsic_rank default 2
    atom_manifold="circle",                       # circle / cylinder / sphere / product
    atom_basis="duchon",                          # duchon / bspline / fourier
    basis_order=2, n_basis_per_atom=8,
)
sae = ManifoldSAE(cfg)
out = sae(x)               # forward → ManifoldSAEOutput(z, x_hat, positions, ...)
fitted = sae.fit(x, max_iter=None, random_state=0, learning_rate=None)
```

`forward` returns a `ManifoldSAEOutput` dataclass with fields `z`, `x_hat`,
`positions`, `amplitudes`, `curves`, `gate`, `assignments`, `reml_score`,
`lambdas`, `raw_magnitudes`. `ManifoldSAE.fit(...)` delegates to
`gamfit.sae_manifold_fit` and shares the Rust kernel, so the closed-form and
torch paths return identical numerics on equivalent configs. The torch-interop
details (config fields, the closed-form `.fit()` restrictions on cylinder /
bspline / decoder penalties) are in the
[manifold smooths gallery](manifold-smooths.md#torch-side-gamfittorchmanifoldsae)
and [torch interop guide](torch.md).
