# Manifold SAE: a sparse dictionary of typed shapes

`gamfit.sae_manifold_fit(...)` fits a **sparse manifold dictionary**. The
data matrix `Z` (shape `(N, p)`, one ambient vector per token) is
reconstructed as a sparse mixture of `K` *atoms*. Each atom is a small,
low-dimensional **typed shape** — a line, circle, sphere, torus, or
Euclidean patch — carrying its own ambient embedding (a decoder block) and a
per-token coordinate on that shape. A token's reconstruction is the sum of
the atoms it activates, evaluated at their per-token coordinates.

The result is interpretable by construction: instead of opaque feature
vectors, each atom answers **where it lives** (a typed manifold embedded in
ambient space), **what shape it is** (the topology and its fitted curve),
and **how confident** that shape is (a posterior band).

```python
import numpy as np
import gamfit

Z = ...  # (N, p) activations / embeddings to decompose

fit = gamfit.sae_manifold_fit(
    X=Z,
    K=16,                       # dictionary size
    d_atom=1,                   # intrinsic dim per atom (int, or per-atom list)
    atom_topology="circle",     # line/circle/sphere/torus/euclidean (or per-atom atom_basis=)
    assignment="ibp_map",       # IBP-MAP sparsity
)

print(fit)              # ManifoldSAE(K=16, d_atom=1, atom_topology='circle', ...)
print(fit.summary())    # K, active dims, avg active atoms, reconstruction R², ...
recon = fit.predict(Z)  # (N, p) reconstruction; == fit.fitted on training Z
```

`sae_manifold_fit` returns a [`ManifoldSAE`](#the-manifoldsae-result). Both
`gamfit.sae_manifold_fit` and `gamfit.ManifoldSAE` are top-level exports.

## Topology types

Each atom carries a topology, set globally by `atom_topology=` or per atom by
passing a list to `atom_basis=`:

| `atom_topology` | Intrinsic dim `d_atom` | Underlying basis | Shape |
| --- | --- | --- | --- |
| `circle` (alias `periodic`) | 1 | periodic Fourier | closed loop, seamless at the wrap |
| `euclidean` | any | Duchon / thin-plate patch | open Euclidean patch (a "line" at `d_atom=1`) |
| `sphere` | 2 | intrinsic S² kernel | sphere, no pole artefacts |
| `torus` | 2 | doubly-periodic Fourier | torus, seamless on both axes |

A `d_atom=1` Euclidean atom is the **line** primitive; a `d_atom=1` circle is
the **loop**. Mixing topologies across atoms is supported — pass an
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
  is selected by the assignment prior. The canonical choice is `"ibp_map"`
  (Indian Buffet Process MAP): it adapts the *number* of active atoms per
  token and produces **true zeros** rather than a soft simplex. Alternatives
  are `"softmax"` (dense, simplex-normalized) and `"jumprelu"`
  (hard threshold). `top_k=` optionally caps the per-token active set.

- **Cross-atom decoder incoherence** (`decoder_incoherence_weight=1.0`, **on
  by default**, issue #671). The separability lever. For `K >= 2` it
  penalizes the squared output-space cross-Gram `||B_j B_k^T||_F^2` between
  the `(M_k, p)` decoder blocks of *co-activating* atom pairs, weighted by
  their empirical co-activation `mean_n gate_j·gate_k`. This drives co-firing
  atoms toward perpendicular ambient subspaces, conditioning the joint solve
  and making the decomposition identifiable. Set the weight to `0.0` to
  disable.

- **Nuclear-norm embedding-rank selection** (`nuclear_norm_weight=1.0`, **on
  by default**; `nuclear_norm_max_rank=` optional cap, issue #672). Adds a
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

- **Isometry gauge** (`isometry_weight=0.0`, **off by default**, issue #673).
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
to load on a single feature cluster, issue #240), and
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

This is an **honest epistemic posterior**, not a cosmetic ribbon: it shrinks
as `~1/sqrt(N)` with more data, scales with the reconstruction dispersion
`fit.dispersion`, and **fans out automatically** for a poorly-identified
atom (a near-singular Schur block widens the band). It is a *different,
tighter* quantity than the per-observation data scatter — with many tokens
the manifold is pinned far more tightly than any single noisy observation.

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
`project(X, k)` and `per_atom_latent_for(X)` (coordinates),
`shape_uncertainty(k)`, `coordinate_range(k)`, `typical_shape(k)`,
`curvature()` / `atom_curvature(k)`, `summary()`, `get_decoder()` /
`get_anchors()`, and `to_dict` / `from_dict` / `save` / `load`.

### Out-of-sample and encoder distillation (issue #357)

`X=` is the data to reconstruct — it is **not** a warm start.
To seed the joint solve from an amortized encoder's per-token prediction,
pass `a_init` (assignment logits `(N, K)`) and/or `t_init` (coordinates
`(K, N, D_max)`) and a small `n_iter` for a bounded refinement; read the
converged supervision targets back with `fit.converged_latents(X)` (and the
standalone `encode` / `project`). This enables the "encoder predicts →
solver refines → distill the gap" loop.

## Torch mirror: `gamfit.torch.ManifoldSAE`

A trainable `nn.Module` mirror of the closed-form primitive is documented in
the [manifold smooths gallery](manifold-smooths.md#torch-side-gamfittorchmanifoldsae).
`ManifoldSAE.fit(...)` delegates to `gamfit.sae_manifold_fit` and shares the
Rust kernel, so the two return identical numerics on equivalent configs.
