# Manifold SAE: a sparse dictionary of typed shapes

`gamfit.sae_manifold_fit(...)` fits a **sparse manifold dictionary**. The
data matrix `Z` (shape `(N, p)`, one ambient vector per token) is
reconstructed as a sparse additive combination of `K` *atoms*. Each atom is a small,
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
    assignment="softmax",           # production default
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
| `assignment` | `"softmax"` | gate kind: `softmax` / `ordered_beta_bernoulli` / `threshold_gate` / `topk` |
| `schedule` | `None` | `GumbelTemperatureSchedule` for annealed gates |
| `isometry_weight` | `1.0` | unit-speed gauge penalty (on by default) |
| `ard_per_atom` | `True` | ARD pruning of unused coordinate axes |
| `decoder_feature_sparsity_groups` | `None` | output-feature partition for decoder group-lasso |
| `n_iter` | `50` | joint-solve iterations |
| `sparsity_weight` | `1.0` | strength of the `coord_sparsity` coordinate-shrinkage penalty |
| `coord_sparsity` | `"scad"` | coordinate (latent `t`-block) magnitude penalty: `scad` / `mcp` / `l1` |
| `scad_mcp_gamma` | `None` | SCAD/MCP concavity (defaults SCAD 3.7, MCP 2.5) |
| `smoothness_weight` | `1.0` | roughness penalty strength |
| `alpha` | `1.0` | ARD/precision seed (`float` or a string policy) |
| `learning_rate` | `None` | optional step size override |
| `random_state` | `0` | RNG seed |
| `block_orthogonality_weight` | `0.0` | orthogonalize latent axes (needs `d_atom >= 2`) |
| `nuclear_norm_weight` | `1.0` | embedding-rank selection penalty |
| `nuclear_norm_max_rank` | `None` | optional cap on the embedding rank |
| `decoder_incoherence_weight` | `1.0` | cross-atom incoherence (separability lever) |
| `top_k` | `None` | optional cap on per-token active atoms. |
| `t_init` | `None` | coordinate warm start `(K, N, D_max)` |
| `a_init` | `None` | assignment-logit warm start `(N, K)` |
| `tau` | `None` | Gumbel-softmax temperature |
| `threshold_gate_threshold` | `0.0` | center of the smooth `threshold_gate` assignment |
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
| `circle` (aliases `periodic`, `periodic_spline`) | 1 (each axis if `d>1`) | periodic Fourier, sine-first per harmonic: `[1, sin(2π·h·t), cos(2π·h·t), …]` | closed loop, seamless at the wrap |
| `euclidean` (alias `euclidean_patch`) | any | monomial / polynomial patch | open Euclidean patch (a "line" at `d_atom=1`) |
| `duchon` | any | Duchon thin-plate RKHS | open Euclidean patch with the thin-plate roughness Gram |
| `sphere` | 2 | lat/lon product chart, basis `[1, x, y, z, xy, yz, xz]` | sphere chart with pole singularities (longitude is gauge-degenerate at the poles; the quadratic part is not rotationally invariant) |
| `torus` | 2 (each axis) | doubly-periodic Fourier | torus, seamless on both axes |
| `cylinder` (**discovery-only**) | 2 | periodic circle axis ⊗ flat line axis | cylinder `S¹ × ℝ`, periodic in axis 0, open in axis 1. The line-axis roughness is a canonical `[0,1)` **reference-domain** penalty (not an intrinsic roughness integrated over all of `ℝ`, which would diverge for polynomials), so the line coordinate's scale/origin matter through that reference interval. **Not seedable**: cylinder atoms are birth-discovered by the structure search, not accepted as a closed-form `atom_topology` / `atom_basis` seed (the seed path rejects it). |
| `poincare` (aliases `hyperbolic`, `poincare_patch`) | any | monomial patch, hyperbolic roughness metric | Poincaré-ball tangent patch at curvature `c = −1` (wiggle measured in hyperbolic arc length) |
| any other string | any | caller-supplied | `Precomputed`: a precomputed basis you attach yourself |

`duchon` and `euclidean` share the same flat-`ℝᵈ` latent **domain**, but they
use *different* decoders as well as different roughness penalties. `euclidean`
is a monomial/polynomial patch. `duchon` is an RKHS/spline surface: a
radial-kernel block `Φ_radial(t)·Z` (centered on a fixed set of centers) plus a
null-space polynomial block `P(t)` (see `DuchonCoordinateEvaluator` in the Rust
core), penalized by the thin-plate roughness Gram. A `euclidean`-vs-`duchon`
comparison therefore differs in *both* the basis family and the penalty, not the
penalty alone. `poincare`
likewise reuses the Euclidean tangent chart and monomial decoder but penalizes
roughness in the *hyperbolic* metric: its effective smoothness Gram is the
conformal Dirichlet energy `∫ gᵃᵇ ∂_a f ∂_b f dμ_g` of the Poincaré ball pulled
back to the tangent chart (`gam_geometry::manifolds::poincare::conformal_dirichlet_penalty`
at curvature `c = −1`, the single source of truth for the hyperbolic metric),
wired into the atom's `refresh_intrinsic_smooth_penalty`. It therefore differs
from the flat patch wherever the conformal factor departs from 1 — growing toward
the ball boundary (tree-/hierarchy-like structure); for `d = 1` the tangent chart
is intrinsically flat but runs at half arc length, so the Gram is exactly `½` the
flat first-jet Dirichlet Gram. String matching is case-insensitive and treats `-`
and `_` interchangeably.

A `d_atom=1` linear atom is the true rank-1 **line** primitive. A
`d_atom=1` Euclidean atom is a stronger polynomial patch, not the pure-linear
baseline used for #1026 reconstruction parity. A `d_atom=1` circle is the
**loop**. Mixing topologies across atoms is supported — pass an
`atom_basis=[...]` list and read the resolved per-atom labels off
`fit.atom_topologies` (the scalar `fit.atom_topology` collapses to `"mixed"`
when atoms disagree).

### `K` larger than the data supports

`K` is not auto-reduced. If the requested `K` (for the given `d_atom` /
`atom_topology`) exceeds what `Z` can identify — e.g. two `circle` atoms
asked to explain a single planted loop, with no second independent signal
for the second atom to claim — the co-collapse detector reseeds the
redundant atom onto a residual principal component a bounded number of
times and, if that never recovers a materially better fit, the outer solve
refuses to mint a fit rather than return a degenerate dictionary with a
duplicated atom. This surfaces as a `RemlConvergenceError` (see
[Exceptions](exceptions.md)) whose message includes the outer iteration
count, the final objective value, and the gradient norm. It is not a
transient failure: retrying with the same `K` on the same data raises
again. Lower `K`, change `atom_topology` / `d_atom`, or pass more atoms'
worth of independent signal in `Z`.

## The objective

The fit minimizes reconstruction error plus a stack of penalties, with
smoothing weights selected by REML. Each piece plays a distinct role
(default state in parentheses):

- **Reconstruction.** Squared error between `Z` and the sparse sum of
  per-atom decoded points. Reported as `fit.reconstruction_r2`.

- **Gate sparsity (`assignment=`, canonical).** The per-token, per-atom gate
  is selected by the assignment prior. The three supported kinds are
  `"softmax"` (default), `"ordered_beta_bernoulli"`, `"threshold_gate"`, and
  `"topk"`.
  The ordered Beta--Bernoulli route uses posterior-mean relaxed indicators
  `z_ik = σ(ℓ_ik/τ)` directly in reconstruction. Its independent column rates
  satisfy `π_k ~ Beta(a_k, 1)` with
  `a_k = μ_k/(1−μ_k)` and ordered means
  `μ_k = (α/(α+1))^(k+1)`. These means define a geometric shrinkage schedule;
  the columns do not share latent sticks and the model is therefore not an IBP.
  Integrating each nuisance rate exactly gives the per-column penalty
  `−log a_k − logΓ(M_k+a_k) − logΓ(N−M_k+1) + logΓ(N+a_k+1)`, where
  `M_k = Σ_i z_ik`. Logit, concentration, Hessian, and REML/LAML channels all
  differentiate this same scalar. The prior mean is not multiplied into the
  reconstruction, so shrinkage is scored exactly once. `"softmax"` is a dense,
  simplex-normalized gate. `"threshold_gate"` is the smooth bounded gate
  `σ((ℓ−threshold)/τ)` with its exact logistic derivative; its threshold is
  configured by `threshold_gate_threshold=`. `top_k=` optionally caps
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

- **Isometry gauge** (`isometry_weight=1.0`, **on by default**).
  `IsometryPenalty` drives the pulled-back metric
  `g = J^T J` toward a unit-average-speed chart, making `t` easier to read as
  near arc length. It is not required for topology comparison: the smoothness
  penalty is measured on the decoded FUNCTION's intrinsic geometry (below), not
  on the raw coordinate, so `fit.penalized_loss_score` is gauge-invariant under
  reparameterizing `t` even with `isometry_weight=0.0`. Set the weight to `0.0`
  to disable the gauge.

- **Smoothness** (`smoothness_weight=1.0`). Roughness penalty on each atom's
  decoded curve/surface, measured intrinsically so it is invariant to
  reparameterizing the latent coordinate `t` (SPEC: penalise the final function,
  not the chart). For a non-Poincaré atom the effective Gram is the total squared
  SECOND FUNDAMENTAL FORM of the decoded embedding `γ(t) = BᵀΦ(t)`,
  `∫_M ‖II‖²_g dμ`, at every latent dim `d ≥ 1`; for `d = 1` this is exactly the
  reparameterization-invariant bending `∫ κ² ds = Σ_i ‖P_N γ''(t_i)‖²/‖γ'(t_i)‖³`
  (the NORMAL-projected acceleration — a straight segment traced as `γ(t)=t²e₁`
  scores zero, whereas a raw coordinate second-difference Gram would charge
  `γ''=2e₁`). Poincaré atoms use the hyperbolic conformal Dirichlet Gram instead
  (see the topology table). A fixed finite-/cyclic-difference Gram in the latent
  coordinate is the *base operator*; the intrinsic (function-space) Gram is
  refreshed from it between assemblies via the current decoder pullback
  (lagged diffusivity). Each inner solve differentiates the frozen quadratic
  surrogate; it does not include derivatives of the geometry-dependent Gram.

- **Coordinate-magnitude penalty** (`coord_sparsity="scad"` default; `"l1"` /
  `"mcp"` alternatives). Despite the parameter name, `scad` and `mcp` do **not**
  penalize the gate/assignment: they emit a row-block `ScadMcpPenalty` on the
  latent coordinate (`"t"`) block — coordinate shrinkage *inside* an active atom
  — using `sparsity_weight=` as its strength and `scad_mcp_gamma=` as the
  concavity/taper (defaults SCAD `3.7`, MCP `2.5`). A more accurate name for the
  knob would be `coord_sparsity`. `l1` emits no coordinate penalty and keeps the
  historical assignment-prior sparsity path.

Two more knobs: `decoder_feature_sparsity_groups=` group-lassoes the decoder
over a partition of the `p` output features (encouraging each basis function
to load on a single feature cluster), and
`block_orthogonality_weight=` orthogonalizes the latent coordinate axes
(requires `d_atom >= 2`) with the exact public convention
`½·weight·Σ_{g<h} ‖T_gᵀT_h‖²_F`.

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

The band is the **joint** decoder covariance: `Cov(β_k)` is the `k`-th block of
the joint inverse Hessian, so it carries the cross-atom covariance and the
decoder↔coordinate Schur couplings, and the per-channel `sd` genuinely varies
across output features. This holds after **structure search** too — when a
certified birth / fission / fusion (or a demoted death) re-converges the whole
dictionary at a new smoothing state, the joint factor is rebuilt at the final
model, so the reported band still reflects the joint covariance of the returned
(possibly grown) dictionary, seed and born atoms alike.

!!! note "Full fitted-state persistence"
    `save` / `load` and `to_dict` / `from_dict` use the strict Rust-owned v3
    artifact schema. They retain each atom's decoder coefficients, fitted
    per-token coordinates, resolved topology, shape-band grid/mean/sd, and a
    compact per-output-channel covariance factor. Loading reconstructs the
    dense covariance surface used by `shape_uncertainty`, so the curve and its
    uncertainty band can be rendered again without refitting. The on-disk
    factor is `(p, M_k, M_k)`, not the unused dense `(M_k·p)^2` matrix.

!!! note "Degenerate atoms and the huge-`p` fallback"
    A genuinely **unidentified** atom (no active rows, or a non-SPD joint block)
    keeps an honest `NaN` band rather than a fabricated number. If the joint
    inverse-Hessian factor cannot be reformed at the final state after a
    structure move, or the ambient `p` is too large to admit the dense Schur
    factor, the band for the affected atoms degrades to a **per-atom marginal**
    `Var_c(t) = φ · Φ_k(t)ᵀ H_k⁻¹ Φ_k(t)` from that atom's own penalized inner
    Hessian — which omits the cross-atom and coordinate couplings and is
    identical across output channels — or to a `NaN` band. This is the honest
    fallback, not the joint covariance; it is used only where the joint factor is
    unavailable.

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
| `atoms` | list of Rust-owned atom objects (decoder, coords, `active_dim`, uncertainty) |
| `atom_topology` / `atom_topologies` | scalar (`"mixed"` if heterogeneous) / per-atom topology |
| `assignment` / `assignment_label` | canonical gate kind / the string you passed |
| `coords` | `[ (N, d_k) ]` per-atom per-token coordinates |
| `decoder_blocks` | `[ (M_k, p) ]` per-atom decoder matrices |
| `fitted` / `assignments` | `(N, p)` training reconstruction / `(N, K)` gates |
| `reconstruction_r2` / `penalized_loss_score` / `dispersion` | fit-quality / negative penalized-loss objective (NOT REML / evidence; `reml_score` is a deprecated read alias) / noise scale |

Methods: `predict` / `reconstruct(X)`, `encode(X)` (out-of-sample gates),
`project(X, k)`, `per_atom_latent_for(X)` and `featurize(X)` (coordinates),
`per_atom_active_set(X)`, `shape_uncertainty(k)`,
`curvature()` / `atom_curvature(k)`, `summary()`,
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
proj_k = fit.project(X, atom_k=0)        # (N, d_k) on-manifold coordinate block for the atom
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
fit.shape_uncertainty(0, n_sd=1.96)  # {"coords","mean","sd","lower","upper"} shape band
```

The observed coordinate extent of an atom is read directly off
`fit.coords[k]` (see [Typical coordinate range](#typical-coordinate-range)).

The top-level helpers `gamfit.sae_trust_diagnostics(payload)` and
`gamfit.atom_trust_scores(diagnostics)` compute the same quantities from a raw
fit payload / diagnostics mapping (for batch or offline analysis).

## Shape adjudication: `adjudicate_atom_shape`

`gamfit.adjudicate_atom_shape(coords, ...)` adjudicates the representational
shape of a 2-D point set without forcing a topology: it races a smooth **S¹
ring**, a **Euclidean Gaussian**, the best free **k-cluster Gaussian mixture**,
and a constrained **ring of clusters** whose component centers share one fitted
circle. The last class is the generative model for discrete cyclic concepts:
it explains clumpy density without throwing away cyclic order. All four race on
held-out predictive density (deterministic cross-fitted stacking), while the
two mixture rungs carry certified, rank-aware Laplace evidence. The winner is
whichever class predicts held-out points best. It is a top-level export, used
throughout `tests/sae/`, and pairs naturally with `fit.coords[k]` from
[`sae_manifold_fit`](#the-manifoldsae-result).

```python
import gamfit

verdict = gamfit.adjudicate_atom_shape(
    coords, folds=5, seed=11, mean_l0=dictionary_mean_l0
)
# coords: contiguous float64 (n, 2), n >= 4 (a few dozen points recommended);
# k_ladder=[2, 3, ...] optionally overrides the mixture orders raced.
```

Returns a dict:

| Key | Meaning |
| --- | --- |
| `winner` | `"circle"`, `"euclidean"`, `"mixture_k{k}"`, or `"ring_clusters_k{k}"` |
| `circle_wins` | bool, the winner is either circular density class |
| `circular_margin` | best circular stacking weight minus the best non-circular weight (`NaN` if weights are unavailable) |
| `mixture_k` | the mixture order selected inside the mixture rung |
| `ring_clusters_k` | the order selected inside the constrained ring-of-clusters rung |
| `candidate_names` / `stacking_weights` | per-candidate names and held-out stacking weights |
| `negative_log_evidence` | per-candidate rank-aware negative log evidence |
| `headline` | `"stacking"` or `"evidence"` — which criterion produced the verdict |
| `is_cross_class` | bool, the race crossed shape classes |
| `matched_controls` | full verdicts after shuffling these supplied coordinates and replacing them by a covariance-matched Gaussian |
| `control_false_circle_floor` | circular-win fraction across those two adjudicator-input controls |
| `dictionary_mean_l0` | the dictionary sparsity supplied alongside the rate |

Either mixture EM can refuse to certify convergence (a `GamError`). That is a
typed missing adjudication, not a negative topology verdict; record it as such
and diagnose the failed candidate rather than converting it into a winner for
another class.

### Usage caveats on real LLM activations

These were measured on real residual-stream activations (Qwen3-8B and
OLMo-2 weekday/month token sets, plus injected synthetic controls):

1. **Discrete cyclic concepts have their own density class.** Older shape races
   called confirmed weekday/month circles `mixture_k` because a uniform ring
   could not explain their clustered angular mass. `ring_clusters_k{k}` now
   fits the shared center/radius and component angles directly, with `2k+3`
   parameters instead of the free 2-D mixture's `6k-1`. A remaining
   `mixture_k` win therefore means the unconstrained centers predict held-out
   points better; it is no longer silently standing in for an untested cyclic
   arrangement.
2. **Sparse non-negative codes carry a false-circle floor.** Structureless
   controls (per-dimension-shuffled activations, covariance-matched Gaussians)
   pushed through a sparse-SAE + per-group 2-D PCA pipeline still produce
   adjudicator circle wins in double-digit percentages of groups. Never
   interpret raw circle-win rates without running the byte-identical pipeline
   on matched structureless controls.
3. **2-D PCA masks low-relative-variance rings.** A genuine ring sharing its
   group with a linear factor at ~1× its radius is already lost after a top-2
   variance projection, and at 2× the adjudicator prefers a cluster mixture
   (validated by injection). Absence of circle wins is not absence of circles.

### Running an unsupervised census honestly

If you adjudicate many feature groups (an unsupervised topology census), two
rules keep the verdict rates meaningful:

- **Gate on dictionary health, and report it.** Circle verdicts only appeared
  at mean L0 below ~300 in the measured censuses (0 circle wins across 377
  dense-dictionary groups) — so a raw verdict rate is uninterpretable without
  the code sparsity it was measured at. Report mean L0 next to any rate.
- **Run matched structureless controls.** Push a per-dimension-shuffled copy
  and a covariance-matched Gaussian copy of the same matrix through the
  byte-identical pipeline, and report verdict rates against that per-run
  false-circle floor (measured floors reached double digits), never raw.

The controls embedded in `adjudicate_atom_shape` begin at its 2-D coordinate
input, so they isolate the adjudicator's own false-circle floor. A full census
must also catch artifacts introduced by SAE training, co-activation grouping,
and PCA. Generate those controls one at a time at the census pipeline entry and
rerun every stage:

```python
for kind in ("per_dimension_shuffle", "covariance_matched_gaussian"):
    controlled_activations = gamfit.shape_matched_control(
        activations_float64, kind=kind, seed=11
    )
    # Re-run the identical SAE -> grouping -> PCA -> adjudication pipeline.
    run_census(controlled_activations, control_kind=kind)
```

One control is returned per call so corpus-scale workflows can release it
before generating the next instead of retaining both copies in memory.

`examples/topology_census_recipe.py` documents the full validated recipe.

### On real activations, lead with deterministic readouts

Everything that *established* circles on real LLM activations in the measured
studies was fit-free or supervision-seeded: label-class-mean planes,
circular-linear ordering against chart-rebuilt permutation nulls, per-label
angular binding, and layer transport on those coordinates. A manifold SAE is a
reconstruction model; its latent angle is not guaranteed to be a stable
ordering coordinate. Reconstruction EV alone does not validate that coordinate.
When ordering matters, report the deterministic projection alongside the
native fit and verify cross-seed concordance explicitly.

Across seeds, stack corresponding fitted coordinates as `(replicates, rows)` and
report their exact rotation/reflection-quotiented agreement:

```python
report = circular_concordance(torch.stack(seed_coordinates), period=1.0)
print(report.minimum_aligned_score)
print(report.pairs)
```

The report exposes every rotation and reflection score, the selected O(2) gauge,
phase shift, and a two-axis embedding-rank certificate for each replicate. It
does not impose a qualitative cutoff. If a replicate is collapsed, pairwise and
aggregate scores are `None` instead of granting a vacuous perfect alignment.

### Availability note

`adjudicate_atom_shape` is currently only available when building from
source: the released PyPI wheel predates the symbol. Source builds require
the repo's `.git` directory (the build script's tracked-file audits read the
git index); building from a source *archive* without `.git` skips those
audits with a warning.

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

## Visualization

`gamfit.plot_atom(fit, k, ax=None)` draws one atom's fitted curve and band;
`gamfit.plot_fit(fit)` draws the whole dictionary.

## Frozen Torch adapter: `gamfit.torch.ManifoldSAE`

The Torch surface wraps a converged native fit; it does not define or train a
second SAE objective.

```python
import torch
import gamfit
from gamfit.torch import ManifoldSAE

fit = gamfit.sae_manifold_fit(
    X=activations,
    K=F,
    d_atom=1,
    atom_topology="circle",
    assignment="softmax",
)
module = ManifoldSAE(fit)
out = module(torch.as_tensor(new_activations, dtype=torch.float64))
```

`out.reconstruction`, `out.codes`, and `out.coordinates` come from one native
converged-latent solve. `out.penalized_loss_score` is a fit diagnostic, not an
evidence value, and `out.selected_smooth_lambdas` reports the smoothing
precisions selected by that fit. The adapter has no trainable parameters and
rejects inputs requiring gradients. Its state dict serializes the complete
native fit, so loading it restores the same inference model.
