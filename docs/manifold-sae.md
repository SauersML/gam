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
    atom_topology="circle",     # exact token; omit for native discovery
    assignment="softmax",           # production default
)

print(fit)              # ManifoldSAE(K=16, n=..., p=..., topology='circle', ...)
print(fit.summary())    # K, active dims, avg active atoms, reconstruction R², ...
recon = fit.reconstruct_training()  # exact stored (N, p) training reconstruction
```

`sae_manifold_fit` returns a [`ManifoldSAE`](#the-manifoldsae-result). Both
`gamfit.sae_manifold_fit` and `gamfit.ManifoldSAE` are top-level exports.

### `sae_manifold_fit` parameters

The full signature, with defaults (keyword-only arguments follow the `*`):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `X` | required | `(N, p)` data to decompose |
| `K` | required | dictionary size (number of atoms) |
| `d_atom` | `2` | intrinsic dim per atom (int, or per-atom list) |
| `atom_topology` | `None` | exact global topology token; omitted means native `auto` discovery |
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
| `alpha` | `None` | assignment-concentration seed (`float`, `None`, or exact policy `"auto"`) |
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
| `separation_barrier_strength` | `None` | optional native cross-atom separation strength override |
| `promote_from_residual` | `True` | admit evidence-certified residual promotions during native structure search |

## Topology types

Each atom carries a topology, set globally by `atom_topology=` or per atom by
passing a list to `atom_basis=`:

The string is resolved by `SaeAtomBasisKind` in the Rust core. The full set
of typed shapes:

| `atom_topology` / `atom_basis` | Intrinsic dim `d_atom` | Underlying basis | Shape |
| --- | --- | --- | --- |
| `linear` | 1 | affine rank-1 line | true linear atom `gamma(t)=b0+t*b1` |
| topology `circle`; basis `periodic` | 1 (each axis if `d>1`) | periodic Fourier, sine-first per harmonic: `[1, sin(2π·h·t), cos(2π·h·t), …]` | closed loop, seamless at the wrap |
| `euclidean` | any | monomial / polynomial patch | open Euclidean patch (a "line" at `d_atom=1`) |
| `duchon` | any | Duchon thin-plate RKHS | open Euclidean patch with the thin-plate roughness Gram |
| `sphere` | 2 | lat/lon product chart, basis `[1, x, y, z, xy, yz, xz]` | sphere chart with pole singularities (longitude is gauge-degenerate at the poles; the quadratic part is not rotationally invariant) |
| `torus` | 2 (each axis) | doubly-periodic Fourier | torus, seamless on both axes |
| `cylinder` (**discovery-only**) | 2 | periodic circle axis ⊗ flat line axis | cylinder `S¹ × ℝ`, periodic in axis 0, open in axis 1. The line-axis roughness is a canonical `[0,1)` **reference-domain** penalty (not an intrinsic roughness integrated over all of `ℝ`, which would diverge for polynomials), so the line coordinate's scale/origin matter through that reference interval. **Not seedable**: cylinder atoms are birth-discovered by the structure search, not accepted as a closed-form `atom_topology` / `atom_basis` seed (the seed path rejects it). |
| `poincare` | any | monomial patch, hyperbolic roughness metric | Poincaré-ball tangent patch at curvature `c = −1` (wiggle measured in hyperbolic arc length) |

`duchon` and `euclidean` share the same flat-`ℝᵈ` latent **domain**, but they
use *different* decoders and declared reference-function seminorms. `euclidean`
is a monomial/polynomial patch. `duchon` is an RKHS/spline surface: a
radial-kernel block `Φ_radial(t)·Z` (centered on a fixed set of centers) plus a
null-space polynomial block `P(t)` (see `DuchonCoordinateEvaluator` in the Rust
core), penalized by its validated thin-plate function Gram. A
`euclidean`-vs-`duchon` comparison therefore differs in both the basis family and
the reference seminorm. `poincare` likewise reuses the Euclidean tangent chart
and monomial decoder but declares a hyperbolic reference seminorm: the conformal
Dirichlet energy `∫ gᵃᵇ ∂_a f ∂_b f dμ_g` of the Poincaré ball pulled back to a
fixed set of reference coordinates
(`gam_geometry::manifolds::poincare::conformal_dirichlet_penalty` at curvature
`c = −1`). Those reference coordinates are validated and consumed when the atom
or a structural reparameterization is constructed. The resulting Gram is frozen;
fitted latent coordinates and decoder coefficients do not redefine it. Missing
or invalid Poincaré reference coordinates are an error, never a silent flat-Gram
fallback. For `d = 1` the declared tangent chart is intrinsically flat but runs
at half arc length, so this reference Gram is exactly `½` the flat first-jet
Dirichlet Gram. Tokens are exact and case-sensitive. Removed aliases and
unknown precomputed seed kinds are errors rather than compatibility conversions.

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
smoothing weights selected by a custom penalized quasi-Laplace criterion. It
uses the solver's PSD/Gauss--Newton factor and explicit rank charges around the
converged penalized mode; because the smooth assignment priors are improper, it
is not normalized LAML, REML, or model evidence. Each piece plays a distinct role
(default state in parentheses):

- **Reconstruction.** Squared error between `Z` and the sparse sum of
  per-atom decoded points. Reported as `fit.reconstruction_r2`.

- **Gate sparsity (`assignment=`, canonical).** The per-token, per-atom gate
  is selected by the assignment prior. The four supported kinds are
  `"softmax"` (default), `"ordered_beta_bernoulli"`, `"threshold_gate"`, and
  `"topk"`.
  The ordered Beta--Bernoulli route uses relaxed indicators
  `z_ik = σ(ℓ_ik/τ)` directly in reconstruction. Its independent column rates
  satisfy `π_k ~ Beta(a_k, 1)` with
  `a_k = μ_k/(1−μ_k)` and ordered means
  `μ_k = (α/(α+1))^(k+1)`. These means define a geometric shrinkage schedule;
  the columns do not share latent sticks and the model is therefore not an IBP.
  Integrating each nuisance rate exactly gives the per-column penalty
  `−log a_k − logΓ(M_k+a_k) − logΓ(N−M_k+1) + logΓ(N+a_k+1)`, where
  `M_k = Σ_i z_ik`. Logit, concentration, Hessian, and penalized
  quasi-Laplace channels all
  differentiate this same scalar. The prior mean is not multiplied into the
  reconstruction, so shrinkage is scored exactly once. `"softmax"` is a dense,
  simplex-normalized gate. `"threshold_gate"` is the smooth bounded gate
  `σ((ℓ−threshold)/τ)` with its exact logistic derivative; its threshold is
  configured by `threshold_gate_threshold=`. `"topk"` is a distinct hard-support
  model and requires `top_k=`; smooth assignment families reject that argument
  and are never truncated. `tau=` sets the smooth-gate temperature.

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
  near arc length. This gauge matters because each smoothing prior is defined
  relative to a declared reference chart/measure; the smoothing Gram does not
  move with the fitted decoder. Set the weight to `0.0` to disable the gauge.

- **Smoothness** (`smoothness_weight=1.0`). Each atom declares a fixed
  reference-function seminorm. If
  `S_ref[i,j] = ⟨Lφ_i,Lφ_j⟩_(ν_ref)` and
  `f_B(t)=Bᵀφ(t)`, then bilinearity gives
  `‖Lf_B‖²_(ν_ref)=Σ_c b_cᵀS_ref b_c=tr(BᵀS_refB)`. The implemented prior is
  therefore exactly `½ λ tr(BᵀS_refB)`, with gradient `λS_refB` and Hessian
  `λ(S_ref⊗I)`. Finite-/cyclic-difference, Duchon, and caller-provided Grams are
  explicit representations of their declared reference operator and measure;
  they are validated as finite, symmetric, and positive semidefinite. Poincaré
  atoms instead build the conformal-Dirichlet Gram at their declared frozen
  reference coordinates (see the topology table). This is a final-function
  seminorm in a fixed reference geometry. It is not the moving intrinsic bending
  energy of the current decoder, and no omitted `dS/dB` or `dS/dt` terms exist.

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

The fitted band is stored directly on each Rust-owned atom, evaluated along the
atom's own coordinates (so it reports uncertainty exactly where the data
lives):

```python
atom = fit.atoms[0]
atom.decoder_covariance   # (M_k*p, M_k*p), row-major (basis, channel) flat layout
atom.shape_band_coords    # (G, d_k)
atom.shape_band_mean      # (G, p)
atom.shape_band_sd        # (G, p)
lower = atom.shape_band_mean - 1.96 * atom.shape_band_sd
upper = atom.shape_band_mean + 1.96 * atom.shape_band_sd
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
    dense covariance and shape-band surfaces, so the curve and its uncertainty
    band can be rendered again without refitting. The on-disk
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

atom = fit.atoms[0]
# The band coordinates are an evenly-strided subset of the per-token
# coordinates, so shape_band_mean traces the shape across the fitted range.
in_range = ((atom.shape_band_coords[:, 0] >= p5[0]) &
            (atom.shape_band_coords[:, 0] <= p95[0]))
typical_curve = atom.shape_band_mean[in_range]  # central 90% of use
typical_sd    = atom.shape_band_sd[in_range]
```

For out-of-sample tokens, `fit.converged_latents(X)["coords"]` returns all
per-atom coordinates from one frozen-decoder solve, so the same extent and
percentile calculation applies to new data.

### Per-atom curvature report

Fresh SAE fits also carry a first-class per-atom curvature report:

```python
curv = fit.curvature_report
curv["atoms"][0]["kappa_hat"]  # fitted estimate, when present in the report
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
| `reconstruction_r2` / `penalized_loss_score` / `penalized_quasi_laplace_criterion` / `dispersion` | fit quality / negative penalized loss / terminal custom quasi-Laplace value (PSD/Gauss--Newton factor plus rank charges; lower is better, not normalized evidence) / noise scale |

Methods: `reconstruct_training()`, `predict(X)` / `reconstruct(X)`,
`encode(X)`, `converged_latents(X)`, `reconstruct_from_assignments(codes)`,
`frozen_dictionary()`, `summary()`, `description_length()`, `steer(...)`,
`attach_fisher(...)` / `detach_fisher()`, and the strict `to_dict` /
`from_dict` / `to_json` / `from_json` / `save` / `load` serialization surface.

### Out-of-sample inference

`X` is the data to reconstruct. The fitted object is immutable: held-out
inference solves assignments and coordinates against the frozen decoder and
the smoothing state selected by the fit. It does not accept an eager encoder,
warm-start tensors, or a second training objective.

The out-of-sample surface:

```python
reconstruction = fit.predict(X)       # (N, p), same operation as reconstruct(X)
gates = fit.encode(X)                 # (N, K)
latents = fit.converged_latents(X)    # one coherent solve
latents["fitted"]                     # (N, p)
latents["assignments"]                # (N, K), exactly the applied codes
latents["logits"]                     # (N, K)
latents["coords"]                     # [ (N, d_k) ]
```

Call `converged_latents` when several outputs are needed; separate
`predict`/`encode` calls each perform their own frozen-decoder solve.

## Steering and causal intervention

`fit.steer(atom_k, metric_row, amplitude, t_from, t_to)` builds a **causal
intervention plan** that
moves one atom's latent coordinate from `t_from` to `t_to` and reports the
resulting ambient delta with output dosimetry:

```python
plan = fit.steer(
    atom_k=3,
    metric_row=0,
    amplitude=1.0,
    t_from=np.array([0.1]),
    t_to=np.array([0.4]),
)
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

The immutable artifact exposes the native report as `fit.certificates` and the
serialized structure certificate as `fit.structure_certificate_json`. These
are the claims certified during fitting; reading the model does not rerun or
retune the certification procedure.

The same machinery is exposed at top level for working with raw claim ledgers:
`gamfit.e_bh_dictionary_certificate(...)` and
`gamfit.plan_probe_for_contested_claim(...)` (probe design for a contested
claim).

## Trust, diagnostics, and curvature

Per-atom trust scores fold tangent conditioning, activation frequency, and
support into a single `[0, 1]` score:

The fitted diagnostics mapping is available as `fit.diagnostics`; per-atom
shape uncertainty is available on `fit.atoms[k].shape_band_*`, and curvature
is available in `fit.curvature_report` when the fit emitted it.

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
held-out predictive density (deterministic cross-fitted stacking). Each outer
training fold selects the free-mixture and ring-cluster orders using only that
fold's training rows before scoring its untouched rows; the all-data rung fits
are reporting/deployment fits and never choose an outer predictive model. The
two mixture rungs also carry certified, rank-aware Laplace evidence.
`winner_class` names the best predictive model-selection procedure, while
`reporting_winner` names its all-data fitted representative. The circular
topology verdict instead aggregates
the stacking mass of the smooth-circle and ring-of-clusters densities before
comparing it with the aggregate non-circular mass; this is invariant to an
otherwise arbitrary split between two predictors of the same circular class.
It is a top-level export, used throughout `tests/sae/`, and pairs naturally with
`fit.coords[k]` from
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
| `winner_class` | `"circle"`, `"euclidean"`, `"mixture"`, or `"ring_clusters"`; the winning outer-held-out procedure |
| `reporting_winner` | the all-data representative (`"mixture_k{k}"` / `"ring_clusters_k{k}"` for a discrete class) |
| `circle_wins` | bool, total circular stacking mass exceeds total non-circular mass |
| `circular_stacking_weight` / `noncircular_stacking_weight` | stacking mass aggregated within the two topology classes |
| `circular_margin` | total circular mass minus total non-circular mass |
| `mixture_reporting_k` / `ring_clusters_reporting_k` | orders selected by the all-data reporting fits; never used to choose an outer-fold predictor |
| `mixture_fold_selected_k` / `ring_clusters_fold_selected_k` | training-only order selected in each outer fold |
| `mixture_fold_k_histogram` / `ring_clusters_fold_k_histogram` | counts of the fold-local orders, for stability/provenance |
| `candidate_names` / `stacking_weights` | per-candidate names and held-out stacking weights |
| `negative_log_evidence` | per-candidate rank-aware negative log evidence |
| `headline` | `"stacking"` or `"evidence"` — which criterion produced the verdict |
| `is_cross_class` | bool, the race crossed shape classes |
| `matched_controls` | full verdicts after shuffling these supplied coordinates and replacing them by a covariance-matched Gaussian |
| `control_false_circle_floor` | circular-win fraction across those two adjudicator-input controls |
| `dictionary_mean_l0` | the dictionary sparsity supplied alongside the rate |
| `detection_floor` | Marchenko–Pastur reconstruction-energy edge when `n_eff`, `ambient_p`, and `dispersion_r` are supplied together; otherwise `None` |

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
and projection/subspace search. Use `run_shape_controlled_census` at pipeline
entry. It invokes the exact same deterministic `(matrix, seed)` callback on the
observed activations and both controls, isolates callback mutation with private
matrix copies, and retains only one control matrix at a time:

```python
def complete_pipeline(matrix, seed):
    # Fresh fit: SAE -> grouping -> projection/search -> adjudication.
    return run_census(matrix, seed=seed)

controlled = gamfit.run_shape_controlled_census(
    activations,
    complete_pipeline,
    control_seed=11,
    pipeline_seed=11,
)
print(controlled.observed)
print(controlled.per_dimension_shuffle)
print(controlled.covariance_matched_gaussian)
```

Float32 activation corpora stay float32 through the control generator; stable
means/covariances and the covariance eigendecomposition are accumulated in
float64 without materializing an `n × p` float64 copy. Float64 inputs remain
float64. The callback contract requires a fresh deterministic fit and receives
the same `pipeline_seed` for all three runs.

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
converged-latent solve. `out.penalized_loss_score` is the inner fit diagnostic,
while `out.penalized_quasi_laplace_criterion` is the terminal custom native
quasi-Laplace criterion. It is not relabeled as LAML, REML, or model evidence.
`out.selected_smooth_lambdas` reports the smoothing
precisions selected by that fit. The adapter has no trainable parameters and
rejects inputs requiring gradients. Its state dict serializes the complete
native fit, so loading it restores the same inference model.
