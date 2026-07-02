# Manifold-Crosscoder — cross-layer curved atoms as shared charts

Design doc for WS-I (ambition seam). Companion to `crosscoder.py` (the standard
Anthropic-2024 shared-encoder / per-layer-decoder Crosscoder already in the tree) and
`layer_transport.py` / `chart_transfer.rs` (the inter-layer transport machinery). It
specifies the object, its reduction onto the existing `sae_manifold_fit` FFI, and the
small Rust seams requested from **A2-sac-rust**. No Rust is implemented here.

## 1. What a Crosscoder is, and what changes

The Anthropic-2024 Crosscoder (`crosscoder.py`) ties a single shared encoder over the
concatenation of activations from several layers to a *family* of per-layer decoders
`W_dec[ℓ] : R^{n_atoms} → R^{d_ℓ}`. Because the code (latent) `z` is shared across
layers by construction, the per-layer decoder **column norms** `‖W_dec[ℓ][:, k]‖₂`
say how strongly atom `k` is expressed in layer `ℓ` — the cross-layer readout
(`atom_layer_affinity`, `harmonic_atoms`). But atom `k` is a **scalar latent**: a
single direction `W_dec[ℓ][:, k]` per layer, scaled by one number `z_k`.

The manifold-Crosscoder keeps everything structural about that picture and replaces
the scalar latent with a **chart**. A cross-layer atom `k` is:

- one **shared intrinsic coordinate** `t_k(x) ∈ M_k` per token (a circle angle, an
  interval position, …) — the direct analogue of the shared code `z_k`, but living on
  a 1-manifold instead of the non-negative real line;
- one **shared gate** `a_k(x)` (the IBP/threshold assignment) deciding whether the
  atom is active for that token — shared across layers exactly as `z_k`'s support is;
- a **family of per-layer decoder frames** `B_k^{(ℓ)} : basis(M_k) → R^{d_ℓ}`, so the
  atom's contribution to layer `ℓ` is the *curve image* `Φ_k(t_k(x)) · B_k^{(ℓ)}`,
  not a scaled direction.

The readout survives verbatim but sharpens: `‖W_dec[ℓ][:, k]‖₂` becomes the decoder
**frame norm** `‖B_k^{(ℓ)}‖_F`, the strength with which the *shape* (not a direction)
is written into layer `ℓ`. `harmonic_atoms` (present in every layer) and
`atom_layer_affinity` (per-layer normalized strength) carry over unchanged in meaning.
What is *new*: because the shared object is a chart, the same atom now tells you (a) the
token's position on the feature (the angle), not just its magnitude, and (b) via
`layer_transport`/`chart_transfer`, whether the blocks between layers **transport** that
chart rigidly (a degree-±1 isometric rotation) or **recompute** it — the "which block
rotates the circle" diagnostic. A scalar-latent Crosscoder cannot pose that question;
there is no coordinate to transport.

## 2. Two coupling regimes for the shared coordinate

There are two honest notions of "shared across layers", and the manifold-Crosscoder
supports both:

- **(H) Hard-tied.** One coordinate `t_k(x)` drives every layer's decoder frame:
  `x^{(ℓ)} ≈ Σ_k a_k(x) · Φ_k(t_k(x)) · B_k^{(ℓ)}` with the *same* `t_k(x)` in every
  block. This is the literal analogue of the Crosscoder's shared `z`. It assumes the
  feature's coordinatization is invariant across the chosen layers up to the decoder
  frame — true exactly when the intervening blocks transport the chart isometrically.
- **(T) Transport-tied.** Each layer has its own coordinate `t_k^{(ℓ)}(x)`, and the
  layers are linked by a fitted transport `t_k^{(ℓ+1)} = h_k^{(ℓ)}(t_k^{(ℓ)})`
  (`layer_transport.fit_transport`, degree-±1 circle map). The shared object is then
  the *equivalence class of charts under transport*, and the `‖B_k^{(ℓ)}‖_F` readout
  is joined by the transport's `isometry_defect` per hop: near-zero ⇒ that block is a
  TRANSPORT layer, large ⇒ a COMPUTE layer (`inference/layer_transport.rs` §Isometry
  defect). Regime (H) is the special case `h_k^{(ℓ)} = id`.

The key SAC observation (below) is that **regime (H) reduces to the existing K=1 fit
with zero new machinery**, and regime (T) is (H) plus a Python-level transport fit that
already exists — so the ambition object is nearly free.

## 3. The SAC reduction — a cross-layer atom is a K=1 stacked fit

SAC (SAC_PLAN Part 2) builds `K` from `K=1`: forward births on a running residual `R`
under a structured `Σ`, each birth a proven K=1 curved fit. A **cross-layer** atom is
that same K=1 fit, run on the **horizontally-stacked per-layer residuals with a single
shared coordinate**. Concretely, for layers `ℓ = 1..L` with residuals
`R^{(ℓ)} ∈ R^{n×d_ℓ}` (the running SAC residual restricted to layer `ℓ`'s columns):

1. **Stack the ambient target.** Form `X_stack = [R^{(1)} | R^{(2)} | … | R^{(L)}]`
   `∈ R^{n × Σd_ℓ}` — exactly the Crosscoder's `x_concat`, but on the residual, and
   used as the **reconstruction target** (`sae_manifold_fit`'s `X`), not an encoder
   input. Record the block boundaries `layer_dims = [d_1, …, d_L]`.

2. **Fit one shared-coordinate atom.**
   `sae = gamfit.sae_manifold_fit(X_stack, K=1, d_atom=1, atom_topology="circle")`.
   The fit produces exactly one chart: one coordinate `t_i` per row (the shared
   coordinate, tied across layers *by construction* — there is a single chart), one
   gate/assignment `a_i` (the shared gate), and one decoder `B ∈ R^{M × Σd_ℓ}` mapping
   the chart basis `Φ(t) ∈ R^{n×M}` into the whole stacked ambient.

3. **Read off the per-layer decoder frames.** Column-slice `B` by the block
   boundaries: `B^{(ℓ)} = B[:, off_ℓ : off_ℓ + d_ℓ]`. Then
   `x^{(ℓ)} ≈ a_i · Φ(t_i) · B^{(ℓ)}` is precisely the manifold-Crosscoder decode of
   layer `ℓ`, and `‖B^{(ℓ)}‖_F` is the cross-layer expression readout.

That is the whole reduction: **stacking = horizontal concat of residual blocks;
coordinate-tying = a single K=1 chart over the stacked ambient; per-layer frames =
column slices of the one decoder.** Every ingredient — the K=1 curved fit, the shared
gate, the isometry gauge, ARD over the (now stacked) decoder — is already in the tree.
SAC's forward-birth loop then yields *cross-layer* atoms one at a time on the stacked
residual, with the guard stack bypassed exactly as in the single-layer SAC path.

### Mapping onto concrete `sae_manifold_fit` inputs

| Crosscoder concept | manifold-Crosscoder object | `sae_manifold_fit` input |
|---|---|---|
| `x_concat` (encoder input) | stacked residual **target** | `X = hstack(R^{(ℓ)})` |
| shared code `z_k` | shared coordinate `t_k` + gate `a_k` | the fit's `coords[k]`, `assignments[:,k]` |
| per-layer decoder `W_dec[ℓ]` | per-layer frame `B_k^{(ℓ)}` | column slice of `decoder_B` by `layer_dims` |
| decoder col-norm readout | frame norm `‖B_k^{(ℓ)}‖_F` | `‖decoder_B[:, block_ℓ]‖_F` |
| `n_atoms` | births from SAC | `K` grown by SAC (one K=1 fit per atom) |
| decoder-weighted L1 (cross-layer coupling) | per-block scale balance | see FFI seam #1 |
| warm start | tied/transport-tied coordinates | `t_init` (already exists) |
| — (no analogue) | transport `h_k^{(ℓ)}` between layers | `layer_transport.fit_transport` (regime T) |

Regime (T) adds one step: fit each atom's per-layer coordinate `t_k^{(ℓ)}` by running
the K=1 fit per layer (or reading the shared fit's per-block projection), then
`fit_transport(t_k^{(ℓ)}, t_k^{(ℓ+1)})` to obtain `h_k^{(ℓ)}` and its isometry defect.
This needs no Rust beyond what `layer_transport` already exposes.

## 4. Why this is faithful (and where it is not, yet)

- **Cross-layer coupling.** The Crosscoder couples layers two ways: the shared code and
  the decoder-weighted L1 (which stops the encoder shrinking `z` while a decoder inflates
  `W_dec`). SAC gets the shared code for free (one chart over the stack). The
  decoder-scale gaming is *already* handled by the manifold fit's isometry gauge
  (`isometry_weight`, `g/ḡ` unit-speed pin — decoder-scale-invariant value **and**
  curvature, #795) plus ARD, so there is no separate L1 to tune. The one place the naive
  stack is unfaithful is **per-layer scale imbalance**: residual norms differ across
  layers, so a raw `hstack` lets high-norm layers dominate the shared chart and biases
  ARD. That is the single real seam (#1 below).

- **Gates.** The IBP/threshold gate is per-atom and, on the stacked fit, automatically
  shared across layers — matching the Crosscoder's shared-support semantics. Nothing to
  add.

- **Per-layer reconstruction R².** The Crosscoder reports `per_layer_r2`. On the stacked
  fit this is a per-block R² of `Φ(t)·B^{(ℓ)}` against `R^{(ℓ)}`, computable in Python by
  slicing, but cleaner as a native block-aware readout (seam #2).

## 5. FFI seams requested from A2-sac-rust

These are small, additive, and each has a zero-FFI Python workaround so WS-I is not
blocked on them — they make the reduction faithful and ergonomic rather than possible.

1. **Per-block ambient weighting (the only load-bearing seam).** Add an optional
   `block_weights: list[(usize, f64)]` or `feature_weights: &[f64]` (length `Σd_ℓ`) to
   the `sae_manifold_fit` inner loss so the reconstruction MSE / evidence / EV is
   weighted per layer, i.e. each block `R^{(ℓ)}` enters at `w_ℓ · ‖R^{(ℓ)} − Φ(t)B^{(ℓ)}‖²`.
   This is exactly per-block whitening of the stacked residual (`w_ℓ = 1/σ_ℓ²`) and is
   what keeps a high-norm layer from capturing the shared chart. It is a diagonal weight
   on the ambient residual — it does **not** touch the coordinate/gate solve structure,
   only the block that already forms the residual Gram. *Zero-FFI workaround:* pre-scale
   each block by `1/σ_ℓ` before `hstack` and post-multiply `B^{(ℓ)}` by `σ_ℓ`; correct
   for reconstruction but it perturbs the isometry-gauge normalizer `ḡ` (which averages
   over the stacked decoder), so the native weight is preferred for clean ARD/evidence.
   *SPEC note:* `σ_ℓ` must be REML/estimand-derived (e.g. the block's `Σ` diagonal from
   WS-A's `StructuredResidualModel`), not a hand-set constant — no magic knobs.

2. **Block descriptor + per-block readout (ergonomic).** Accept an optional
   `layer_dims: list[usize]` tagging the stacked ambient into `L` contiguous blocks, and
   surface in the fit payload: per-block reconstruction R² and per-block decoder-frame
   norms `‖B_k^{(ℓ)}‖_F`. This turns the manifold-Crosscoder readout
   (`atom_layer_affinity`, `harmonic_atoms`, `per_layer_r2`) into a native, tested field
   instead of caller-side column slicing that must re-derive block offsets. *Zero-FFI
   workaround:* slice `decoder_B` and call `sae_manifold_reconstruction_r2` per block in
   Python (what the WS-I harness does today).

3. **(Optional, regime T) coordinate-coupling hook.** For transport-tied atoms, a way to
   fit the `L` per-layer charts under a *shared* births/gate but *per-layer* coordinates
   linked by `h_k^{(ℓ)}`. This is fully constructible in Python on top of the existing
   `t_init` warm-start + `layer_transport.fit_transport`, so it is **not** requested as
   Rust work now; flagged only so the SAC driver's `fit_stagewise` signature leaves room
   for a per-atom coordinate-coupling callback later. No action required unless regime
   (T) becomes the default.

Net: seam #1 is the only one that changes fit *numerics*; #2 is a readout convenience;
#3 is a Python composition. The manifold-Crosscoder is, as SAC_PLAN §WS-I states,
"nearly free" — a K=1 stacked fit with a diagonal per-block weight.
