# Provenance — `qwen3_l11_l17_l23_theta.json`

Real per-row circular chart angles `θ` at three residual-stream layers of
**Qwen3.5-35B-A3B**, consumed by `tests/atlas_real_transport.rs` to compute a
composed end-to-end error bound and the composition-triangle loop holonomy on a
genuine model transport.

## How it was produced

- **Activations**: the MSI cache `${GAM_MSI_DATA}/msae_l17/data/shards/*.safetensors`,
  each shard holding `acts_L11`, `acts_L17`, `acts_L23` (residual stream, hidden
  size 2048) for the same token rows. The first 4000 rows were used.
- **Coordinate**: the angle in each layer's **top-2 principal plane** of the
  centered activations — center, take the leading two eigenvectors of the ambient
  gram `Xᶜᵀ Xᶜ` as an orthonormal plane frame, project, and read
  `arctan2 → θ ∈ [0, 2π)`. This is the geometric circle coordinate of the
  activation cloud (the same 2-plane the `chart_transport_l11_l23.py` plane frame
  recovers), computed directly from the real activations.
- **Why not the SAE atom angle**: the example's `fit_layer_circle` reads the
  angle off a K=1 cyclic SAE atom (`gamfit.sae_manifold_fit`). At the MSI wheel
  (gamfit 0.1.248) that fit **live-locks** on this data — the Strong-Wolfe line
  search fails at BFGS iter 1 and backtracks for 13+ minutes without advancing,
  the K=1 pathology its own code comments flag. The top-2 plane angle needs no
  iterative fit and is the honest geometric circle coordinate; the loose planarity
  (top-2 plane holds ~12% of variance on this general-corpus cache) shows up
  faithfully downstream as large isometry defects, never hidden. Filed as a gam
  issue against gamfit.
- **Driver**: `~/msi-node/atlas_theta_dump.py`, run on the MSI login node (gram +
  `eigh`, a few seconds).
- The Rust side (transport fitting, O(2) classification, contract composition,
  loop holonomy) all runs locally from this `θ` fixture.

## Why a closed triangle needs no cross-layer gauge

The loop holonomy composes `h_ab, h_bc, h_ac⁻¹` around a CLOSED triangle back to
L11, so each layer's arbitrary plane gauge/orientation enters exactly twice —
once as a transport source and once as a target, inversely — and cancels around
the loop. The measured net `O(2)` element and the composition-law verdict are
therefore gauge- and orientation-invariant, no cross-layer plane alignment
required.

## Schema

```json
{ "model": "...", "coordinate": "top2-pca-plane-angle of real residual-stream activations",
  "layer_keys": ["acts_L11","acts_L17","acts_L23"], "n_tokens": 4000,
  "theta": { "acts_L11": [ ... ], "acts_L17": [ ... ], "acts_L23": [ ... ] } }
```

## Measured result (recorded from `tests/atlas_real_transport.rs`, n=4000)

| edge | winding | phase (rad) | isometry defect | \|h′\|max | resid rms |
|------|:---:|---:|---:|---:|---:|
| L11→L17 | +1 | +0.333 | 3.04e-1 | 2.397 | 7.26e-1 |
| L17→L23 | +1 | −0.073 | 1.72e-2 | 1.235 | 4.18e-1 |
| L11→L23 (direct) | +1 | +0.331 | 2.61e-1 | — | 8.48e-1 |

- **Composed ε (L11→L17→L23 shadowing bound)** = **1.315** (per-stage 0.897, 0.418):
  the L11→L17 hop dominates and is amplified by the later Lipschitz — the
  early-defect-amplified ordering the composer predicts. L11→L17 reshapes the
  chart metric (`|h′|max ≈ 2.4`, a COMPUTE hop); L17→L23 is near-isometric
  (`|h′|max ≈ 1.2`, a TRANSPORT hop).
- **Loop holonomy** (triangle `h_ab·h_bc·h_ac⁻¹`): net_sign **+1**, net_angle
  **−0.070 rad**, derived tolerance **Σdefect = 0.089 rad**. Since
  `|net_angle| < tolerance`, the **composition law HOLDS** at the loop's own
  defect scale — the two-hop composition equals the direct L11→L23 as an `O(2)`
  element within the measured noise (measure-don't-latch: a nontrivial verdict is
  not asserted when the loop's own defects cannot exclude the identity).
