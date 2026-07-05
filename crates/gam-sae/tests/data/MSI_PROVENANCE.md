# Provenance — `qwen3_l11_l17_l23_theta.json`

Real per-row circle-chart angles `θ` at three residual-stream layers of
**Qwen3.5-35B-A3B**, consumed by `tests/atlas_real_transport.rs` to compute a
composed end-to-end error bound and the composition-triangle loop holonomy on a
genuine model transport.

## How it was produced

- **Activations**: the MSI cache
  `${GAM_MSI_DATA}/msae_l17/data/shards/*.safetensors`,
  each shard holding `acts_L11`, `acts_L17`, `acts_L23` (residual stream,
  hidden size 2048) for the same token rows. The first 8000 rows were used.
- **Chart fit**: one cyclic SAE atom per layer via `gamfit`'s
  `sae_manifold_fit` (the released wheel in MSI `saevenv`, version noted in the
  fixture header), using the fitted-circle machinery in
  `examples/chart_transport_l11_l23.py` (`fit_layer_circle`). All three charts
  are anchored to ONE ambient-parallel-transport gauge
  (`anchor_gauges_to_first_layer`) — per-layer label pinning would absorb any
  real rotation into the gauges and read zero holonomy by construction, so it is
  deliberately NOT done. Each layer's `θ` is in `[0, 2π)`.
- **Driver / job**: `~/msi-node/atlas_theta_dump.py` submitted via
  `~/msi-node/atlas_theta.sbatch` on the `preempt-gpu` partition (A40).
- The Rust side (transport fitting, O(2) classification, contract composition,
  loop holonomy) all runs locally from this `θ` fixture — the heavy SAE fit is
  the only MSI step.

## Schema

```json
{ "model": "...", "layer_keys": ["acts_L11","acts_L17","acts_L23"],
  "n_tokens": 8000, "theta": { "acts_L11": [ ... ], "acts_L17": [ ... ], "acts_L23": [ ... ] } }
```
