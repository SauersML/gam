# Real overcomplete ManifoldSAE on Qwen3-30B-A3B (32B MoE) L17

_Draft — numbers filled after the MSI a100 run completes._

## What this is

The **real** gamfit ManifoldSAE product run on **real** Qwen3-30B-A3B (MoE, "q36b")
L17 residual activations — an honest **flat-vs-curved** comparison at matched
**overcomplete** K and matched sparsity, using the **same `sae_manifold_fit`
machinery** for both arms (only the atom manifold type differs). This supersedes
the earlier 8B / global-PCA / k-means toy setup.

- Model / layer: Qwen3-30B-A3B (MoE) residual stream layer 17
- Data: `msae_l17/L17_train.f32.npy`, shape (1204602, 2048); subsample N (see below)
- Tier-0 space: `tier0_recentered.json`, `x' = (x - per_dim_mean)/global_rms_scale`
  (scale 0.0710, per-dim mean = true streamed col mean fixing the L2~1.07 offset;
  rogue dims [1269,1924,491] kept). EV baseline "zero" = tier-0 origin (train mean).
- gamfit version: 0.1.248

## The comparison (matched K, matched sparsity)

Both arms are `gamfit.sae_manifold_fit` with structure search OFF, same overcomplete
K, same `top_k` active-set cap, `d_atom=1`, differing ONLY in `atom_topology`:

- **Linear / flat SAE** — `atom_topology="linear"` (the genuinely linear baseline;
  note gamfit docs warn `"euclidean"` is a *quadratic* patch, NOT the linear control).
- **Curved SAE** — `atom_topology="circle"` (plus a small-K structure-search run that
  selects among Circle/Torus/Sphere for "which typed manifolds get chosen").

| arm | K | top_k | EV |
|---|---|---|---|
| linear SAE (atom_topology=linear) | _TBD_ | _TBD_ | _TBD_ |
| curved SAE (atom_topology=circle) | _TBD_ | _TBD_ | _TBD_ |

Curved chart types selected (small-K structure search): _TBD_

## Provenance

- Driver: `experiments/real_manifold_sae/run_real_msae_32b.py`
- Run host: MSI a100 (GPU required — sparse/manifold score backends need CUDA)
- Linear-PCA / k-means numbers here are context only, NOT the baseline (per spec).
