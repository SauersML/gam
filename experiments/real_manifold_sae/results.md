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

---

## HEADLINE (LANDED): overcomplete curved-vs-flat via the BLOCK-CHART compose lane

The `sae_manifold_fit_stagewise` / `_fit_stagewise_t2` lane **HANGS** on the 32B L17
residual (self-concordance / collapse-barrier metric bug — 75+ min silent on 4k rows),
so the stagewise arms above never completed. This section reports the real number via
the **block-chart compose lane** (`gamfit.block_sparse_dictionary_fit` +
`BlockSparseDictionaryFit.compose_block_charts`) — the SAME engine route2 used to close
the geometric wall on Qwen-8B L18 (there curved beat flat, +0.047). Here it runs to
completion on the 32B in ~4 min.

- model: **Qwen3-30B-A3B (MoE, 32B) L17**, `L17_train.f32.npy` (1,204,602 × 2048)
- centering: **tier0_recentered**; EV baseline "zero" (tier-0 origin == train mean)
- T1: **frozen overcomplete `decoder_K32000.npy`** (K=32000 = **15.6× p**, unit-norm),
  reconstructed with the frozen **RIDGE-LS** transform (per-row top-|score| active=32
  support + ridge least-squares codes; NOT dot-product — dot-product overshoots on
  correlated unit-norm atoms → EV ≈ −64)
- T2: block-chart **FLAT (24 linear blocks)** vs **CURVED (12 circle blocks × (4+4))**
  at **matched parameter budget** (96 flat units == 96 curved units), fit **stratum-local**
  (energy-exponent strata; the pooled residual fails the routability floor
  √(2 ln K / p)=0.10, so births must be stratum-local)
- N=40,000 subsample (seed 0); 10/16 strata fitted (≥512 rows); gamfit 0.1.250

| tier | EV (composed, held-in, baseline zero) |
|---|---|
| T1 alone (overcomplete K=32000, ridge-LS active=32) | **0.8505** |
| T1 + flat block-SAE (24 linear blocks) | **0.8612** |
| T1 + curved block-chart (12 circle × (4+4)) | **0.8586** |

**ΔEV(curved − flat) = −0.00258** (pooled-residual floor drop = **−0.0173**).
**Curvature does NOT help on the 32B L17 overcomplete residual** — flat wins at matched
params in **every** fitted stratum:

| stratum | rows | flat floor | curved floor | drop (flat−curved) | accepted charts | curvature proxy |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 515 | 0.6577 | 0.7737 | −0.1160 | 0 | 0.000 |
| 4 | 679 | 0.7128 | 0.8052 | −0.0924 | 0 | 0.000 |
| 5 | 933 | 0.7547 | 0.8267 | −0.0720 | 0 | 0.000 |
| 6 | 1040 | 0.7717 | 0.8386 | −0.0669 | 0 | 0.000 |
| 7 | 892 | 0.7589 | 0.8403 | −0.0814 | 2 | 0.071 |
| 8 | 590 | 0.7057 | 0.8547 | −0.1490 | 12 | 0.203 |
| 11 | 1106 | 0.7775 | 0.8865 | −0.1090 | 12 | 0.186 |
| 12 | 6638 | 0.9033 | 0.9333 | −0.0299 | 12 | 0.102 |
| 13 | 21060 | 0.9376 | 0.9491 | −0.0115 | 12 | 0.069 |
| 14 | 4967 | 0.9233 | 0.9464 | −0.0231 | 12 | 0.078 |

The compose lane **is** promoting curvature (up to 12 accepted charts/stratum, large
deviance gains, curvature proxy up to 0.20) — yet even in those strata the flat 24-block
dictionary reconstructs better than the 12-block-curved base + chart corrections at equal
budget. This is a genuine **negative for curvature on the 32B**, opposite to the Qwen-8B
L18 wall closure (+0.047) under the identical matched scheme. Honest, either sign.

### Provenance
- Driver: `experiments/real_manifold_sae/run_curved_vs_flat_32b_blockchart.py`
  (reuses `experiments/geometric_wall/wall_closure_common.py` `fit_stratum` verbatim)
- Lane: block-chart compose (`block_sparse_dictionary_fit` + `compose_block_charts`),
  **NOT** stagewise (`_fit_stagewise_t2`, which hangs). CPU only, no GPU.
- Run: MSI msismall CPU sbatch, job 12710386, gamfit 0.1.250, wallclosure_venv.
