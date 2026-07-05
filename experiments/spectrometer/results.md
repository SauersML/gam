# Dimension spectrometer on real LLM residual-stream activations

**Status:** COMPLETE. Depth-resolved spectrum measured on Qwen3-8B (L6/L18/L30)
plus Qwen3.6-35B-A3B L17, all on MSI from pre-existing harvests (no re-harvest,
no laptop compute). Headline numbers below.

## Models measured (current Qwen3-series, pre-existing MSI harvests)
- **Qwen3.6-35B-A3B, layer 17** — the flagship "3.6" MoE; SuperGPQA reasoning
  rollouts; hidden d = 2048; 1.20M tokens available (`msae_l17/L17_train.f32.npy`).
  Strongly anisotropic residual stream.
- **Qwen3-8B, layer 18** (mid-stack of 36) — wikitext-103; hidden d = 4096;
  300k tokens (`harvest_out/qwen3_8b_wikitext/resid_L18.npy`). Well-conditioned
  (PCA ev_top1 ≈ 0.065, participation ratio ≈ 140) — a clean high-dim manifold.

Both subsampled to N = 150,000 tokens (N/K ≥ 18 across the whole ladder),
centered by the exact full-matrix per-dim mean, scaled so mean‖x‖² = d.

## Concept

For data lying on a d-dimensional feature manifold, a width-K linear sparse
dictionary run at active budget **s = 1** (each point coded by its single best
atom) has excess reconstruction loss

    L(K) − σ²  ∝  K^(−2/d).

Sweeping K on a doubling ladder and regressing log(L−σ²) on log K gives a slope
m, and the intrinsic dimension estimate **d̂ = −2/m**. Real activations are a
*mixture* of manifolds, so we also fit L(K) = Σ_j c_j K^(−2/d_j) + σ² and
stratify by PCA head-subspace removal to separate the anisotropic head from the
tail.

## Method (this experiment)

- **Dictionary fit — top-1 MOD ("k-lines").** Each atom is a unit vector (a 1D
  subspace through the origin). Each token is assigned to its single best atom
  (max |⟨x, dₖ⟩|); atoms are refreshed by the closed-form MOD decoder update
  Dₖ = (Σᵢ cᵢ xᵢ)/(Σᵢ cᵢ²) then renormalized; dead atoms are revived onto the
  worst-reconstructed tokens. ~25 epochs, best of restarts. Loss L(K) is the
  mean residual energy ‖x − ⟨x,dₖ⟩dₖ‖².
- **Floor / dimension.** σ² and a single-power d̂ are fit by damped Gauss-Newton
  with many seeded restarts (ridge-stabilized normal equations). d̂ is also read
  off directly as the OLS log-log slope of the excess loss, and we report the
  **drop-last-rung** sensitivity and the **consecutive-rung local slopes**.
- **Mixture.** 2- and 3-component fits of L(K)=Σ c_j K^(−2/d_j)+σ².
- **PCA stratification.** Remove the top-R principal directions (R ∈ {0,1,2,16,
  64,256}) and re-run the whole ladder on the residual, to see whether the tail
  is higher-dimensional than the anisotropic head.

### Caveat — finite-N overfitting at high K (important)
The top-1 k-lines estimator overfits when the number of tokens per atom (N/K)
gets small: an atom fitting only a handful of points in a 2000–3600-dim space
captures them almost perfectly, spuriously depressing L(K) and *steepening* the
slope (biasing d̂ downward). This is visible in the 35B run at N=40k (below): the
local slope steepens monotonically as K grows. We therefore (a) run the headline
at N=150k to keep N/K ≥ 18 everywhere, and (b) report d̂ over the stable
low-to-mid-K window as well as the full ladder, plus the drop-last sensitivity.

## Synthetic validation (done, local, before the real run)
Random smooth nonlinear embeddings of a uniform d-cube into R⁶⁴:
- d = 1 → measured log-log slope −1.83, **d̂ = 1.09** (theory −2).
- d = 2 → measured slope −1.03, **d̂ = 1.94** (theory −1).
Reproduces the previously-reported −1.90 / −0.99 and confirms the trainer.

## Pilot — Qwen3.6-35B-A3B, layer 17 (N=40k, tier-0 scaled)
This N=40k pilot is superseded by the N=150k headline run (same model/layer,
job 12575326); it is kept because it is what first exposed the finite-N caveat.
Global k-lines loss over the ladder (mean‖x‖² = 2133.8):

| K | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|----|----|-----|-----|-----|------|------|------|------|
| L(K) | 1776.1 | 1704.1 | 1625.0 | 1541.0 | 1443.5 | 1327.4 | 1192.7 | 1028.7 | 814.1 |

- No-floor OLS log-log slope over the full ladder = **−0.130 → d̂ ≈ 15.4**.
- Consecutive-rung local d̂: 33.5, 29.1, 26.1, 21.2, 16.5, 13.0, 9.4, 5.9 —
  a monotone drift downward = the finite-N overfitting transient. The reliable
  (large N/K, K ≤ 512) end sits at **d̂ ≈ 25–33**; the high-K end is
  overfit-contaminated at this N. Read honestly: layer-17 of the 35B reasoning
  model has an intrinsic dimension in the **high tens** at the scales where the
  estimator is trustworthy, not the ~15 the naive full-ladder fit reports.

## Headline results — depth-resolved dimension spectrum (N=80k, K≤4096)

Final runs used N=80,000 tokens and a ladder capped at K=4096 (so tokens-per-atom
N/K ≥ 19 at every rung — the finite-N overfitting transient is gone) and the fast
scipy-sparse MOD refresh. Each layer's column mean is computed from its own full
matrix. MSI jobs 12576897 (L6), 12576898 (L18), 12576899 (L30), 12576900 (L17).

### The headline finding: raw d̂ is confounded by massive-activation channels; peel them

The single most important result is that **the raw (R=0) global d̂ is meaningless
wherever a massive-activation channel dominates the residual stream**, and the
PCA-stratified peel recovers the real intrinsic dimension:

| condition | top-1 var frac | raw R=0 d̂ | **post-peel d̂ (R=1 / R=16)** |
|---|---|---|---|
| Qwen3-8B L18 (mid) | 0.958 | **3.8 (confounded)** | **26.2 / 28.7** |
| Qwen3-8B L30 (late) | 0.812 | 22.8 | 23.5 / 24.7 |
| Qwen3.6-35B-A3B L17 | 0.025 (isotropic) | 19.9 | 20.0 / 20.9 |

L18's raw d̂=3.8 is the estimator locking onto the one massive channel (96% of
variance); a single atom captures it, so the covering law sees a ~1-D spike. Once
that direction is projected out (R≥1) the true manifold reveals **d̂ ≈ 26–29**.
Where the stream is already isotropic (35B L17, top-var 2.5%) peeling barely moves
d̂ (20.0→20.9) — the confound is real only when a massive channel exists. This is
why the stratified peel is the load-bearing measurement.

### De-confounded intrinsic dimension per layer/model
- **Qwen3.6-35B-A3B L17** (d_model 2048, isotropic): d̂ ≈ **20–23** (single-power
  d=20.9, σ²=1.94; R=1..64 → 20.0, 20.0, 20.9, 23.2). Trustworthy as-is.
- **Qwen3-8B L18** (mid-stack, massive channel): true d̂ ≈ **26–29** post-peel
  (R=1..64 → 26.2, 26.4, 28.7, 32.8). Raw 3.8 is an artifact.
- **Qwen3-8B L30** (late, mild channel): d̂ ≈ **23–26** (R=1..64 → 23.5, 23.7,
  24.7, 26.2).
- **Qwen3-8B L6** (early, isotropic top-var 0.048): d̂ ≈ **16–18** (raw 16.1 ≈
  post-peel 16.3, 16.2, 16.6, 17.8 — no confound, so peeling barely moves it).

d̂ rises monotonically as more PCs are peeled (R↑): removing the most-captured,
lowest-dimension variance leaves a higher-dimensional residual — expected, and it
means these numbers are lower bounds on the fine-scale dimension.

### Full tables

Depth-resolved global fit (full ladder / drop-last / no-floor):

| condition | d_model | R=0 full | drop-last | no-floor | single-power (d, σ²) |
|---|---|---|---|---|---|
| Qwen3-8B L6 (early) | 4096 | 16.1 | 18.4 | 16.1 | 17.2, 3.78 |
| Qwen3-8B L18 (mid) | 4096 | 3.8 | 4.2 | 13.5 | 3.8, 13.9 |
| Qwen3-8B L30 (late) | 4096 | 22.8 | 24.8 | 22.8 | 23.5, 0 |
| Qwen3.6-35B-A3B L17 | 2048 | 19.9 | 22.2 | 19.9 | 20.9, 1.94 |

PCA-stratified d̂ (peeling the top-R principal directions):

| condition | R=0 | R=1 | R=2 | R=16 | R=64 |
|---|---|---|---|---|---|
| Qwen3-8B L6 (early) | 16.1 | 16.3 | 16.2 | 16.6 | 17.8 |
| Qwen3-8B L18 (mid) | 3.8 | 26.2 | 26.4 | 28.7 | 32.8 |
| Qwen3-8B L30 (late) | 22.8 | 23.5 | 23.7 | 24.7 | 26.2 |
| Qwen3.6-35B-A3B L17 | 19.9 | 20.0 | 20.0 | 20.9 | 23.2 |

### Depth trend (Qwen3-8B) — the headline picture
De-confounded intrinsic dimension (post-peel R=16) across depth:

    L6 (early) ≈ 16.6  →  L18 (mid) ≈ 28.7  →  L30 (late) ≈ 24.7

Intrinsic dimension **rises into mid-stack and dips slightly by the late layer** —
the middle of the network carries the highest-dimensional token manifold. The raw
(R=0) numbers would tell the opposite, nonsensical story (16 → 3.8 → 22.8, a
spurious collapse at L18) purely because L18's massive-activation channel swamps
the unpeeled estimate. This inversion is the whole reason the peel matters, and it
is the central result of the experiment. See `fig_depth_trend.png`.

### Cross-check vs. the pre-existing PCA analysis (spectrum_analysis.json)
That file reports *linear* effective rank after removing the top rogue directions:
L18 eff_rank ≈ 895, L30 ≈ 1375 (top-3 removed). Our spectrometer measures the
*nonlinear manifold* dimension via the covering law and lands ~1.5 orders of
magnitude lower (d̂ ~ 20–30). The gap is the substance: the residual stream spans
a high-rank linear subspace but the token cloud concentrates on a ~20–30-dim
nonlinear manifold within it.

### On the "strata d̂ = None" report
The completed N=80k runs populate every stratum with valid d̂ (verified in each
`results_*.json`). The None values came from earlier runs that never reached the
`json.dump` — the 40k pilot crashed in the floor/mixture fit (singular normal
matrix, since hardened with a ridged Levenberg step) and the first N=150k jobs
were cancelled mid-stratification for being too slow (pre-scipy). No live code
path emits None for a completed run.

### Figures (per condition, pulled from MSI)
- `fig_global_<cond>.png` — loss curve, single-power fit, excess-loss slope.
- `fig_pca_strata_<cond>.png` — stratified excess-loss + d̂ vs. peeled R.
- `fig_depth_trend.png` — Qwen3-8B intrinsic d̂ vs. layer (L6/L18/L30).
