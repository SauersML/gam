# Dimension spectrometer on real LLM residual-stream activations

**Status:** headline runs in progress on MSI — Qwen3.6-35B-A3B L17 (job
`12575326`) and Qwen3-8B L18 (job `12575327`). Finalized once `results.json`
lands. Methods + synthetic validation below are final. All compute runs on MSI
(no laptop compute); activations were already harvested on MSI (no re-harvest).

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

## Headline results
_(filled in when MSI jobs 12575326 / 12575327 complete; outputs land in
`q36b_L17/` and `q8b_L18/` on MSI and are pulled back small.)_

### Qwen3.6-35B-A3B, layer 17 (N=150k) — global d̂, mixture, PCA strata
### Qwen3-8B, layer 18 (N=150k) — global d̂, mixture, PCA strata
### Figures
- `fig_global_q36b_L17.png`, `fig_pca_strata_q36b_L17.png`
- `fig_global_q8b_L18.png`, `fig_pca_strata_q8b_L18.png`
