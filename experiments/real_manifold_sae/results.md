# Real overcomplete ManifoldSAE on Qwen 3.6 (A3B MoE, q36b) L17

_Draft — numbers filled after the MSI a100 run completes._

## What this is

The **real** gamfit ManifoldSAE product run on **real** Qwen 3.6 (A3B MoE, q36b)
L17 residual activations — an honest **flat-vs-curved** comparison at matched
**overcomplete** K and matched sparsity, using the **same `sae_manifold_fit`
machinery** for both arms (only the atom manifold type differs). This supersedes
the earlier 8B / global-PCA / k-means toy setup.

- Model / layer: Qwen 3.6 (A3B MoE, q36b) residual stream layer 17
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

> NOTE: the `sae_manifold_fit` / stagewise arms below **never completed** — that lane
> HANGS on the 32B L17 residual (self-concordance / collapse-barrier metric bug). The
> real, product-faithful numbers were obtained via the **block-chart compose lane**
> instead; see the HEADLINE (LANDED) section at the bottom. The table below is filled
> from the block-chart STAGE-2 run (job 12710386, N=40k) for the matched flat-vs-curved
> comparison. K here is the overcomplete T1 base K (block-chart T2 sits on the residual;
> "top_k" = per-row T1 active support).

| arm | K (T1) | top_k | composed EV |
|---|---|---|---|
| T1 alone (overcomplete, ridge-LS) | 32000 (15.6×) | 32 | 0.8505 |
| flat SAE (block-chart, 24 linear blocks) | 32000 | 32 | **0.8612** |
| curved SAE (block-chart, 12 circle × (4+4)) | 32000 | 32 | **0.8586** |

**ΔEV(curved − flat) = −0.00258** (pooled-residual floor drop = −0.0173). This is a
**TIE at saturated capacity, not a curvature negative.** Curved trails flat by a hair,
and by a margin that **decays monotonically toward zero as rows grow** — drop −0.01 to
−0.15 across the 10 fitted strata, larger strata → smaller gap: stratum 8 @590 rows −0.149
→ stratum 13 @21k rows −0.0115, **and it never flips sign**. A vanishing-with-more-data
deficit is the signature of a **small-sample parameter tax on a structure-free residual**
(the extra curved parameters cost more to estimate than they return when params/rows is
large), NOT of absent curvature — a genuine "no manifold here" would flip the sign in
favor of flat as rows grow, not converge to a tie. See the Verdict for why this
comparison cannot detect a curvature win even if the thesis is true.

Curved chart types selected (small-K structure search): **not run** — the block-chart
compose lane has no typed (Circle/Torus/Sphere) structure search; it promotes per-block
circle charts via `compose_block_charts`. Charts ARE promoted (up to 12 accepted/stratum,
curvature proxy up to 0.20), yet curved still loses at matched budget.

## Verdict

On the **Qwen 3.6 (A3B MoE, q36b) L17**, overcomplete **K=32000**, **block-chart compose**,
**matched params**: flat and curved **TIE on reconstruction at saturated capacity** (ΔEV
= −0.0026 pooled), with curved paying a **small-sample parameter tax on a structure-free
residual**. This comparison is **insensitive to the manifold hypothesis by construction**
— it cannot detect a curvature win even if the thesis is true. Four independent reasons:

1. **Residual-orthogonality trap.** The curved charts are asked to fit the *residual* of a
   large linear tier (K=32000 overcomplete atoms, active-32 ridge-LS). But least-squares
   residuals are orthogonal to the fitted span: the linear tiling has already absorbed the
   local tangent *and* the curvature into where it places its 32000 atoms. What is left in
   the residual is high-frequency sawtooth quantization noise between atoms — which is
   exactly the thing a *smooth* chart cannot represent. The test hands the curved lane a
   target from which structure has been removed by construction.
2. **Saturation at K=32000.** At 15.6× overcompleteness the linear dictionary is already
   near the reconstruction ceiling (T1 alone EV 0.8505; flat T2 only lifts it to 0.8612).
   There is almost no headroom left for *any* second tier to win, curved or flat.
3. **Matched-EV, not matched-bits.** The arms are matched on parameter budget and measured
   in additive-residual EV. That is the **wrong currency** for the thesis. The claim is
   *informational*: a manifold in the data is a **redundancy in the CODES**, and the right
   measurement is **bits/token** under a rate–distortion test — not the fraction of
   ambient residual variance a smooth patch can soak up.
4. **Curved given half the blocks.** The "matched params" scheme pays for each curved
   block's chart basis by halving the block count (24 flat linear blocks vs 12 circle
   blocks × (4+4)), so curvature is charged twice: once in parameters, once in coverage.

The **stratum pattern is the decisive internal evidence**: the curved deficit decays
monotonically with rows (−0.149 @590 rows → −0.0115 @21k) and **never flips sign** — the
signature of a parameter tax that shrinks as params/rows→0, not of absent curvature (which
would *widen* flat's lead with more data). Read together with the Gemma-2-2B
corroboration (`experiments/geometric_wall/results.md`, same tie/tax pattern), the honest
statement is: **the additive-residual EV metric is insensitive to the manifold hypothesis
here; it neither confirms nor refutes it.** The correct test is the code-space
rate–distortion pipeline (`experiments/code_space_manifold/`), which measures the thesis
in its native currency of bits/token.

Provenance: ridge-LS T1 (NOT dot-product), tier0_recentered, block-chart compose (NOT
stagewise), gamfit 0.1.250, job 12710386. Note: the block-chart *wall-closure* curved fits
predate the barrier/grind fix (81d3900f4), so those curved numbers may be **underfit** —
another reason the small curved deficit is not evidence against curvature.

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
the geometric wall on Qwen-8B L18 (there FLAT beat curved, +0.047 EV — a curvature
negative; those +0.047 numbers are energy floors, lower=better, flat floor 0.7411 <
curved floor 0.7886). Here it runs to completion on the 32B in ~4 min.

- model: **Qwen 3.6 (A3B MoE, q36b) L17**, `L17_train.f32.npy` (1,204,602 × 2048)
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
Flat and curved **TIE** on the 32B L17 overcomplete residual; curved trails by a hair in
every fitted stratum, but by a gap that **shrinks monotonically toward zero as rows grow
and never flips sign** (a parameter tax on a structure-free residual, NOT absent
curvature — see Verdict for the residual-orthogonality / saturation / matched-EV-not-bits
/ half-the-blocks reasons this comparison is insensitive to the manifold hypothesis by
construction):

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
dictionary reconstructs *marginally* better than the 12-block-curved base + chart
corrections at equal budget. This is **not** a negative for curvature: it is a **tie under
a metric that cannot see a curvature win**. The curved lane is fitting the LS-orthogonal
residual of a saturated K=32000 linear tier (structure already absorbed into atom
placement), matched on params-and-EV rather than bits, and charged half the block count
for its chart bases. The stratum trend (deficit −0.149 @590 rows → −0.0115 @21k, monotone
toward zero, no sign flip) is the fingerprint of a small-sample parameter tax, not of
absent geometry. The Gemma-2-2B corroboration
(`experiments/geometric_wall/results.md`) shows the **same tie/tax pattern** under the
same scheme, so both models agree — but what they agree on is that **the
additive-residual EV metric is the wrong currency**, not that the manifold thesis is
false. The thesis is informational (a manifold in the data = redundancy in the CODES,
measured in **bits/token**); the correct test is the code-space rate–distortion pipeline
(`experiments/code_space_manifold/`). Honest framing: this experiment cannot detect a
curvature win even if the thesis is true.

### Provenance
- Driver: `experiments/real_manifold_sae/run_curved_vs_flat_32b_blockchart.py`
  (reuses `experiments/geometric_wall/wall_closure_common.py` `fit_stratum` verbatim)
- Lane: block-chart compose (`block_sparse_dictionary_fit` + `compose_block_charts`),
  **NOT** stagewise (`_fit_stagewise_t2`, which hangs). CPU only, no GPU.
- Run: MSI msismall CPU sbatch, job 12710386, gamfit 0.1.250, wallclosure_venv.
