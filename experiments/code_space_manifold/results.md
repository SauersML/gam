# Code-space 1-parameter manifold detection + rate-distortion on the frozen Qwen3.6 K=32000 dictionary

**Verdict (honest, both seeds): NEGATIVE.** On the msae_l17 (Qwen3.6, layer 17) block
dictionary, the code stream does **not** decompose into 1-parameter secant families that
buy bits, and it contains **no calendar cycles** (no 7-cycle weekday, no 12-cycle month).
Manifold-coding never sits below-and-left of flat-coding in the bits-vs-EV plane: at every
co-fire threshold the manifold re-code reconstructs the corpus *worse* than plain flat
coding, even with unlimited coordinate precision. The co-fire structure that *does* exist
is real (genuinely cross-block, not the dictionary's own block pairs) but rare and mostly
tree-/blob-shaped, not clean paths or circles.

## What was tested (no new fit, no hang)

The thesis: the manifold is structure in the **code stream** of the already-fitted flat
dictionary. A token at latent *t* is claimed to fire two *adjacent secant atoms* with
barycentric amplitudes ~`scale·(1−u, u)`; as *t* sweeps, the top-2 code walks a PATH or
CYCLE through the co-fire graph, two knots at a time. This is a pure **detection** on the
frozen K=32000 dictionary + its T1-exact code stream — deterministic, hang-proof.

- **Dictionary**: `msae_l17/t1_out/decoder_K32000.npy` (K=32000, unit-norm atoms; a
  block dictionary, G=16000 blocks × block_size 2). Data: `L17_train.f32.npy`
  (1,204,602 × 2048), tier0_recentered space. **N = 200,000** random tokens, seeds 0 & 1.
- **Codes (STAGE 1)**: T1-exact `top-active + active-set ridge-LS`, `active=32` (the frozen
  T1 recipe; index set = top-|score|, amplitudes = least-squares on the active set).
- **Detect (STAGE 2)**: top-2 "secant edge" co-fire counts (streamed via
  unique-on-encoded-pair-keys — never a dense K×K or N×G matrix) → threshold → union-find
  connected components → per-component **graph Betti** (exact Euler characteristic:
  `b1 = E − V + 1`; path = two deg-1 / rest deg-2, cycle = all deg-2) → Fiedler spectral
  seriation (knot order) → decoder-space adjacency + barycentric two-hot + sign-agreement.
- **Re-code + price (STAGE 3)**: each group firing re-coded as `(group_id, t, scale)`;
  reconstruction is **deterministic** — manifold recon = flat-base recon (all 32 actives)
  − the two secant atoms coded flat + the barycentric secant interpolation. With L1 scale
  `|c0|+|c1|` the interpolation reproduces the top-2 **exactly** when they are
  seriated-consecutive knots, so the measured EV drop is purely *top-2 non-adjacency +
  sign disagreement* — the literal thesis signature, measured not assumed.
- **RD as a CURVE across thresholds**: co-fire edge threshold swept over
  `frac ∈ {5e-4, 2e-4, 1e-4, 5e-5, 2e-5}` (edge counts 100→4 at N=200k), trading coverage
  against re-code distortion.

## Rate-distortion result (seed 0; seed 1 identical in shape)

Flat coding reaches its full fidelity cheaply:

| scheme | bits/token | EV |
|---|---|---|
| flat, b_amp=6 | 671 | 0.7683 |
| flat, b_amp=8 | 735 | **0.7732** |
| flat, b_amp=12 | 863 | 0.7735 |

Manifold coding, swept over the co-fire threshold (max-fidelity = unlimited t/amp bits, the
best EV the re-code can ever reach):

| frac | #groups | coverage (firing slots) | manifold max-fid EV | distortion vs flat | within-block edge frac | #cycles |
|---|---|---|---|---|---|---|
| 5e-4 | 2 | 0.36% | 0.7713 | −0.0022 | 0.12 | 0 |
| 2e-4 | 8 | 1.2% | 0.7431 | −0.0304 | 0.11 | 0 |
| 1e-4 | 29 | 3.0% | 0.7093 | −0.0642 | 0.08 | 0 |
| 5e-5 | 46 | 1.6% | 0.7580 | −0.0154 | 0.02 | 0 |
| 2e-5 | 228 | 2.3% | 0.7591 | −0.0144 | 0.00 | 0 |

At the operating quantization (headline `frac=1e-4`), manifold coding costs **721 bits/token
at EV 0.689** vs flat **735 bits/token at EV 0.773** — fewer bits but strictly, dominatingly
*worse* EV. **No manifold point is below-and-left of any flat point at any threshold.** There
is no crossover.

### Why it fails
- **Max-fidelity EV < flat at every threshold.** Even with unlimited coordinate precision,
  re-coding the discovered secants as adjacent-knot interpolations *loses* reconstruction —
  i.e. the top-2 co-firing atoms are frequently **not** seriated-adjacent knots and/or
  disagree in sign. The code's dominant pair is not a barycentric secant of a smooth path.
- **Coverage is tiny** (0.4–3% of firing slots). Loosening the threshold multiplies groups
  (2 → 228) but they fragment into small noisy trees; distortion is *worst* at the mid
  threshold (`frac=1e-4`, −0.064) where the graph collapses into one dense `b1=55`
  ~130-atom blob that mis-seriates badly.
- **Topology is trees + a blob, not circles.** Across thresholds the components are
  dominated by trees and paths plus a few high-genus graphs; **cycle count is 0 everywhere**.
  Paths do exist (6 at `frac=1e-4`, 93 at `frac=2e-5`) but they carry no bits win.

### Calendar check: NONE
Zero cycle-topology groups of size 6–8 (weekday) or 11–13 (month) at any threshold, in
either seed. `n_calendar_cycle_groups = 0`. No ground-truth-validated 1-parameter circle
surfaced — the existence proof the strong thesis needs is absent at this dictionary.

### The co-fire is real, just not a manifold
The **within-block edge fraction is 0.08–0.12** at the useful thresholds (→0 at the loosest),
so the detected co-fire edges are genuinely **cross-block** — not the dictionary's own
orthonormal block pairs. So the negative is not an artifact of trivial within-block pairing:
there is real cross-atom co-activation structure, it is simply not organized as
bits-saving 1-parameter secant families.

## CRITICAL CAVEAT — layer 17 is (reportedly) a linear-attention layer

Qwen3.6 uses a **hybrid** stack: most layers are linear-attention (gated-DeltaNet / SSM),
with **full self-attention only periodically (~every 4th layer)**. **Layer 17 — the layer
this entire experiment ran on — is a linear-attention layer.** A linear-attention layer's
residual stream mixes tokens through a decaying state rather than explicit
content-addressed retrieval, so it is plausible that clean *positional/periodic* code-space
manifolds (weekday/month circles) live at **full-attention** layers and are absent from
linear-attention layers. **This negative result may therefore be a statement about
linear-attention layers, not about the model.** It should not be read as "Qwen3.6 has no
code-space manifolds" — only "L17 (linear-attention) has none that buy bits or form circles."

## FOLLOW-UP (queued, cheap, activations already banked)
`msae_l17/` already holds harvested train activations for **L11 and L23** as well as L17
(`L{11,23}_train.f32.npy`, 9.87 GB each). If the every-4th-full pattern holds, **L11 and L23
are full-attention layers.** The follow-up is the *identical* detection+RD pipeline on a
full-attention layer — it requires only (1) a tier0 recentering for that layer and (2) a
K=32000 block-dictionary fit on it (the T1 block fit, which is **not** the stagewise hang
path and completes in minutes), then the same `code_space_manifold.py --data L{11,23}_train`.
**If calendar cycles appear at a full-attention layer but not at L17, that is a real finding
about where manifold structure lives in the stack.** Status: launching this follow-up next.

## Reproduce
```
# MSI amdsmall, 64 cores, ~9 min at N=200k (STAGE1 ~215s dominates)
python code_space_manifold.py --n-tokens 200000 \
  --frac-sweep 5e-4,2e-4,1e-4,5e-5,2e-5 --headline-frac 1e-4 --seed 0
```
Artifacts: `rate_distortion.json` (flat curve + per-threshold manifold sweep, both seeds),
`discovered_groups.json` (headline `frac=1e-4` groups: atoms, knot sequence, edges, Betti,
topology, barycentric/sign-agreement/decoder-adjacency stats), `summary_seed{0,1}.json`.
Driver: `code_space_manifold.py`; sbatch: `code_space_manifold.sbatch`; post-analysis:
`cs_postanalyze.py`.
