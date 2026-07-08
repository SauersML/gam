# Code-space manifolds in Qwen3.6 — the weekday circle is real; unsupervised RD missed it

**Corrected verdict.** A supervised ground-truth probe proves the weekday **circle is
present** in Qwen3.6-35B — in the raw residual stream at layers 11, 17, 23 (permutation-null
`p≈0.002`, correct cyclic calendar order) **and preserved in the K=32000 SAE code space at
L17** (`p<0.001`). The earlier unsupervised rate-distortion sweep that found "no bits-saving
1-parameter family and no calendar cycle" was **a detection/corpus artifact, not absence**:
the one manifold we can ground-truth is *rare* in the SuperGPQA corpus L17_train was harvested
from, so its co-fire never crosses a frequency threshold, and it lives in distributed centroid
geometry rather than a clean per-token top-2 secant 7-cycle. So: the code-space manifold thesis
is **confirmed by ground truth**; the unsupervised RD pipeline is not sensitive enough to
surface a rare feature on a mismatched corpus.

## The decisive test — supervised weekday-circle falsifier

DOSE weekday battery: 70 prompts = 10 templates × 7 weekdays, last-token residual harvested
through Qwen3.6-35B-A3B (CPU) at L11/L17/L23. For the 7 weekday centroids we test a RING:
Pearson `corr(calendar ring-distance min(|i−j|,7−|i−j|), centroid distance)` over the 21
day-pairs, with a **20,000-permutation label null** p-value, plus angular calendar-order
recovery in the top-2 PCA plane.

### RAW activation space — does the MODEL encode the ring?

| layer | attention type | ring_corr | perm-p | calendar-cyclic | angular order |
|---|---|---|---|---|---|
| L11 | full | **+0.606** | 0.0027 | ✓ | Thu-Fri-Sat-Sun-Mon-Tue-Wed |
| L17 | **linear** | **+0.660** | 0.0020 | ✓ | Thu-Fri-Sat-Sun-Mon-Tue-Wed |
| L23 | full | **+0.677** | 0.0024 | ✓ | Wed-Thu-Fri-Sat-Sun-Mon-Tue |

All three layers ring, significantly, in correct cyclic calendar order. **This is not a
"wrong layer" / attention-type story** — the linear-attention L17 encodes the weekday circle
as strongly as the full-attention layers. (Attention types confirmed from the config:
`full_attention_interval=4`, full layers = [3,7,11,15,19,23,…]; L17 = linear.)

### SAE CODE space (L17, K=32000, active=32) — does the DICTIONARY preserve it?

`ring_corr +0.469`, `perm-p = 0.0009`, near-calendar order (one adjacent swap),
`recon_EV = 0.257`. **The SAE preserves the weekday circle** at high significance. The
reconstruction EV of 0.26 (vs 0.77 on the training corpus) reflects the dictionary — fit on
SuperGPQA reasoning tokens — reconstructing simple weekday sentences under distribution shift,
not a space/hook mismatch (harvest per-dim rms 0.071 matches the tier0 scale 0.071).

Plottable geometry: `weekday_probe_raw.npz` — per layer, the 7 centroid 2D coords + 70
per-token 2D coords + per-token weekday label + per-token angle (RAW plane; L17 also CODE).

## Why the unsupervised RD sweep missed it (the located bug)

The unsupervised pipeline (below) detects 1-parameter families by thresholding **top-2 secant
co-fire** and requiring **exact cycle topology**. It cannot see the weekday circle because:
1. **Corpus rarity.** L17_train is SuperGPQA reasoning rollouts; weekday tokens are scarce, so
   the weekday co-fire edges never reach the `frac≥2e-5` threshold. The circle is real but
   *rare* in this corpus.
2. **Detection strictness.** The ring is in the distributed centroid geometry, not a clean
   per-token 2-hot secant walk; and the calendar check demanded `topology == "cycle"`
   (`b1==1`, all degree-2), so a ring carrying one chord (`b1==2`) is discarded.
So the unsupervised negative bounds *this detector on this corpus*, not the model.

## The unsupervised RD sweep (for the record — bounds the detector, not the model)

Detection: T1-exact top-32 codes → top-2 secant co-fire edges (streamed) → connected
components → graph Betti (Euler char) → Fiedler seriation. Re-code each group firing as
`(group_id, t, scale)`; measure bits/token vs EV deterministically. N=200k, seeds 0/1.

- Flat coding reaches EV **0.7735** at 735 bits. Manifold max-fidelity EV is **below flat at
  every co-fire threshold** (best 0.7713 @0.36% coverage; −0.064 @3% coverage). No crossover.
- `n_cycle_groups = 0` at every threshold, both seeds — no cycle *surfaced by this detector*.
- Topology is trees + one dense `b1=55` ~130-atom blob.

### Nuisance peel (candidate bug #1 — checked)
The `b1=55` blob is a **frequency-hub cluster** (its atoms fire 12–37× the mean), though there
is **no single pos0 mega-sink** here (the most-frequent atom fires on only 3.7% of tokens).
Peeling hub atoms (firing-rate > threshold) from the co-fire graph **dissolves the blob**
(topology → trees/paths + one `b1=7`; within-block edge frac → 0.01) but yields **still zero
cycles, coverage ~1.5%, still dominated** — the negative is robust to the peel. So the blob was
a genuine nuisance artifact, but removing it does not surface the weekday circle (it is rare,
per above), consistent with the located bug.

### Dictionary health (candidate bug #3 — ruled out)
The L17 K=32000 block dictionary is well-formed: `frac_dead_blocks = 0.0`, orthonormality dev
4.6e-8, EV_insample 0.799. Not undertrained. (Small-N smoke fits at N=6k showed 72% dead — an
undertraining artifact of the smoke, not the real dict.)

## Files
- `weekday_probe_raw.npz` — plottable circle geometry (raw + L17 code), per layer.
- `weekday_circle_summary.json` — per-layer ring_corr, perm-p, calendar order, recon_EV.
- `harvest_q36.py`, `weekday_circle_all.py` — supervised harvest + circle probe.
- `code_space_manifold.py` (`--peel-hub-rate`), `rate_distortion.json`,
  `discovered_groups.json`, `summary_seed{0,1}.json` — the unsupervised RD sweep.

## Reproduce
```
# supervised falsifier (CPU, ~6 min: 35B load + 70 forwards + probe)
python harvest_q36.py --model $R/models/qwen3.6-35b-a3b --layers 11 17 23 --out weekday_acts_q36.npz
python weekday_circle_all.py            # -> weekday_probe_raw.npz + summary
# unsupervised RD sweep (amdsmall 64c, ~9 min)
python code_space_manifold.py --n-tokens 200000 --frac-sweep 5e-4,2e-4,1e-4,5e-5,2e-5 --seed 0
```

## Open follow-ups
- L11/L23 **code-space** panels (full-attn dictionaries) once their K=32000 fits finish — raw
  already rings for both; expect the SAE to preserve it there too.
- A **weekday-enriched** corpus (or relaxed near-cycle topology + supervised anchor) to let the
  unsupervised detector surface the circle it currently misses — turning the ground-truth
  circle into an unsupervised discovery.
