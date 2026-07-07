# Curved vs Linear on real Qwen3-8B L18 (sink-peeled) — Results

Head-to-head of a **curved chart atlas** (K local charts) vs a **global linear
code** on the real Qwen3-8B wikitext L18 residuals (`resid_L18.npy`), after the
position-0 attention-sink is peeled. Run:
`curved_vs_linear.py --positions positions_L18.npy --peel pos0 --layer 18`
(job 12670777, rc=0, N=30000 subsample, 30000 x 4096).

## Peel audit (pos0 null-gated causal peel, on REAL L18)

- directions peeled: **1** (`peel_is_causal=True`)
- observed positional absorption: **0.9026**; permuted-position null_max: **0.00020**
  (permutation p < 1/201) — the sink is unambiguously real, the null collapses.
- **cos(sink_dir, PC1) = 1.000** — the pos0 first-token indicator IS PC1 on L18.
- frac rows at pos0 ~ 0.0059; var frac top PC before peel ~ 0.99.

This is the independent confirmation that on L18 the pos0 causal peel and the
variance top-PC peel coincide (used by the Run-1 ceiling `--raw-ok` path).

## Head-to-head — matched coordinates/row (d active atoms)

| K  | d | atlas EV | linear EV | curved gain |
|----|---|----------|-----------|-------------|
| 64 | 2 | 0.946    | 0.929     | **+0.017**  |
| 32 | 1 | 0.941    | 0.924     | **+0.017**  |
| 16 | 1 | 0.938    | 0.924     | +0.015      |
| 32 | 2 | 0.943    | 0.929     | +0.015      |
| 16 | 2 | 0.941    | 0.929     | +0.012      |
| 8  | 1 | 0.935    | 0.924     | +0.011      |

Linear global-PCA EV by M: {1:0.924, 2:0.929, 4:0.933, 8:0.938, 16:0.943,
32:0.949, 64:0.955}.

## Head-to-head — matched description length (bits, reverse water-filling)

D* = 2.833e-02 (derived noise floor = smallest global covariance eigenvalue).

| K  | matched bits/row | atlas EV | linear EV | MDL gain | gate bits |
|----|------------------|----------|-----------|----------|-----------|
| 64 | 2240.99          | 1.000    | 0.989     | **+0.011** | 5.55    |
| 32 | 3302.90          | 0.999    | 0.993     | +0.007   | 4.63      |
| 16 | 4794.57          | 0.999    | 0.996     | +0.003   | 3.51      |
| 8  | 6339.88          | 0.998    | 0.997     | +0.001   | 1.88      |

## Interpretation

Curvature is **modestly exploitable** on real, sink-peeled L18: the curved atlas
beats the global linear code by **+0.011 to +0.017 EV** at matched coordinates/row,
and by **+0.001 to +0.011 EV** at matched description length (the MDL gain grows
with K). The gain is real (positive under both capacity currencies, and the peel
is causally validated) but small.

This is **consistent with Run-1`s INFORMATION_CEILING** verdict (eta=0.9317): the
single curved chart sits ~7% below the top-M linear envelope because that gap is
an information ceiling of the data, and once you spend a real atlas (K charts) you
recover a small, genuine curvature advantage — the manifold-SAE premise holds
directionally on real data without overclaiming.

The actual gamfit manifold-SAE stack is importable in this env (ManifoldSAE,
sae_manifold_fit, StagewiseSAE, ...), so the atlas here is a faithful proxy for
the fitter`s curved-code family.

## Artifacts

- `curved_vs_linear.json` — full numbers (peel audit, both head-to-heads, RD curves).
- `curved_vs_linear.png` — EV vs coords/row (left) and EV vs bits MDL-fair (right).
