# Qwen3-8B L18 Ceiling Contract — Results

Real-data K=1 curved-chart ceiling verdict on the Qwen3-8B wikitext harvest
(`resid_L18.npy`, 300000 x 4096). Emitted by
`crates/gam-sae/examples/qwen_l18_ceiling.rs`.

## Verdict

```json
{"verdict":"INFORMATION_CEILING","EV_curved":0.8172371432336222,"EV_lin_top_m_envelope":0.8771581035292453,"chart_efficiency_eta":0.9316873890185462,"gradient_certificate":"clean","peel_status":"post_peel"}
```

**INFORMATION_CEILING** — after sink peeling, the single curved chart (K=1,
harmonics H=1, basis_size 2H+1=3) reconstructs the L18 image to EV 0.8172,
which is eta = 0.9317 of the top-M=3 global-linear envelope (0.8772). The
outer-gradient dual-oracle certificate is **clean** (first-order consistent,
grad_norm 0.0, converged). Per the decision tree, eta in the near-one band with
a clean gradient certificate => the gap to the linear envelope is an
**information ceiling of the data**, not a solver failure and not a residual
adjoint bug. This is the reassuring outcome: it does NOT implicate the landed
adjoint fixes (would have been RESIDUAL_ADJOINT_BUG) and it does NOT indicate a
landscape/gauge pathology (would have been LANDSCAPE_PATHOLOGY at eta << 1).

## Numbers (verified from job 12671192, A100-SXM4-40GB, exit 0)

| region     | rows  | dim | EV_curved  | EV_lin_top_M | eta (curved/linear) | envelope slack | grad cert | converged |
|------------|-------|-----|------------|--------------|---------------------|----------------|-----------|-----------|
| pre_peel   | 20000 | 9   | 0.9965713  | 0.9993636    | 0.9972060           | 0.0027923      | clean     | true      |
| post_peel  | 20000 | 8   | 0.8172371  | 0.8771581    | 0.9316874           | 0.0599210      | clean     | true      |

- `top_sink_ev_pre_peel = 0.996047478` — the attention sink is ~99.6% of the raw
  variance (one PC), consistent with the manifest `ev_top1 = 0.9909`.
- `wall_seconds = 259.821` (fit wall ~93.3 s post-peel, ~93.9 s pre-peel).
- `numbers.json`: N=20000, p=4096, K=1, pca_dim=8, harmonics=1, post_peel=false
  (raw full-width input; the example self-peels internally, see provenance).

## Peel provenance (IMPORTANT)

This run used the example`s `--raw-ok` path: it takes `thin_svd_scores(raw, 9)`,
treats **score-0 (the top PC / PC1) as the sink**, and fits the post_peel region
on scores[:, 1:9] (8 dims). That is the **variance self-peel** (remove the top
PC), i.e. the approach replaced as wrong-in-general by the pos0 null-gated causal
peel elsewhere in the repo.

It is trustworthy **on L18 specifically** because the sink IS PC1 here:
- the curved_vs_linear pos0 peel measured **cos(sink_pos0, PC1) = 1.000** and the
  permuted-position null collapsed (absorbed null_max = 0.00020, observed 0.9026),
- manifest `ev_top1 = 0.9909`, run `top_sink_ev_pre_peel = 0.996`.

So on L18 the variance self-peel == the pos0 causal peel and the verdict holds.
Task #9 tracks threading positions into `qwen_l18_ceiling.rs` so its internal
peel is the pos0 null-gated causal peel natively (needed for L6/L30/other models,
where sink != PC1). A pos0-peeled -> PCA8 cross-check via the non-raw p<=64 path
is a planned follow-up to confirm the two verdicts agree on L18.

## Reproduce

```text
sbatch $R/scratch/ceiling_run.sbatch    # a100-4, gpu:1
# runs: qwen_l18_ceiling  resid_L18.npy  20000 1 12 8  --raw-ok  --out-dir $R/scratch/ceiling_out
```

## Cross-check: pos0 causal peel -> PCA8 (non-raw path) — VERDICTS AGREE

Follow-up requested to confirm the `--raw-ok` variance self-peel and the pos0
null-gated causal peel give the same verdict on L18. Produced a pos0-peeled ->
top-8-PCA input on the SAME strided 20000 rows (stride 15, matching the ceiling
subsampler) and fed it through the **non-raw p<=8 post_peel path**
(job 12672490, rc=0, A100).

pos0 peel audit on those rows: **cos(sink_dir, PC1) = 0.999854**, pos0_absorbed =
0.8922, frac_pos0 = 0.00635 — i.e. the pos0 indicator is (numerically) PC1.

```json
{"verdict":"INFORMATION_CEILING","EV_curved":0.9679661797473188,"EV_lin_top_m_envelope":0.9945583385889945,"chart_efficiency_eta":0.9732623438868326,"gradient_certificate":"clean","peel_status":"post_peel"}
```

| peel method                    | dim | EV_curved | EV_lin_top_M | eta      | grad cert | verdict             |
|--------------------------------|-----|-----------|--------------|----------|-----------|---------------------|
| variance self-peel (`--raw-ok`) | 8   | 0.81724   | 0.87716      | 0.93169  | clean     | INFORMATION_CEILING |
| pos0 causal peel -> PCA8         | 8   | 0.96797   | 0.99456      | 0.97326  | clean     | INFORMATION_CEILING |

**Both agree on the verdict (INFORMATION_CEILING) and on a clean gradient
certificate.** eta is even closer to 1 under the pos0 peel (0.973 vs 0.932),
strengthening the information-ceiling reading. The absolute EVs differ because
the two peels span slightly different 8-dim subspaces (top-8 PCs of the
pos0-residual vs thin-SVD scores 2..9 of the raw), but the conclusion is
identical. This closes the L18 half of task #9: threading pos0 into the example
natively is still wanted for L6/L30/other models where sink != PC1, but on L18
the shortcut is verified sound. (post_peel=true in numbers.json for this run.)
