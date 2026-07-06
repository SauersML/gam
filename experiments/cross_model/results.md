# Matched-token cross-model transport

Status: complete on MSI job `12595475` (`interactive-gpu`, 2x L40S, elapsed
3:35). This reruns the Qwen3-8B vs Qwen3.6-35B comparison on matched text
positions instead of unrelated activation banks.

## Setup

- Source text: `Salesforce/wikitext`, `wikitext-103-raw-v1`, train split.
- Alignment: exact shared tokenizer character spans in the same documents. Only
  spans where both tokenizers produced the identical `(start, end)` character
  interval were kept.
- Matched rows: 30,000 token spans from 181 WikiText documents.
- Layers: Qwen3-8B L18 of 36, Qwen3.6-35B-A3B L20 of 40.
- Coordinate: model-local top-2 PCA activation-plane angle after peeling the
  top PCA direction in each model.
- MSI outputs: large `.npy` arrays are kept off-repo in the run's scratch
  directory (`cross_model_matched_harvest/`); committed numbers are in
  `numbers.json`.

## Peel diagnostics

| model/layer | top peeled PC variance fraction | top-2 plane score norm |
|---|---:|---:|
| Qwen3-8B L18 | 0.991311 | 6155.31 |
| Qwen3.6-35B-A3B L20 | 0.570772 | 104.25 |

The matched Qwen3-8B L18 sample is again dominated by the position/sink-like
top direction, so the top-PC peel is load-bearing before reading semantic
geometry from the plane angle.

## Transport result

| metric | value |
|---|---:|
| winding | -1 |
| phase | -2.325492 rad (-133.241 deg) |
| O(2) defect | 0.640206 |
| shift resultant | 0.353960 |
| reflect resultant | 0.359794 |
| gauge defect scale | 0.011547 |
| smooth isometry defect | 1.147805 |
| smooth residual RMS | 1.349586 |
| min directional derivative | -3.340592 |
| topology preserved | false |

Verdict: **not a shared feature by this coordinate**.

The matched-token result is now meaningful: the two models saw the same text
spans in the same order, but the top-2 post-peel activation-plane coordinates do
not form a rigid or topology-preserving circle transport. The O(2) defect
`0.640206` is far above the matched-sample gauge scale `0.011547`, and the shift
and reflection resultants are not separated enough to avoid the `mixing`
classification.
