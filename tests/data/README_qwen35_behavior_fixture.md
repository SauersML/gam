# Real behavior-anchored fixture — Qwen3.5-9B (#2015)

Row-aligned activation/behavior pair: 2000 WikiText-103-raw (test split) token
positions through **Qwen3.5-9B** (dense, 31 text layers), harvested on GPU by
`scripts/qwen_joint_behavior_harvest.py` and reduced for the in-crate gate.

## qwen35_9b_actsL21_pca64_2000.npy — shape (2000, 64) float32

Residual-stream hidden state at layer 21 (the 2n/3 auto pick), column-centered
and PCA-reduced with its own basis to 64 components, **EVR 0.549**. One row per
token position (positions 0..n−2 of each document, i.e. positions that have a
genuine next-token prediction).

## qwen35_9b_behavior_probs64_2000.npy — shape (2000, 64) float32

The model's TRUE next-token distribution `softmax(logits)` at the SAME
positions, coarse-grained exactly: columns 0..62 are the global top-63 tokens
(ranked by mean probability over the harvest), column 63 is the renormalized
tail bucket (residual vocabulary mass). Tail mass mean 0.488. Rows are exact
probability vectors up to f32 rounding — consumers must renormalize rows before
the sphere-tangent embedding (the gate does).

Why coarse-grained is honest: the sphere-tangent behavior block keeps all V−1
tangent coordinates, so V=64 keeps the fit tractable while every KL statement
downstream is EXACT for the coarse-grained readout; the coarse-graining map is
part of the recorded provenance, not a hidden approximation.

- **Use:** `crates/gam-sae/src/manifold/tests_behavior_qwen_real.rs` — the
  unified behavior-anchored entry (`run_auto_sae_behavior_fit`) on real data.
- **Full-width archive (4000 rows, acts_L10 + acts_L21 at 4096 dims, probs,
  token ids, JSON provenance):** `qwen35_9b_joint_behavior.npz` (62 MB) in
  the MSI project home.
- **Regenerate:** the harvest script in-repo; any 24 GB GPU fits the 9B model.
  (The `msae_l17` 35B shards CANNOT supply this pair: no token ids, and the
  logit-lens readout at cached layer 23/40 measures near-uniform — top-63
  global tokens ≈ 0.3% of mass — an uninformative behavioral signal.)
