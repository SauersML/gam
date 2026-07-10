# Real LLM activation fixtures — OLMo-3-32B

## olmo_mixedlayer_pca64_768.npy — CROSS-LAYER mixture, shape (768, 64) float32

- **Source:** OLMo-3-32B residual-stream activations, real model run, not synthetic.
- **Provenance (the honest story — see #1199):** this is **NOT a single-layer L25
  slice**. The banked corpus `activations.npy` is `635 prompts × 64 layers × 5120`.
  A true single-layer L25 slice (`activations.npy[:, 25, :]`) has exactly **635** rows,
  one per prompt. This fixture has **768** rows, which it cannot have as a single-layer
  slice — it is a **cross-layer mixture**: activations sampled across the flattened
  `(prompt, layer)` axis of the corpus and PCA-reduced to **64 components**. Because the
  PCA basis is fit across layers, the leading axes mostly capture the (large)
  between-layer mean/scale nuisance variation, not within-layer semantic structure.
  This makes it a deliberately ill-conditioned cross-layer mixture, which is why the
  manifold-SAE pins at the `cost=1e12` sentinel on it (#1189).
- **Do not** label this "L25" or call it a faithful single-layer SAE input. The
  `_768` row count is the tell.
- **Spectrum** (explained-variance fraction): PC1=0.253, cum@K8=0.459, cum@K32=0.631,
  cum@K64=0.737 — a long-tailed signal whose leading axes are dominated by cross-layer
  nuisance variation.
- **Use:** real-data e2e *stress* regression for the manifold-SAE on a hard,
  ill-conditioned cross-layer cloud (`tests/test_sae_manifold_olmo_real_recon_ev.py`).

## olmo_l18_qualia_635.npz — the principled SINGLE-LAYER fixture, shape (635, 64)

`X` = (635, 64) float32 — PCA-64 of OLMo-3-32B **layer-18** residual stream (the
best-qualia layer, probe AUC 0.95), **one vector per prompt** (635 rows = 635 prompts,
the single-layer tell). `experiential` = (635,) int8 — 1 if the prompt frames the entity
as having subjective experience ("genuinely feels…"), 0 if not ("feels nothing…").

This is the CORRECT single-layer manifold-SAE input: one semantic layer across prompts.
(The cross-layer `olmo_mixedlayer_pca64_768.npy` mixed across all 64 layers, which is
ill-posed — see #1189/#1199: the SAE pins at cost=1e12 on the cross-layer mixture but
converges within a single layer to a curved Θ≈2π atom with held-out ΔEV≈0.27.)

> A genuine single-layer L25 slice can be regenerated with `tests/sae/extract_olmo_fixture.py`
> (`--layer 25`), which produces a **635-row** `olmo_l25_pca64.npy` — note 635, not 768.

## olmo_l18_pair_pca64_635.npy + olmo_l19_pair_pca64_635.npy — the ROW-ALIGNED crosscoder pair, shape (635, 64) float32 each

The #2231 Inc D manifold-crosscoder fixture: layers **18 and 19** of the same
banked OLMo-3-32B corpus (`activations.npy`, 635 prompts × 64 layers × 5120),
sliced at the SAME 635 prompts so row `i` of both files is the same prompt seen
one layer apart. Each layer is column-centered and PCA-reduced **with its own
per-layer basis** to 64 components (explained variance: L18 = 0.496,
L19 = 0.509) — per-layer bases keep the two ambient charts honestly distinct,
which is exactly what the crosscoder's per-layer decoders must bridge. Equal
widths (64/64) make the cross-layer drift report and per-atom phase-transport
measurements well-defined.

- **Use:** `crates/gam-sae/src/manifold/tests_crosscoder_olmo.rs` — the
  unified-engine crosscoder entry (`run_auto_sae_crosscoder_fit`) on real
  consecutive-layer data with measured drift + transport in the wire report.
- Regenerate from the MSI bank (`olmo_data/base/activations.npy`): center each
  of `acts[:, 18, :]` and `acts[:, 19, :]`, SVD, project onto that layer's top
  64 right singular vectors, save float32.
