# Real LLM activation fixture — OLMo-3-32B

`olmo_l25_pca64_768.npy` — float32, shape (768, 64).

- **Source:** OLMo-3-32B residual-stream activations (layer L25), real model run, not synthetic.
- **Construction:** 768 token-position activations sampled (seed 0) from the banked
  635×64×5120 corpus, mean-centered, projected onto the top **64 PCA components**
  (SVD of the centered subsample). The manifold-SAE pipeline PCA-reduces its input,
  so this is a faithful, low-dim slice of what the SAE actually consumes.
- **Spectrum** (explained-variance fraction): PC1=0.253, cum@K8=0.459, cum@K32=0.631,
  cum@K64=0.737 — a structured-but-long-tailed signal (structured minority + unstructured bulk).
- **Use:** real-data e2e regression test for the manifold-SAE (`tests/...`). Held-out
  reconstruction EV / structure recovery on genuine LLM activations.

## olmo_l18_qualia_635.npz (the principled single-layer fixture)

`X` = (635, 64) float32 — PCA-64 of OLMo-3-32B **layer-18** residual stream (the best-qualia layer, probe AUC 0.95), one vector per prompt. `experiential` = (635,) int8 — 1 if the prompt frames the entity as having subjective experience ("genuinely feels…"), 0 if not ("feels nothing…").

This is the CORRECT manifold-SAE input: a single semantic layer across prompts. (The earlier `olmo_l25_pca64_768.npy` mixed all 64 layers, which is ill-posed — see #1189: the SAE pins at cost=1e12 cross-layer but converges within-layer to a curved Θ≈2π atom with held-out ΔEV≈0.27.)
