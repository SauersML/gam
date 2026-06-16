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
