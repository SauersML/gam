# #2502 — unsupervised overcomplete manifold dictionary on Qwen3.5-4B-Base

Campaign scripts (A10 lane, `~/i2502`). The dictionary itself is the Rust
manifold SAE (`gamfit.sae.sae_manifold_fit`, hard-TopK support lane at K > P);
Python here is orchestration, plotting, and the LLM harness only.

Data path: `harvest_qwen35.py` (residual stream after blocks 8/16/22,
wikitext-103, 600k token-rows, fp16 + token/doc/pos metadata) →
`fitprep.py` (pos0 sink peel w/ permutation causality test, 6σ log-norm guard,
train-only PCA chart, affine splice map c0/c1) → `flagship_pipeline.py`
(fit K=32,000 circle atoms + census + top-token interpretation + unsupervised
calendar scan + overcompleteness figures + on-manifold steering deltas +
splice reconstructions; single process because the support-sparse model has no
from_dict) → `gpu_stage_a.py` / `gpu_stage_b.py` (LLM capture / patch+measure,
decoupled from the Rust model lifetime) → `steer_figs.py`, `benchmark_fig.py`.

Baselines: `torch_topk_sae.py` (standard TopK SAE, Gao et al. 2024, matched
K and L0 + PCA ladder). `fit_flagship.py` is the lane prober (pilot arms per
assignment kind). `chain.sh`/`chain2.sh` are the box-side executors that fire
each stage as its precondition lands.

Solver work this campaign forced: #2517 (extensive raw KKT certified against
an absolute tolerance → scale-invariant certificate; rayon-parallel
coordinate sweep; per-cycle telemetry; eps^(1/4) relative inner tolerance).
