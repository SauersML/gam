# Geometric-Wall Closure Results

Real, non-proxy cross-model check. A production LLM residual stream (Gemma-2-2B)
is fit with GAM's actual block-chart promotion lane
(`gamfit.block_sparse_dictionary_fit` + `BlockSparseDictionaryFit.compose_block_charts`,
gamfit 0.1.250), not the earlier numpy-k-means matched-quadratic proxy. Each
layer is residualized against the nuisance-atlas position-0 atom (pos0 peel),
then a FLAT linear block-SAE is compared against the CURVED block-chart lane
**at exactly matched parameter count**, pooled over energy strata.

> **Primary result lives elsewhere.** The headline overcomplete curved-vs-flat
> result is on **Qwen 3.6 (Qwen3-30B-A3B MoE)** — see
> `experiments/real_manifold_sae/results.md`. This file is the Gemma-2-2B
> cross-model corroboration.

## Metric semantics (read this first)

`pooled_flat_floor` / `pooled_curved_floor` are **unexplained residual energy
fractions** (`||x - x_hat||^2 / ||x||^2`). **Lower floor = better fit**, so
`explained variance = 1 - floor`, and
`pooled_drop = pooled_flat_floor - pooled_curved_floor` is **positive only if
curvature helps under this metric**. A negative drop means the linear baseline
reconstructs the ambient residual marginally better. Caveat (see Verdict): a
negative drop here does **not** refute the manifold thesis — this additive-residual
EV comparison is insensitive to it by construction, so it can only ever produce a
tie or a small-sample parameter tax, never a detectable curvature win. Treat the
sign as diagnostic of the *metric*, not of the geometry.

## Verdict

**Flat and curved TIE on reconstruction — and this metric cannot detect a curvature
win even if the thesis is true.** At matched capacity the flat linear baseline
reconstructs the pos0-peeled Gemma residual stream *marginally* better than the curved
block-chart lane on both layers (drop −0.036 / −0.032). This is **not** a null for the
manifold thesis; it is a comparison that is **insensitive to the manifold hypothesis by
construction**, for the same four reasons documented in the Qwen 3.6 primary
(`experiments/real_manifold_sae/results.md`):

1. **Residual-orthogonality trap** — the curved charts fit the LS-orthogonal residual of
   the flat linear tier, from which the tangent and curvature have already been absorbed
   into atom placement; what is left is high-frequency quantization noise a smooth chart
   cannot represent.
2. **Saturation** — the flat dictionary is already near its reconstruction ceiling, so
   little headroom remains for any second tier.
3. **Matched-EV, not matched-bits** — the arms are matched on parameter budget and scored
   in additive-residual EV, the wrong currency for an *informational* thesis (manifold =
   redundancy in the CODES, measured in bits/token).
4. **Curved given half the blocks** — 24 flat blocks vs 12 curved blocks × (4+4 chart
   basis), so curvature is charged twice (params and coverage).

The two layers agree with each other and with the Qwen 3.6 primary, but what they agree
on is that **the additive-residual EV metric is the wrong currency** — not that curvature
is absent. The small flat lead is the same **small-sample parameter tax on a
structure-free residual** seen in the Qwen strata (where the deficit decays monotonically
toward zero with rows and never flips sign). Consistent with the "Geometric Wall is
observational" positioning; the correct test of the thesis is the code-space
rate–distortion pipeline (`experiments/code_space_manifold/`), which measures bits/token.
Note also that the block-chart curved fits predate the barrier/grind fix (81d3900f4), so
the curved arm here may be **underfit**.

| model / layer | pos0 absorbed | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop (flat−curved) | flat EV | curved EV | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Gemma-2-2B L12 | 0.008949 | 2/5 | 29519 | 221184 | 221184 | 0.628195 | 0.664426 | **−0.036231** | 0.371805 | 0.335574 | flat wins; curvature loses |
| Gemma-2-2B L25 | 0.012585 | 4/7 | 29803 | 221184 | 221184 | 0.398981 | 0.430500 | **−0.031519** | 0.601019 | 0.569500 | flat wins; curvature loses |

Matched params per layer: 24 flat blocks × block_size 4 == 12 curved blocks ×
(4 + 4 chart basis) = 96 units × model dimension `d`=2304 → 96·2304 = 221184.
Gemma's position-0 sink is weak (absorbs ~1% of centered energy, unlike the
Qwen sink), so the peel barely changes the target; curvature still loses.

## Provenance

- model: `google/gemma-2-2b` L12 / L25, residuals at `hidden_states[layer+1]`,
  harvested from an architecturally identical ungated mirror (no HF token for
  the gated repo on MSI); see `harvest_out/gemma2_2b_wikitext/manifest.json`.
- corpus: Salesforce/wikitext wikitext-103-raw-v1, `add_special_tokens=False`,
  within-document positions restart at 0 per document; 30000 tokens per layer.
- pos0 nuisance peel via OLS design `[intercept, position0_indicator]`; only
  strata meeting `min_stratum_rows=512` are fitted (L12: 2 of 5, L25: 4 of 7).
- engine: `gamfit.block_sparse_dictionary_fit` +
  `BlockSparseDictionaryFit.compose_block_charts`, gamfit 0.1.250.
- run harness: MSI msismall, 32 cores, `wallclosure_venv` (CPU-only, no torch;
  harvested activations consumed from cache — the driver's cache-hit path
  short-circuits before any torch import).
- raw numbers: `gemma_numbers.json`.

## Model-scope note and superseded proxy

An earlier Qwen-3-8B arm was run and then **removed per a model-scope decision
(8B is out; the primary is Qwen 3.6 / A3B)**; its activation data was deleted, so
no 8B result is retained here. Separately, the Run-2 result was a **numpy k-means
matched-quadratic proxy, not the real fitter** (tangent-PCA plus quadratic
features, not `compose_block_charts`). With the real block-chart compose lane at
matched capacity, curvature does **not** win on reconstruction — the proxy is
superseded for decision purposes.
