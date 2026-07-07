# Geometric-Wall Closure Results

Real, non-proxy verdict. Production LLM residual streams are fit with GAM's
actual block-chart promotion lane (`gamfit.block_sparse_dictionary_fit` +
`BlockSparseDictionaryFit.compose_block_charts`, gamfit 0.1.250), not the earlier
numpy-k-means matched-quadratic proxy. Each layer is residualized against the
nuisance-atlas position-0 atom (pos0 peel), then a FLAT linear block-SAE is
compared against the CURVED block-chart lane **at exactly matched parameter
count**, pooled over energy strata.

## Metric semantics (read this first)

`pooled_flat_floor` / `pooled_curved_floor` are **unexplained residual energy
fractions** (`||x - x_hat||^2 / ||x||^2`). **Lower floor = better fit**, so
`explained variance = 1 - floor`, and
`pooled_drop = pooled_flat_floor - pooled_curved_floor` is **positive only if
curvature helps**. A negative drop means the linear baseline wins. This matches
the README's decisive criterion exactly: positive drop confirms wall-closure,
negative drop refutes it.

## Verdict

**Curvature does NOT close the wall on reconstruction.** At matched capacity the
flat linear baseline reconstructs the pos0-peeled residual stream *better* than
the curved block-chart lane. This is a clean NULL for the
curvature-helps-reconstruction hypothesis, consistent with Run-1's ceiling
(curved sat below the linear envelope) and the "Geometric Wall is observational"
positioning.

| model / layer | pos0 absorbed | fitted strata | rows | flat params | curved params | flat floor | curved floor | drop (flat−curved) | flat EV | curved EV | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Qwen3-8B L18 | 0.9276 | 2/15 | 29728 | 393216 | 393216 | 0.741080 | 0.788604 | **−0.047524** | 0.258920 | 0.211396 | flat wins; curvature loses |
| Gemma-2-2B L12 | _pending job 12709803_ | | | 221184 | 221184 | | | | | | |
| Gemma-2-2B L25 | _pending job 12709803_ | | | 221184 | 221184 | | | | | | |

Matched params per layer: 24 flat blocks × block_size 4 == 12 curved blocks ×
(4 + 4 chart basis) = 96 units × model dimension `d` (Qwen `d`=4096 →
96·4096=393216; Gemma `d`=2304 → 96·2304=221184).

### Qwen3-8B L18 detail (per stratum)

| stratum | rows | flat floor | curved floor | drop | flat EV | curved base EV | accepted charts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28701 | 0.75467 | 0.79918 | −0.04451 | 0.2435 | 0.2077 | 12 |
| 1 | 1027 | 0.55666 | 0.64509 | −0.08844 | 0.3909 | 0.3154 | 12 |

The curved lane's accepted chart corrections are real (12 accepted per stratum),
but even with them the fewer-blocks-plus-charts allocation loses to
more-flat-blocks at matched capacity in every stratum. Charts slightly *helped*
the curved base in the small high-energy stratum (+0.04 EV) and slightly *hurt*
in the large low-energy stratum (−0.007 EV); neither closes the gap to flat.

## Provenance

- models: `Qwen/Qwen3-8B` L18 and `google/gemma-2-2b` L12/L25, residuals taken
  at `hidden_states[layer+1]`. Gemma was harvested from an architecturally
  identical ungated mirror (no HF token for the gated repo on MSI); see
  `harvest_out/gemma2_2b_wikitext/manifest.json`.
- corpus: Salesforce/wikitext wikitext-103-raw-v1, `add_special_tokens=False`,
  within-document positions restart at 0 per document.
- tokens per layer: 30000; pos0 nuisance peel via OLS design
  `[intercept, position0_indicator]` (Qwen L18 pos0 absorbed 92.76% of centered
  energy). Only strata meeting `min_stratum_rows=512` are fitted (Qwen: 2 of 15).
- engine: `gamfit.block_sparse_dictionary_fit` +
  `BlockSparseDictionaryFit.compose_block_charts`, gamfit 0.1.250.
- run harness: MSI msismall, 32 cores, `wallclosure_venv` (CPU-only, no torch;
  harvested activations consumed from cache).
- raw numbers: `qwen_numbers.json` (and `gemma_numbers.json` once it lands).

## This supersedes the Run-2 matched-quadratic proxy

The earlier Run-2 result was a **numpy k-means matched-quadratic proxy, not the
real fitter** — tangent-PCA plus quadratic features, not GAM's
`compose_block_charts` lane. With the real block-chart compose lane at matched
capacity, the sign does not go curvature's way: curvature does **not** win on
reconstruction. The proxy is superseded for decision purposes.
