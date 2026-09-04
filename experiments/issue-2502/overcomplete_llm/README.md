# Overcomplete manifold dictionary on Qwen3.5-4B-Base (issue #2502)

One substrate, one engine, all seven closing criteria off a single set of fits.

**The engine is Rust.** `crates/gam-sae/examples/issue_2502_overcomplete_llm.rs`
does the centring, the alternating block-TopK fit
(`gam_sae::sparse_dict::BlockSparseStreamState`), the held-out transform, the
reconstruction, the explained variance, the usage census and the rate
accounting. The Python here only moves bytes: `transformers` forward passes
(SPEC's explicit PyTorch exception), `matplotlib`, and token lookups.

## Pipeline

| stage | command |
|---|---|
| harvest | `harvest_qwen35.py --layer 16 --train-rows 300000 --eval-rows 100000 --out-dir $A` |
| fit (per arm) | `issue_2502_overcomplete_llm --train $A/train.npy --eval $A/eval.npy --out $F/<arm> --arm <arm> --atoms K --block-size b --block-topk k --gpu required` |
| causal judge | `splice_eval.py --acts-dir $A --arm over=$F/over/eval_recon.f32 ... --out splice.json` |
| interpretation | `interpret_atoms.py --fit-dir $F/over --acts-dir $A --topk k --block-size b --n-blocks G --out interp.json` |
| steering | `steer.py --fit-dir $F/over --acts-dir $A --atoms K --topk k --block-size b --out steer.json` |
| figures | `figures.py --fit-dir $F/over --acts-dir $A --arms-json arms.json ... --out-dir artifacts/issue_2502_overcomplete_llm` |

## The four arms, all from the same engine

At matched **active scalars per token** `m = k·b`:

| arm | `--atoms` | `--block-size` | `--block-topk` | what it is |
|---|---|---|---|---|
| `over` | 8192 | 2 | m/2 | the overcomplete block dictionary, `K/p = 3.2` |
| `crit` | 2560 | 2 | m/2 | critically complete, `K/p = 1.0` |
| `flat` | 8192 | 1 | m | the standard flat TopK dictionary form |
| `pca` | m | m | 1 | with one block the model is `x̂ = γ (x D₁ᵀ) D₁` over a single Stiefel frame, whose minimiser is the top-`m` principal subspace — PCA, from the same loop, with zero selection bits |

## Split contract

Held-out rows come from **articles** disjoint from the training articles:
wikitext-103-raw-v1 rows are grouped into articles by the single-equals heading
(`= Title =`; `= = Section = =` and deeper belong to the article above them), and
each article is assigned by `md5(str(article_index))[:8]/2^32 < eval_doc_frac`.
`meta.json` records `articles_in_both`, which must be 0.
