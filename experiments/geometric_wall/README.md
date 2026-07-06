# Geometric-Wall Closure Rerun

This directory now defines the decisive wall-closure experiment. The old
matched-quadratic proxy is not used.

The rerun has three fixed requirements:

1. Run on the same Qwen and Gemma residual layers as the committed proxy result.
2. Peel the position-0 attention sink first with the nuisance-atlas design
   `[intercept, position0_indicator]`.
3. Compare a linear GAM block-SAE baseline against GAM's real block-chart
   promotion lane at exactly matched parameter count.

The curved lane is `gamfit.block_sparse_dictionary_fit(...).compose_block_charts(...)`.
It is the actual block-chart promotion path, not tangent PCA plus quadratic
features. Parameter matching is strict: the scripts stop unless
`flat_blocks * block_size == curved_blocks * (block_size + chart_basis)`.
With the defaults, `24 * 4 == 12 * (4 + 4)`.

## Qwen

`qwen_wall_closure.py` consumes pre-harvested residual arrays and order-matched
within-document position arrays. Positions are required because the position-0
sink is a known nuisance atom, not an inferred top-PC fallback.

Example shape of the rerun command:

```bash
python experiments/geometric_wall/qwen_wall_closure.py \
  --layer qwen3_8b_L18:/scratch/.../qwen3_8b_wikitext/resid_L18.npy \
  --positions qwen3_8b_L18:/scratch/.../qwen3_8b_wikitext/positions.npy \
  --layer qwen3_8b_L30:/scratch/.../qwen3_8b_wikitext/resid_L30.npy \
  --positions qwen3_8b_L30:/scratch/.../qwen3_8b_wikitext/positions.npy \
  --layer qwen36_35b_L17:/scratch/.../qwen36_35b_wikitext/resid_L17.npy \
  --positions qwen36_35b_L17:/scratch/.../qwen36_35b_wikitext/positions.npy \
  --out-dir /scratch/.../geometric_wall/qwen_real_chart_post_pos0
```

The labels on `--positions` must exactly match the labels on `--layer`.

## Gemma

`gemma_wall_closure.py` harvests Gemma residuals and records `positions.npy`
itself, then runs the same post-position-0-peel real block-chart lane. The
default layers remain the existing Gemma closure layers:

```bash
python experiments/geometric_wall/gemma_wall_closure.py \
  --model google/gemma-2-2b \
  --layers 12,25 \
  --out-dir /scratch/.../geometric_wall/gemma_real_chart_post_pos0
```

## Decisive Prediction

Curved-residual theory predicts that after the position-0 nuisance peel, the
real curved-block lane lowers the held-out reconstruction floor relative to the
matched linear block-SAE floor. The expected scale is a drop proportional to
`kappa^2 * ell^4 / tile`, and the drop should be larger in strata where the
accepted chart-correction energy is larger.

Density-only theory predicts no meaningful floor change after matching parameter
count and removing the position-0 sink. A null result is therefore decisive only
if the post-peel real curved lane is statistically indistinguishable from the
matched linear lane, or worse, across the exact Qwen/Gemma layers.

Positive `pooled_drop = pooled_flat_floor - pooled_curved_floor` confirms the
wall-closure prediction for a layer. Near-zero or negative `pooled_drop` refutes
it for that layer under the real chart lane.
