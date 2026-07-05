# Nuisance-atlas on Qwen3-8B wikitext residuals — the L18 massive channel is the position-0 sink

Positional-only nuisance regress-out (`qwen_nuisance_msi.py`, run on MSI via
`qwen_nuisance.sbatch`, job 12581201) on the Qwen3-8B wikitext harvest
`/projects/standard/hsiehph/sauer354/harvest_out/qwen3_8b_wikitext`
(`[300000, 4096]` fp32 per layer, seq_len 512). Per-token within-doc position was
reconstructed by re-tokenizing the identical corpus (300000 tokens, aligned
row-for-row with the harvest's doc-then-position order).

Each cell is the centred aggregate R² a positional design absorbs from the full
activation bank (fraction of total variance explained). `pos0` is a single
first-token indicator; `fourier` is a 16-harmonic normalized-position Fourier
series; `combined` is fourier + 8 early-position indicators (41 columns total);
`null` permutes the positions against the activations.

| Layer | PC1 var (ev_top1) | participation ratio | fourier | **pos0** | early | combined | **permuted null** |
|------:|------:|------:|------:|------:|------:|------:|------:|
| **L18** | 0.9909 | 1.02 | 0.2721 | **0.9096** | 0.9098 | 0.9099 | 0.00019 |
| **L30** | 0.8466 | 1.40 | 0.2336 | 0.7686 | 0.7692 | 0.7698 | 0.00018 |
| **L6**  | 0.0648 | 139.8 | 0.0090 | 0.0067 | 0.0107 | 0.0138 | 0.00013 |

## Finding

**The massive-activation channel that dominates Qwen3-8B L18 is the position-0
attention sink.** A *single* first-token indicator absorbs **91% of L18's total
variance**; the 16-harmonic Fourier basis and positions 1–7 add essentially
nothing beyond it (`combined` 0.9099 vs `pos0` 0.9096). This directly explains
the spectrometer confound — L18 has 99.1% of its variance in one PC (participation
ratio ≈ 1.02), and that one PC *is* the first-token sink, so the intrinsic-
dimension estimate was measuring a positional artifact, not semantic geometry.

**The permuted-position null is decisive.** Shuffling positions against the
activations drops the combined absorbed fraction from 0.91 to **0.0002** (the
`M/N` overfit floor for a 41-column design). So the 0.91 is genuinely positional
and causal — not an artifact of a low-rank design fitting a near-rank-1 matrix
(the obvious failure mode when PC1 already holds 99% of the variance).

**Layer contrast.** The effect tracks the rank pathology monotonically: L30
(PC1 = 85%, PR 1.4) is the same story one notch weaker (pos0 absorbs 0.77), while
the healthy high-dimensional L6 (PC1 = 6.5%, PR 140) absorbs only 1.4% — position
is not a meaningful nuisance there, and its variance is genuinely distributed.
This is the control that rules out the method trivially absorbing variance.

## Takeaway for charting

On L18/L30 the nuisance atlas must run before any coordinate/dictionary charting,
or ~91%/77% of the "signal" a chart sees is the attention sink. On L6 the pre-pass
is a near-no-op, as it should be. The Fourier block absorbing 0.23–0.27 on L18/L30
(vs 0.91 for pos0) shows there is also mild smooth positional structure, but the
first-token sink is the overwhelming component.

## Reproduce

```
# login node (network): reconstruct + save positions
python qwen_nuisance_msi.py --harvest <harvest_dir> --model Qwen/Qwen3-8B \
    --layers 18 --save-positions <dir>/qwen_positions.npy
# compute node (no network): the passes
sbatch qwen_nuisance.sbatch     # uses --positions <dir>/qwen_positions.npy
```
