# Position 0 is 91% of your variance: attention sinks confound representation-geometry measurements

## Thesis

On Qwen3-8B, the dominant mid-layer geometry signal is not semantic geometry. It is the position-0 attention sink.

At layer 18, a single first-token indicator absorbs 91% of total residual-stream variance. At layer 30 it absorbs 77%. At layer 6 it absorbs about 1%. This is large enough to invert the measurement: the raw intrinsic-dimension estimate at L18 is `d_hat = 3.8`, while the corrected semantic geometry is about 26-29 dimensions. The naive result is therefore a 7x error with the wrong sign of the depth trend.

Any energy-weighted representation-geometry statistic that does not remove this sink first is mostly measuring a positional mechanism.

## Evidence

The decisive control is the permuted-position null. The same low-rank positional design is fit after shuffling positions against activations. If the 91% number were merely a near-rank-1 activation bank being easy for any low-rank design to fit, the null would stay large. It collapses to 0.0002.

Measured centered aggregate variance absorbed:

| Layer | position-0 sink | permuted-position null |
| ---: | ---: | ---: |
| Qwen3-8B L18 | 0.9096 | 0.00019 |
| Qwen3-8B L30 | 0.7686 | 0.00018 |
| Qwen3-8B L6 | 0.0067 | 0.00013 |

The sink is therefore causal positional structure, not a low-rank-fitting artifact. The layer contrast is also diagnostic: the pathology is huge exactly where the spectrum collapses, and nearly absent where the layer remains high-dimensional.

## Correction

De-confound before geometry. The attention sink must be peeled before any energy-weighted statistic: intrinsic dimension, participation ratio, spectrum summaries, chart selection, atom birth evidence, persistence summaries, or topology comparisons.

Operationally, treat the sink as typed structure rather than discarded nuisance. Fit a fixed-support finite-anchor atom for known sink supports: position 0 and configured delimiter classes. Subtract that atom, chart the residual semantically, and add the sink atom back for reconstruction and description-length accounting.

This preserves honesty in both directions. Reconstruction still pays for the sink it uses, and semantic geometry is no longer dominated by a positional mechanism that accounts for nearly all observed variance.
