# App C — binding / multiplicity via harmonic super-resolution

The signature capability: a harmonic-circle atom can represent **the same feature
firing `m` times at different values in one context** — two dates, two colors, two
weekdays superposed on one circle. A linear SAE cannot even *represent* this: it
has one coefficient per feature, so two instances collapse into one scalar. The
manifold parameterization stores the superposition `z = Σ_j a_j u(t_j)` of `m`
point masses, and `gam_sae::super_resolution` (matrix-pencil / Prony) un-
superposes it in closed form — recovering the **number** of instances (the
multiplicity) and each one's position and amplitude.

## Parts

**Detection (`binding_multiplicity` Rust example + `weekday_binding.py`).**
- *Part A — planted fixtures.* Random `m∈{1,2,3}`-spike measures on `H∈{8,16}`
  circles across noise levels and f32 quantisation; the recovered spike **count**
  is scored against ground truth into a confusion matrix, with position/amplitude
  error when the count is exact. This certifies the count claim with full ground
  truth. (Runs standalone — no model.)
- *Part B — real-code superposition.* Real single-weekday activation codes
  (from `weekday_binding.py`) summed pairwise into genuine two-instance codes;
  recovery must return 2 spikes at the two weekdays' positions. A controlled
  2-spike fixture built from *real activations*.
- *Part C — fully-real two-instance recovery.* Real two-weekday-prompt
  activations projected onto the fitted frame; recover the measure directly and
  score count + positions against the two named weekdays. The model actually
  superposing two instances, read back out.

**Causal (`gamfit/torch/interventions.py` + `causal_spike_edit_demo.py`).**
A spike-level edit op: remove / move / rescale ONE point mass, re-synthesize the
block code, lift through the atom decoder to a p-space `Δx`, inject via the
existing `run_interventions` KL runner. `causal_spike_edit_demo.py` validates the
mechanism against ground truth without a model; `CAUSAL_DESIGN.md` specifies the
full model experiment (instance-specific ablation of one weekday, the other
untouched — impossible for a linear SAE) and what GPU/model it needs
(Qwen3-8B, already on MSI; forward passes only).

## Files

| file | role |
|---|---|
| `crates/gam-sae/examples/binding_multiplicity.rs` | detection Parts A/B/C (real production `recover_spikes` + gated `recover_measure_from_code`) |
| `weekday_binding.py` | model boundary: harvest Qwen weekday circle, fit `2H`-frame, emit `weekday_codes.csv` + `two_instance_codes.csv` |
| `gamfit/torch/interventions.py` | spike-level edit op (`MeasureSpikeEdit`, `spike_edit_delta_x`, `build_spike_intervention_plan`) |
| `causal_spike_edit_demo.py` | model-free causal mechanism validation / self-test |
| `CAUSAL_DESIGN.md` | fully-specified model causal experiment + GPU/model needs |

## Reproduce (MSI)

```
# Detection Part A (fixtures) — CPU, no model:
cargo run --release -p gam-sae --example binding_multiplicity -- $OUT
# add real data (Parts B/C) after harvesting the weekday frame:
python weekday_binding.py --model .../models/qwen3-8b --out-dir $OUT --layer 20
cargo run --release -p gam-sae --example binding_multiplicity -- $OUT \
    --weekday-codes $OUT/weekday_codes.csv \
    --two-instance-codes $OUT/two_instance_codes.csv
# Causal mechanism (self-test):
python causal_spike_edit_demo.py
```

Results land in `$OUT/binding_multiplicity_report.json` (+ `fixture_spike_counts.json`).
