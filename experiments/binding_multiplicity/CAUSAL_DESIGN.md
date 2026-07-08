# App C causal half — spike-level intervention design

The detection half proves the harmonic-circle atom **represents** multiplicity:
super-resolution recovers the number and positions of the point masses a block
code superposes. The causal half proves that representation is **load-bearing**:
editing ONE point mass moves the model's downstream behaviour *for that instance
only*, with the other instance of the same feature untouched. No linear SAE can
even pose this experiment — it has a single coordinate for the feature, so "the
Monday instance" and "the Thursday instance" of one weekday feature are the same
knob.

## What is already built (mechanism, model-free)

`gamfit/torch/interventions.py` now carries the spike-level edit op:

- `MeasureSpikeEdit(kind=remove|move|scale, spike_index, new_t, factor)`
- `synthesize_measure_code(spikes, H)` — forward map `z = Σ_j a_j u(t_j)`
- `spike_edit_code_delta(spikes, edit, H)` — `Δz = synth(edited) − synth(orig)`
- `spike_edit_delta_x(decoder_2HxP, spikes, edit)` — lift to `Δx = Δz · D`
- `build_spike_intervention_plan(...)` — assemble an `InterventionPlan` that
  runs straight through the existing `run_interventions` KL runner.

`causal_spike_edit_demo.py` validates the mechanism end to end against ground
truth (remove/move/scale all recover exactly the intended edited code to machine
precision; the matched-filter readout for the removed instance collapses while
the survivor's is unchanged). This is the "synthetic patched-reconstruction demo"
— it certifies the algebra of the edit independent of any model.

## The model experiment (needs a GPU + the frozen model)

**Model / GPU.** Qwen3-8B lives under the cluster models dir (point the harness
at it via a `--model-dir`/`$HF_HOME` of your choosing; HF cache key
`models--Qwen--Qwen3-8B`). A single A100/A40 (or even CPU at this batch size) is
enough — the experiment is forward passes only, no training, no gradients. This
is the SAME harvest+hook path `experiments/real_circle/real_qwen_circle.py`
already exercises.

**Circle atom.** Use the weekday circle at layer `L` (`weekday_binding.py`
fits the `2H`-frame `D` in the sink-peeled PCA basis `B` and reports its
`harmonic_frame_r2`; require `r2 ≳ 0.9` for a clean atom). The decoder to inject
through is `D` (2H×p_frame); the basis map to the residual stream is `B`
(p_frame×d_model), so `spike_edit_delta_x(D, ...) @ B` is the `d_model` delta,
exactly the `basis` argument of `build_spike_intervention_plan`.

**Battery.** Two-weekday prompts `"... {a} and {b}"` (the `PAIR_TEMPLATES`), each
with a single-token continuation probe that reads each weekday (the restricted
weekday-logit readout from `real_qwen_circle.py`). Ground-truth positions are the
two weekdays' empirical circle coordinates `t_a, t_b`.

**Procedure (per prompt).**
1. Harvest the final-token residual `x` at layer `L`; project to the frame code
   `z = enc·(B·x − centre)`; recover the measure with the Rust readout
   (`recover_measure_from_code`) — confirm 2 spikes at `≈ t_a, t_b`.
2. Build `remove(spike@t_a)` → `Δx_model = spike_edit_delta_x(D, measure, edit)·B`.
3. Splice `x + Δx_model` at layer `L−1`'s output (the `real_qwen_circle` hook),
   rerun, and read the restricted weekday distribution.

**Predictions (the causal claim).**
- **Instance specificity.** Removing the spike at `t_a` drops the model's
  probability/logit mass on weekday `a` toward its no-mention baseline, while
  weekday `b`'s mass is unchanged within measurement noise. Symmetrically for
  removing `t_b`. A linear-SAE ablation of "the weekday feature" moves BOTH.
- **Move = retune.** `move(t_a → t_c)` shifts mass from weekday `a` to weekday
  `c`, leaving `b` fixed — a per-instance rewrite.
- **Dose monotonicity.** `scale(t_a, factor∈{0,¼,½,¾,1})` traces a monotone
  path in weekday-`a` mass from baseline to intact, with `b` flat — a graded
  per-instance handle.
- **Control (G3 null).** `factor=1` (identity edit) yields `Δx = 0` and exactly
  zero KL through the same-path splice; any nonzero is the measurement floor.

**Readouts recorded** (reuse `run_interventions` + the `real_qwen_circle`
restricted-KL/logit machinery): realized full-vocab and restricted-weekday KL,
Δ log-prob of weekday `a` vs `b`, and the instance-specificity ratio
`|Δlogit_a| / (|Δlogit_b| + ε)` (the headline number; ≫ 1 is the signature).

**Baseline contrast (the impossibility).** The same battery through a matched
linear SAE (one weekday latent): its only edit is "scale the weekday feature",
which cannot distinguish the two instances — `|Δlogit_a| / |Δlogit_b| ≈ 1` by
construction. Report the two ratios side by side.

## Runner

`run_weekday_causal.py` (to add when the frame passes the `r2` gate) wires
`weekday_binding.py`'s `D`, `B`, `centre`, and per-prompt measures into
`build_spike_intervention_plan(..., basis=B)` and `run_interventions`, writing an
`InterventionShardData` `.npz` plus the instance-specificity summary. It is a
thin composition of code that already exists; the only model-touching surface is
the harvest+hook already shipped in `real_qwen_circle.py`.
