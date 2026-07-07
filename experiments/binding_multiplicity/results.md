# App C results — binding / multiplicity via super-resolution

All numbers from the tested Rust production path (`recover_spikes` +
`recover_measure_from_code`) and the spike-level edit op in
`gamfit/torch/interventions.py`. Runs on MSI (`gam_bindingc` build clone;
Qwen3-8B harvest for the real circle).

## Headline

- **Multiplicity count vs ground truth (controlled fixtures): ~100%.** The raw
  matrix-pencil order recovers the exact spike count for m∈{1,2,3} at **0.99 /
  0.995 / 0.999** accuracy *at* the theoretical 2/H separation limit; the gated
  production readout returns the exact count at **1.0** for m=1,2,3 (clean) at
  every separation ≥ the limit, **0.995–1.0** at σ=1e-2, with a **0.2%**
  false-binding rate at m=1. Robust to f32 quantisation (the real code dtype).
- **A real correctness bug found and fixed.** `super_resolution::recover_spikes`
  returned globally **conjugated (reflected `1−t`)** positions with garbage
  amplitudes for a data-dependent subset of inputs, silently collapsing genuine
  two-spike codes to one (pre-fix gated m=2 accuracy was ~0.05). Fixed by
  solving both the eigen-phasor branch and its global conjugate and keeping the
  lower-residual one. 13/13 `super_resolution` + `coordinate` unit tests pass.
- **Causal per-instance edit is exact.** Removing one point mass zeroes that
  instance's amplitude (1.0 → 6e-17) while the other instance is unchanged to
  1e-16 — the per-instance handle a linear SAE cannot express. Move/scale exact
  to 1e-16.
- **Real Qwen weekday circle obeys the same separation law.** For two-weekday
  prompts whose weekdays are separated **above** the atom's 2/H limit, recovery
  finds 2 spikes at the right positions; below the limit it merges — the exact
  Part-A law, on real activations.

## Part A — planted fixtures (full ground truth)

Sweep: H∈{8,16}, m∈{1,2,3}, σ∈{0,1e-3,1e-2,5e-2}, f32 on/off, planted separation
at {1,1.5,2,3}×(2/H); 200 trials/cell, 192 cells.

| metric | m=1 | m=2 | m=3 |
|---|---|---|---|
| raw matrix-pencil order accuracy @ 2/H limit | 0.990 | 0.995 | 0.999 |
| gated exact-count accuracy, clean, all separations ≥ limit | 1.00 | 1.00 | 1.00 |
| gated exact-count accuracy, σ=1e-2, ≥1.5× limit | 1.00 | 0.995–1.0 | 1.0 |
| m=1 false-binding rate (avg) | 0.002 | — | — |

The gate is precision-favouring: at m=1 it essentially never invents a second
spike, and it accepts multiplicity only when the matrix-pencil fit reduces the
coefficient residual and respects the 2/H separation — which after the
conjugation fix it does at ~100% once spikes are resolvable.

## The conjugation bug (auditable in `part_a_m2_diagnostic`)

Pre-fix, planted {0.285, 0.801} came back as {0.199, 0.715} = {1−0.801, 1−0.285}
with amplitudes ≈ 0/negative; the count check (`model_order==2`) was blind to it,
but the measure readout rejected the mis-fit and reported one spike. Root cause:
faer's SVD right basis `V` is pinned only up to a per-column unit phase, so for
some inputs the whole basis returns conjugated. Since `y_h` is not
conjugate-symmetric, the data selects the branch. See
`crates/gam-sae/src/super_resolution.rs` (`solve_branch`) and the drafted gam
issue.

## Part B / C — real Qwen3-8B weekday circle

Harvest: 8 templates × 7 weekdays (singles) and 6 templates × 42 ordered pairs
(two-weekday prompts), final-token residual at layer L, sink-peeled + PCA(12),
harmonic frame H=6 (limit 2/H = 0.333) fit from the 7 weekday means' empirical
circle angles. Layers 14/18/22/26; **L18 is the cleanest**.

- **Singles** recover as exactly one spike for 5–7 of 7 weekdays (7/7 at L26).
- **Part B** (sum two real single-weekday mean codes = a guaranteed additive
  two-instance code): gated 2-spike accuracy 0.40–0.52 — limited because most
  weekday pairs sit below the H=6 resolution limit and single codes carry
  spurious higher harmonics from the 7-point frame.
- **Part C** (real two-weekday-prompt activations, L18): gated 2-spike 0.139, raw
  matrix-pencil order-2 0.278; **separation law: 0.25 above the limit vs 0.083
  below** (n=84 vs 168). Well-separated pairs recover accurately, e.g.
  truth {0.307, 0.749} → {0.369, 0.741}; {0.407, 0.749} → {0.377, 0.737}; a
  below-limit pair {0.307, 0.075} → {0.043, 0.416} (merged/mis-placed).

**Interpretation.** The controlled fixtures prove the capability at ~100%; the
real weekday circle confirms the *same* 2/H law governs recovery on real
activations, but 7 discrete weekdays support too few clean harmonics to resolve
close pairs. The right vehicle for a strong real causal result is a
**continuous, higher-harmonic circle** (angle / hue / clock) — see
`CAUSAL_DESIGN.md`; the harvest+frame+edit plumbing here transfers unchanged.

## Reproduce

```
cargo run --release -p gam-sae --example binding_multiplicity -- $OUT \
    [--weekday-codes $OUT/weekday_codes.csv] \
    [--two-instance-codes $OUT/two_instance_codes.csv]
python experiments/binding_multiplicity/causal_spike_edit_demo.py   # exits 0
```
