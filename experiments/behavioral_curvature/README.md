# Behavioral curvature map

**Question.** For a feature that the model represents as a *circle* (the canonical
example is the weekday circle, but any 1-D circular chart works), at which layers
does the network **compute on** the feature, and at which does it merely **carry**
it?

**Instrument.** Fit the feature's circular chart independently at several layers,
then fit the chart-to-chart transports between layers as angle maps
`h_{l→l'}(θ)` with `O(2)` certificates. A transport that is a pure `O(2)` element
`θ ↦ ±θ + φ` (a rotation/reflection — "clock arithmetic") carries the feature
rigidly; departures from `O(2)`, and — crucially — **holonomy** around layer
triangles, are the gauge-invariant signature of computation happening *on the
feature* between those layers.

This is a thin runner over primitives that already live in `crates/gam-sae`:

- `inference/layer_transport.rs` — `fit_transport_map` (penalized-REML angle map
  with winding degree, isometry defect, fold certificate) and
  `composition_defect` (the analytic delta-method composition-law test).
- `inference/transport_class.rs` — `classify_circle_transport_fit`, the
  Fourier-rigidity classifier that returns the `O(2)` element `(winding, phase)`
  and its departure `defect = 1 − max(S₊, S₋)/n`.
- `inference/contracts.rs` — `loop_holonomy` / `invert_o2_edge`, which compose a
  closed loop of `O(2)` transports and return the net element with a
  trivial/nontrivial verdict whose tolerance is **derived** from the loop's own
  composed defect (never a magic constant).

## What the map means

For every layer triangle `(a < b < c)` the runner closes the composition loop

```
a --h_ab--> b --h_bc--> c --h_ac⁻¹--> a
```

and reads its holonomy. By construction (`invert_o2_edge` doc in
`contracts.rs`) this loop's net `O(2)` element measures exactly the failure of the
composition law `h_ac = h_bc ∘ h_ab`. The verdict is null-calibrated: the loop is
declared **trivial** (feature carried, composition holds) only when the net
rotation angle is below `angle_tolerance = Σ edge O(2) defects`, the honest
composed uncertainty of the three transports. A net angle above that tolerance is
**nontrivial** holonomy — real, gauge-invariant evidence that something computes
on the feature within `[a, c]`.

### Per-interval computation score (attribution rule)

Holonomy belongs to a *triangle*, not to a single adjacent interval. To localize
it, each triangle distributes its **excess** holonomy

```
excess(T) = max(0, |net_angle(T)| − angle_tolerance(T))   [radians]
```

uniformly across the adjacent selected-layer intervals its direct edge `(a, c)`
spans. A triangle cannot localize finer than the intervals it covers, so uniform
spreading is the honest, information-preserving choice; summing over all triangles
then concentrates load on the intervals that repeatedly participate in nontrivial
loops. The per-interval `computation_score` (radians) is that summed, null-
calibrated holonomy load. Reported alongside each adjacent interval, as
**complementary local instruments**, are the adjacent transport's own `O(2)`
departure (`local_o2_defect`) and its isometry defect (`local_isometry_defect`).

Complementary per-head view: `attention_kernel.rs`'s stationary circulant fit
describes an attention score by phase difference `t_q − t_k` alone. A head that is
purely circulant implements a rotation/translation on the circle — exactly the
zero-holonomy, "carried" regime this map flags — so a layer whose attention is
well fit by the circulant kernel *and* whose interval shows near-zero
`computation_score` is doubly attested as carrying, not computing.

## Falsifiable predictions

1. **Attention-dominant layers contribute translation-like, near-zero-holonomy
   segments.** Where the residual-stream update on the feature is dominated by
   attention that reads/writes the circle rigidly, the transport is a near-pure
   `O(2)` element and the interval's `computation_score ≈ 0` (`is_trivial` loops).
   Falsified if such intervals carry large holonomy load.
2. **MLP-dominant computation shows up as holonomy.** Where an MLP reshapes the
   circle's metric or mixes harmonics, `local_o2_defect` and the attributed
   `computation_score` rise together and the straddling triangles read
   `is_trivial = false` with a small `composition_defect.p_value`. Falsified if
   layers known (by ablation) to compute on the feature show trivial holonomy.
3. **The map is gauge-invariant.** Re-fitting any layer's chart up to its own
   `Isom(S¹)` gauge (a rotation/reflection) leaves every triangle's holonomy and
   every interval's `computation_score` unchanged (the source gauge cancels in the
   loop; the target gauge is quotiented in `composition_defect`). Falsified if the
   scores move under a pure re-gauge of the input angles.

## Running it

### On the banked smoke fixture

```
cargo run -p gam-sae --example behavioral_curvature_map -- \
  crates/gam-sae/tests/data/qwen3_l11_l17_l23_theta.json \
  /tmp/bcm_qwen3.json
```

The banked file holds one `theta` array per layer (`acts_L11/L17/L23`, 4000
row-aligned tokens). Three layers ⇒ three edges and one triangle — enough to
exercise the full pipeline.

### On the MSI weekday harvests

`experiments/binding_multiplicity/weekday_binding.py` harvests the Qwen3-8B
weekday circle one layer at a time, writing per out-dir:
`weekday_codes.csv` (header `weekday,label,z0,z1,…`), `two_instance_codes.csv`,
`weekday_frame_meta.json` (carries the integer `layer`), and
`weekday_harvest.npz`. Run it once per layer into per-layer subdirectories:

```
for L in 14 18 22 26; do
  python experiments/binding_multiplicity/weekday_binding.py \
    --model /path/to/qwen3-8b --layer $L --harmonics 4 \
    --out-dir harvest/weekday/L$L
done
```

then point the runner at the parent directory:

```
cargo run -p gam-sae --example behavioral_curvature_map -- \
  harvest/weekday /tmp/bcm_weekday.json
```

The runner sniffs the format by filename/shape: it recovers each row's circle
angle from the fundamental harmonic code as `atan2(z1, z0)` and uses the code
amplitude `hypot(z0, z1)` as the per-row on-circle gate (rows whose amplitude is
numerically degenerate in any layer are dropped so the paired-row structure is
preserved). Layer indices come from each subdir's `weekday_frame_meta.json`.

Four layers ⇒ six edges and four triangles, so the layer-14/18/22/26 sweep yields
three adjacent-interval computation scores plus the non-adjacent cross-checks —
the first real-data behavioral curvature map.

## Output

A single JSON report (path printed on stdout as `behavioral_curvature_map_json=…`):

- `layers`, `layer_provenance`, sniffed `input_format`, row counts, and grid;
- `edges` — every pairwise transport: `winding`, `class`, `phase`, `o2_defect`,
  `degree`, `isometry_defect`, `topology_preserved`, `residual_rms`, `transport_edf`,
  `carried_fraction`, and (when gated) `mean_gate`;
- `triangles` — per triple: the `holonomy` block (`net_sign`, `net_angle`,
  `angle_tolerance`, `is_trivial`, `significance`, `excess_holonomy`, `verdict`)
  and the independent analytic `composition_defect` cross-check;
- `intervals` — per adjacent interval: `local_o2_defect`, `local_isometry_defect`,
  `holonomy_load_rad` / `computation_score`, and a `carried`/`computes` verdict;
- a headline `summary` (any nontrivial triangle; the interval of maximal
  computation).

No plotting — JSON only.
