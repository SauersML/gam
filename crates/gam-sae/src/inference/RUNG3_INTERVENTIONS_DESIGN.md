# Rung 3 — interventions in the loop (calibration, guarded): DESIGN

Status: design only. Implementation is gated on (a) Rung-2's calibrated
two-block charts (unit-speed on the behavior block, Inc4) and (b) the DOSE
patch harness (model loading + activation splicing). This document pins the
estimator, the data contract, the Goodhart guards, and the Rust surfaces so
implementation is a translation exercise, not a design exercise.

## 1. What Rung 3 is (and is not)

Rungs 1 and 2 put behavior into the *estimator*: Rung 1 prices reconstruction
error in nats through the pulled-back output Fisher
(`MetricProvenance::BehavioralFisher`, `½ eᵀG e` with `G = JᵀFJ`); Rung 2 fits
a behavioral block jointly with the activation block, sharing gates and latent
coordinates, so the chart is oriented and unit-sped by how the output moves.
Both are **local**: they use the Fisher metric *at the data*, i.e. the
second-order Taylor expansion of KL around the clean activation.

Rung 3 closes the loop with **realized** behavior: sample `(token n, atom k,
dose Δt)`, decode the moved point `x'(Δt) = x_n + [g_k(t_nk + Δt) − g_k(t_nk)]
a_nk`, splice `x'` into the model at layer ℓ, run the rest of the network, and
record the realized `KL(p_clean ‖ p_patched)`. The model itself grades the
chart's currency.

Two modes, in order of priority:

* **Calibration mode (the default, and the only mode this design commits to):**
  fit the map from *predicted* nats to *measured* nats and fold the correction
  into the chart's coordinate speed. No gradients flow through the LM; the LM
  is an oracle that is queried, never differentiated. This is where the
  quadratic (Fisher) prediction is honest for small doses and degrades for
  large ones — calibration measures exactly that degradation and re-speeds the
  chart so `Δt = 0.1` means the same realized nats everywhere.
* **Training-gradient mode (explicitly out of scope here):** using realized KL
  as a training signal. Deferred until calibration mode has demonstrated the
  measurement is stable; any future design must re-derive the Goodhart
  analysis of §4 from scratch for the training case.

What Rung 3 is **not**: it is not DAS. DAS searches an unconstrained supervised
rotation for a direction that moves a probe. Rung 3 restricts the
interventional query to **already-fitted, certificate-passing charts** produced
by the unsupervised, identifiable Rung-1/2 fit. The intervention never
*selects* structure (see guard G1); it only calibrates units on, and validates,
structure that earned its place through reconstruction + behavior evidence.

## 2. Predicted nats — the quantity being calibrated

Every intervention carries a prediction made *before* the model is queried,
from objects the fit already owns:

* **Rung-1 prediction (p-space):** with the per-row behavioral metric
  `G_n = U_n U_nᵀ` (the s-probe sketch), a decoded move `Δx` predicts
  `ν̂₁ = ½ Δxᵀ G_n Δx = ½ ‖U_nᵀ Δx‖²`. Computable for any Δx, needs only the
  Rung-1 harvest shard.
* **Rung-2 prediction (chart-space):** with the behavior decoder `Ψ_k C_k` on
  the √p-sphere tangent, a coordinate move Δt predicts
  `ν̂₂ = 2 ‖Ψ_k'(t) C_k Δt‖²` (locally, KL ≈ 2‖Δq‖²). Under the Inc4
  unit-speed gauge this is `ν̂₂ ≈ ‖Δt‖²·(unit speed)` — the "Δt is nats"
  promise being tested.

Both predictions are recorded in every intervention record. Calibration mode
fits measured-vs-predicted for each; the *gap between the two predictions* is
itself a diagnostic (Rung-2's behavior decoder disagreeing with Rung-1's
pullback flags a chart whose behavioral block under-fit).

## 3. The calibration estimator (all existing gam machinery)

Let `ν` = realized KL (measured, nats) and `ν̂` = predicted nats for one
intervention. The calibration model is a GAM — our own machinery, REML-fitted,
nothing new:

```text
log ν_i = β₀ + f(log ν̂_i) + b_{k(i)} + ε_i
```

* `f` — a monotone smooth (existing monotone-smooth machinery), capturing the
  systematic quadratic-approximation decay with dose. Its departure from the
  identity IS the calibration curve.
* `b_k` — a per-atom random effect (an ordinary penalized factor term): atom
  k's log speed error. The fitted `exp(b_k/2)` is the **chart re-speed
  factor** `s_k` folded back into the chart: `t ← s_k · t` (a scalar per
  1-d atom; for d>1, a per-axis version of the same, one random effect per
  axis). This is the ONLY thing calibration writes back (guard G1).
* Family: Gaussian on log-KL to first order; the realized-KL measurement noise
  floor (§5, the Δt=0 controls) enters as a known lower bound on the response
  variance, not as a tuned constant.

REML selects every smoothing/shrinkage level. No grid, no magic constants, no
wall-clock budgets — SPEC-compliant by construction because the estimator IS a
gam fit.

**Why log-log:** the quadratic prediction is exact as dose→0, so the curve
passes through the identity at small ν̂ and bends below it at large ν̂ (the
Fisher over-predicts once the softmax saturates). Log-log makes both the
identity anchoring and the bend low-order.

## 4. The Goodhart guard (the reason for the structure)

"Predicted = measured" has a degenerate global optimum: prefer atoms where
both are zero. A fit allowed to *select* on calibration error will fill the
dictionary with behaviorally inert atoms that calibrate perfectly. Three
structural guards, all load-bearing:

* **G1 — the causal signal selects nothing.** Atom birth/death/gating stays
  entirely Rung-1/2 evidence (reconstruction + behavior block, REML). Rung-3
  writes back exactly one object: the per-atom coordinate re-speed `s_k`
  (a gauge transformation — it changes units, not structure, not membership,
  not decoders, not gates). Enforced by construction: the calibration output
  type carries only `s_k` and diagnostics; there is no API path from realized
  KL to the fit criterion.
* **G2 — the held-out intervention set is never trained on, eval forever.**
  Interventions are split at the *document/question* level (matching the
  harvest split manifest), before any calibration fit. The held-out half never
  enters any fit, ever, across refits — it is the standing measurement of
  "predicted nats mean what they say", reported as held-out calibration error
  (nats, and as a fraction of realized effect).
* **G3 — min-effect floors, estimated not chosen.** An atom enters the
  calibration fit only if its realized effect at the reference dose exceeds
  the measurement floor. The floor is NOT a constant: it is the null
  distribution of the measurement itself, estimated from **Δt = 0 control
  interventions** (splice the *unmoved* decoded point; any nonzero measured
  KL is reconstruction error + numerical noise). The floor is a quantile of
  that null (the same one-sided evidence convention the certificates use).
  Atoms below floor are reported as "unmeasurable at reference dose" — a
  finding (possibly dormant/inert), never silently calibrated.

## 5. Experimental design (what gets sampled)

* **Tokens:** the designed subsample discipline of the two-tier harvest
  (`RowSamplingMeasure::designed_subsample`) reused verbatim — calibration is
  an estimation role, it needs a designed few-thousand rows, not the corpus.
* **Atoms:** all atoms above the G3 floor screen (screening uses a pilot dose
  at each atom, controls included).
* **Doses:** per-atom, at fixed quantiles of that atom's fitted coordinate
  distribution `t_k` (e.g. moves spanning the interquartile range of occupied
  chart territory). Quantile placement is measurement design tied to the data
  distribution, not a hyperparameter search; the dose ladder is logarithmic in
  predicted nats so the calibration curve is identified across scales.
* **Controls:** every batch interleaves Δt = 0 splices (the G3 null) and
  repeat-doses (measurement repeatability).
* **Budget:** expressed in intervention *count* (a few thousand forward
  passes), never wall-clock. This is the "short interventional phase" — the
  e2e lesson that a few percent of total budget spent on interventions buys
  the calibration, and more buys nothing further.

## 6. Data contract (mirrors the harvest shard discipline)

One `.npz` intervention shard, emitted by the Python patch runner (the
model-interaction boundary, DOSE's harness), consumed by the Rust calibration
fit:

```text
row_id        (m,)  int64   — corpus row of the token
atom          (m,)  int64   — atom index k
dose          (m, d) f64    — Δt applied (0 for controls)
nu_hat_1      (m,)  f64     — Rung-1 predicted nats (½‖UᵀΔx‖²)
nu_hat_2      (m,)  f64     — Rung-2 predicted nats (behavior decoder), NaN if no y-block
nu_measured   (m,)  f64     — realized KL(clean‖patched), nats
group         (m,)  int64   — document/question id (the G2 split unit)
is_control    (m,)  bool    — Δt = 0 splice
layer, seed   scalars
```

The patch runner reuses `_capture_activations`' splice hook (`gamfit/torch/
harvest.py`) — patching IS the splice path the downstream harvest already
exercises, with the spliced row now `x + Δ` instead of a probe.

## 7. Rust surfaces (reusable, small)

* `intervention_shard.rs` — load/validate the shard (the `load_harvest_shard`
  discipline: f32/f64 promotion at the boundary, schema asserts, group-level
  split with a persisted, seeded manifest).
* `calibration_fit.rs` — assemble the §3 GAM from a shard's training split and
  fit with the existing engine; output type
  `ChartCalibration { respeed: Vec<f64> /* s_k per atom-axis */,
  curve: MonotoneSmoothSummary, floor_nats: f64,
  heldout_error: CalibrationHeldout }`. No method on it can touch a term's
  gates/decoders — re-speed applies through the same chart-transfer path
  `chart_canonicalization` uses for gauge moves (it IS a gauge move).
* `calibration_certificate.rs` — the G2 held-out report + G3 floor provenance,
  attached beside the existing fit certificates.

## 8. Acceptance tests (planned with the implementation)

1. **Synthetic oracle:** a toy "model" whose true KL under splice is computable
   in closed form; check the calibration recovers a known per-atom speed
   distortion `s_k` and that held-out error shrinks accordingly.
2. **Guard tests:** (G1) the public API provably cannot route realized KL into
   the fit criterion — type-level check plus a test that a calibration run
   leaves gates/decoders/membership bit-identical; (G2) the split manifest is
   stable across refits; (G3) the floor equals the Δt=0 null quantile, and
   below-floor atoms are excluded with the "unmeasurable" tag.
3. **Consistency:** as dose→0, measured/predicted → 1 within the control-null
   band (the Fisher metric is the correct local limit — this doubles as an
   end-to-end validation of the Rung-1 sketch on the real model).
