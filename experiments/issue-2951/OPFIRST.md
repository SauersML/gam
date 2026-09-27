# Operation-first probes for MPD (#2951)

These probes ask what an explanation of a trained transformer can be built from
if the unit is a computational law with a parameter implementation, rather than
a set of independently removable parameter pieces. Each one is exact linear
algebra or an executed replacement on real weights, with no masks, no fitted
dictionaries and no adversary. Every number below comes from a receipt in
`receipts/` produced by the named script; "exact" means an algebraic identity or
a rank under a derived rounding band, "numerical" means a thresholded or
energy-weighted count, and "empirical" means measured on 512 fineweb-edu tokens
(position 0 excluded unless stated).

Models: Qwen3-0.6B-Base and Qwen3-1.7B-Base (SwiGLU, RMSNorm, GQA, rotary with
q/k norm), EleutherAI Pythia (exact GELU), and a one-layer modular-addition
transformer (p = 113).

## Results

### 1. Query-key operators are not shared across heads
`bench/mpd_opfirst_rope_span_2951.py`, receipts `opfirst_rope_span_*`.

Each head's score operator splits exactly into rank-2 per-plane pieces,
`M_h(Δ) = Σ_j cos(ω_jΔ) A_hj + sin(ω_jΔ) B_hj` (checked against HF scoring to
6e-16). Across a layer the 2048 pieces span all 2048 dimensions in every layer
of both models (numerically certified). The energy spectrum is concentrated, but
a null in which every head is its own orthogonal subspace concentrates it the
same way, so the concentration is uneven sizes, not sharing. The only
substantial sharing is between GQA siblings (a head captures a median 18–20% of
its sibling's operator energy vs about 1% for other heads). The slow planes
(wavelength beyond the 32k context) carry about 70% of the operator energy:
most of the QK operator is position-independent content matching.

The natural attention unit is a GQA group; a cross-head operator dictionary is
not there to find.

### 2. Exact observability is trivial; weighted observability finds mechanisms
`bench/mpd_opfirst_observability_2951.py`, `bench/mpd_opfirst_task_observability_2951.py`.

With the full unembedding as readout, the observable residual subspace is all of
d_model at every layer of Qwen3-0.6B (the 151,936 × 1024 unembedding is exactly
full rank). With a task readout the exact closure still saturates within one to
three layers (modular addition: 128 after one OV step; Qwen weekday task: 1024
after two layers).

The energy-weighted closure is informative. For modular addition the top eight
observable directions before attention are the embedding's key-frequency planes
(principal cosines 0.9999–0.996, 96% of the Gramian energy): the mechanism's
input subspace is recovered from the weights and the task readout alone. For the
weekday task only a soft concentration survives (participation ratio about
20–150 over the last one to three layers).

Exact rank questions are generically trivial on trained weights. What carries
information is a declared weighting: which readout, which task.

### 3. SwiGLU MLPs: a sign-gated law late, smooth nonlinearity early
`bench/mpd_opfirst_mlp_oddeven_2951.py` (modes `oddeven`, `relusplit`, `replace`, `single`).

- The exact odd/even split `silu(g) = g/2 + ψ(g)` does not simplify: the
  bilinear and ψ parts are each two to three times the MLP output and cancel
  (cos ≈ −0.9), because most gates satisfy |g| ≫ 1.
- The exact split `silu(g) = relu(g) + e(g)`, with `e` even and |e| ≤ 0.2785,
  does: from layer 21 the sign-gated bilinear part `W_d[relu(g)⊙u]` explains
  93–99.7% of the output variance, about 20% of units are active per token, and
  90% of the gated energy sits in 2–4% of all units.
- Executed replacement (silu → relu, whole model, next-token KL): the last
  layer alone costs mean KL 0.007 (0.6B) / 0.002 (1.7B); layers 21–27 cost
  0.06 / 0.015; any set containing the early block collapses the model.
  One layer at a time, layers 0–10 cost 0.03–0.23, and 0.6B layer 1 alone costs
  2.5. The damage comes from ordinary tokens: restricting the swap to the
  position-0 massive-activation tokens changes KL by at most 1.5e-8.
- No certified bound proves the late-layer claim in advance: the tightest
  per-token bound on the dropped correction is about 4.5× loose, unit-wise
  bounds are 20–50× loose because the correction cancels, and parameter-only
  bounds are vacuous.

Late MLPs are conditionally sparse sign-gated bilinear laws; early MLPs do real
smooth-nonlinear work. Primitives have to depend on depth.

### 4. Implementation gauge is a small fraction of the parameters
`crates/gam-sae/examples/mpd_gauge_census_2951.rs`, receipt `gauge_census_Qwen3-0.6B-Base.json`.

The gauge detectors in `parameter_decomposition::gauge` certify an exact orbit
dimension of 3,832,763 out of 596,049,920 parameters: 0.643% of Qwen3-0.6B is
pure implementation convention. Per layer it is almost all the OV basis freedom
(131,072 of 136,256), with SwiGLU up/down scales (3,072), norm gains (2,048) and
only 64 in rotary QK (q/k norm leaves little freedom). The tied embedding pins
the residual-stream rotation down to an orbit of 17,579.

Continuous parameter symmetry is not where decomposition ambiguity comes from in
this model; claims about OV still have to treat V and O jointly.

### 5. Plain GELU MLPs and exact module splits
`bench/mpd_opfirst_gelu_modules_2951.py`, receipts `opfirst_gelu_modules_pythia-{70m,160m}.json`.

For an exact-GELU MLP the normal form `F(x) = Ax + b + Σ_j u_j ψ(a_jᵀx + β_j)`
with `ψ(t) = t(Φ(t) − ½)` holds exactly (checked to 4e-15). Since the functions
`{1, ψ′(a_jᵀx + β_j)}` are linearly independent, a replacement-independent split
`F(Px + (I − P)y) = QF(x) + (I − Q)F(y)` exists iff `QC_j = C_jP` for every
coefficient, and for the rank-one `C_j = u_j a_jᵀ` this forces each read and
write to be an eigenvector with a shared eigenvalue: exact splits are the
connected components of the graph joining units with non-orthogonal reads or
writes.

- Planted control: two modules (5- and 7-dimensional) mixed by a dense
  orthogonal change of basis are recovered exactly (certified η ≈ 7e-14; random
  splits of the same sizes have η ≈ 41).
- Trained Pythia: the exact graph is complete (one component, every one of the
  2,096,128 possible edges present). Under thresholds a giant component survives
  to |cos| ≥ 0.15. The best spectral split has a certified defect η equal to
  1.04× that of a random split of the same sizes, and an empirical replacement
  error of about 26% of the output change.

Trained GELU MLP blocks contain no parallel modules in the replacement sense.
Exact module splitting is a diagnostic, not a target.

## What this says about method

- Deletion-based sparsity and exact rank are the wrong primary objects: exact
  statements about trained weights are generically full rank or trivially
  saturated (results 1, 2, 3), as the analytic-rigidity argument predicts.
- Useful structure appears under a declared weighting or a declared replacement
  contract: a task readout (result 2), a depth range where an executed
  replacement stays within a stated KL (result 3).
- Where structure is real it has an exact algebraic form that is not rank one:
  GQA groups, rotation planes, sign-gated bilinear laws.
- Certificates are the gap. The executed replacements are empirical; the
  available bounds are loose or vacuous. Exact finite-change operators for the
  primitives (`parameter_decomposition::secant`) are the first piece of closing it.

## Library pieces that landed with these probes

- `parameter_decomposition::secant`: exact two-endpoint change operators with
  derived error bands. Softmax, `q − p = ℓ⊙δ − ℓ(ℓᵀδ)/s` with logarithmic means
  `ℓ`, applied in O(n); RMSNorm, a scalar times the identity plus a rank-one
  term; the bilinear midpoint rule `W′h′ − Wh = W̄Δh + ΔW h̄`; and SiLU and
  exact-GELU divided differences. On the saturated softmax `(12, −12) → (−12, 12)`
  the local Jacobian predicts a change of norm 2.6e-9 and the secant gives the
  true 1.414. These replace local-gradient reasoning wherever a finite change is
  the question.
- `parameter_decomposition::gauge` and `operators` (restored): the exact
  implementation-gauge detectors behind result 4, and the mask-gauge,
  straight-versus-angle path and commutator facts.
- `parameter_decomposition::state` (kept): quotient and realization contracts
  and the exact linear observability closure behind result 2.
