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
substantial overlap is between GQA siblings (a head captures a median 18–20% of
its sibling's operator energy vs about 1% for other heads). Overlap of operator
subspaces is not sameness of routing law: the planted-toy suite scores a head
whose operator is twice another's as fully shared. The negative conclusion
stands, since that error can only overcount sharing. The slow planes
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
  93–99.4% of the output variance, about 20% of units are active per token, and
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
dimension of 3,832,763 out of 596,049,920 parameters: at least 0.643% of
Qwen3-0.6B is pure implementation convention. This is a lower bound: the
census counts only the declared families, and the planted-toy suite (result 7)
shows families it misses, such as a GL(d) hidden in cancelling paired branches
and a cross-head GL between heads with identical attention patterns. Per layer it is almost all the OV basis freedom
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

- A direct search at fixed rank improves the certified bound by only about 20%
  over a twin layer whose unit directions are redrawn at random, and does not
  improve the measured replacement error (37–73% of the output change).
- The certified bound is about 1000× looser than the function: its scale
  `‖A‖ + κΣ‖C_j‖` is 409–1261 while the measured Lipschitz ratio is 0.13–0.94,
  because thousands of unit terms cancel. A usable certificate has to account
  for that cancellation (conditioned on data or on which units are active).
- Merging units with opposite affine forms adds their write vectors, since `ψ` is
  even; only the linear part `A` carries the sign.

- The more general contract, additivity after any invertible linear input change,
  has its finest blocks at the connected components of `Π = UUᵀ` (`W = UT`), and
  for a neuron subset `S` the closest exactly separated reads have whitened error
  `E* = Σ min(λ, 1 − λ)` over the eigenvalues of `UᵀD_SU`, bounded by twice the cut
  `Σ_{i∈S, j∉S} Π_ij²` (`--mode pi`). Π has one component in every layer. Fiedler
  subsets reach 0.27–0.77× the random-subset `E*` at small sizes, but a twin with
  redrawn unit directions reaches about the same (trained/twin 0.95–1.1); only
  pythia-70m layer 3 at 32 units clearly beats it (0.43×). The native uniform
  bound is 50–150× looser than the measured error.

**Correction (trained known-answer toys, `bench/mpd_opfirst_toys_trained_2951.py`):
this null is a limit of the tools, not evidence of absence.** On a task that is
modular by construction, trained exact-GELU students build the split: exactly at
width 16, approximately at width 64 (true-split E*/rank 0.015, or 0.0016 with
weight decay) and width 256 (0.11, 0.029 with weight decay, against 0.26–0.40 for
random subsets). At width ≤ 64 the Fiedler proposal recovers the split (86–98% of
units, 6–10× below the twin). At width 256 it does not: the proposal is no better
than the twin (0.94×) and thresholded Π components show one block, the same
signature as Pythia, on a network whose true split is 3.4× below random. So the
Pythia result says the proposer cannot find approximate splits in overparameterized
blocks; it does not say the splits are absent. A stronger proposer (search directly
on E*) and a null controlling for unit-norm concentration are needed before
concluding either way.

### 6. An executable program for modular addition, from the weights
`bench/mpd_opfirst_modadd_program_2951.py`, receipts `opfirst_modadd_program*.json`.

For the one-layer modular-addition model (p = 113), a program was built from
the weights alone, with nothing fitted to the model's outputs.

- An exact rewrite of every stage in the embedding and unembedding frequency
  planes over all 56 frequencies matches the model to max KL 2e-16 on all
  12,769 inputs.
- The program P keeps the frequencies holding more than 1% of the weight
  power (embedding {20, 56, 7, 5, 10}, unembedding {20, 56, 5, 7}):
  - the embedding as plane coordinates;
  - four heads, each routing by a score driven by one frequency (56, 20, 7, and
    7 + 20), with attention not uniform over the operands;
  - 512 executed ReLU neurons, each reading only the planes holding at least 1%
    of its read energy (the frequency-5 neurons also read the 2×5 harmonic);
  - the unembedding's plane readout.
  It uses 12,884 reals and 985 integers, 16× fewer than the model's 226,688
  parameters. Its Fourier coefficients give a closed form
  `logit(c) = Σ_k A_k cos(ω_k(a + b − c))` with four reals.
- Exhaustive over all inputs: P has 100% argmax agreement, mean KL 8.3e-7 and
  max KL 5.2e-4; the four-real closed form has 100% and max KL 8e-5. The stage
  interfaces are looser: attention weights within 0.061, neuron pre-activations
  6.7% relative (97.5% ReLU sign agreement), readout coordinates 9–11%.
- Held-out counterfactuals, exhaustive (P was not built from any of them):
  - rotating all five embedding planes (the model then answers a + b + s):
    mean KL 8.5e-7, 100% argmax;
  - rotating the four readout frequencies: mean KL 1.9e-4, 100%;
  - scaling one unembedding plane: 99–100%;
  - rotating or swapping a single plane: the edited model moves by mean KL
    7–22 from the unedited one, and P tracks it to mean KL 0.10–0.14 with
    86–90% argmax agreement.

The single-plane edits move the model out of saturation, where the 7–11%
interface error that saturation hides on clean inputs becomes visible. This is
the explanation working as intended and showing its resolution: correct laws,
exhaustively checked where exhaustive checking is possible, with a stated
counterfactual error where it is coarse.

### 7. Known-answer toys: which tools can be trusted
`bench/mpd_opfirst_toys_planted_2951.py`, receipt `opfirst_toys_planted.json`.

Seven hand-built networks with the correct explanation written down before any
tool runs (a paired-branch copy, planted mixed modules, the same with a
cross-module edge of size ε, a rotation block with a repeated angle, two heads
sharing a routing law, a data-subspace ambiguity, a random null). A fibre
oracle (the nullity of the Jacobian of parameters → outputs) confirmed every
hand-derived redundancy.

Correct: planted modules recovered exactly; the cross-edge refuted by the
certified bound (η/ε = 1.0); the repeated-angle planes reported only as a
4-dimensional invariant subspace while a naive eigendecomposition invents wrong
planes; the data ambiguity reported; no structure claimed on the random null.

Confidently wrong, now design rules:
- The unit graph alone reports eight exact modules on the copy block, whose
  nonlinear parts cancel. Sign-duplicate units must be merged first, and a split
  is never reported without the linear-part compatibility check and its
  certified η.
- Declared gauge families undercount redundancy (result 4 is a lower bound).
- Subspace-overlap metrics cannot tell equal routing laws from different ones;
  an operator-equality test is needed.
- Per-head observability closures are not gauge-invariant when heads share
  patterns; the unit is the routing law.

### 8. Shared routing profiles across heads: absent, but native edits are exact
`bench/mpd_opfirst_routing_profiles_2951.py`, receipts `opfirst_routing_profiles_*.json`.

Per head `h` and RoPE frequency `ω`, the complex factors `X = q₁ + iq₂`, `Y = k₁ + ik₂`
give `Z_hω = XY*` and the score `Σ_ω Re(e^{−iωt} xᵀZ_hω y)/√d_h` exactly (checked to 3e-15
against the model's attention). A shared-profile decomposition across heads and
frequencies (commuting diagonalizable ratio operators) would give reusable routing
blocks with exact finite gain and lag edits through the query weights.

- The span of `{Z_hω}` is full in every layer of SmolLM2-135M (288/288) and in Qwen3-0.6B
  layers 0, 9, 18, 27 (1024/1024), so any shared-profile description needs one block
  per head × frequency. The square-truncated class test is rejected, with commutators
  at the level of a Gaussian null.
- The only structural sharing is the GQA key factor. Sibling query factors align more
  than chance (median max|cos| 0.1–0.7 vs 0.03–0.05), and one near-shared cluster exists:
  SmolLM2 layer 18, one GQA group, slow planes 24–29 (wavelengths 35k–214k tokens),
  three siblings within 6–9% of one profile, holding 42% of that layer's QK energy.
- Executed on SmolLM2 layers 3, 15 and 27: a finite gain and lag edit of one head's
  frequency through `W_Q` gives scores equal to `g·Re(e^{−iω(t+δ)}Z)` to 1e-15, with other
  heads unchanged. The sharp softmax bound TV ≤ tanh(w/4) and the payload-diameter
  output bound held on every row, and the latter is attained on two-key rows.
- On Qwen3 the edit is not exact: the per-head q-norm rescales the query by an RMS
  that the edit changes.

### 9. Native-anchored conditional operators: the template is what fails
`bench/mpd_opfirst_cond_operators_2951.py`, receipt `opfirst_cond_operators_OLMo-2-0425-1B.json`.

For a SwiGLU block `F(h) = D[s(Gh) ⊙ Uh]`, a group `c` of hidden units has native output
`y_c = Σ_{n∈c} d_n s(g_n) u_n`, and scaling `D_c` is a real parameter edit. The explanation
tested is `ŷ_c = g_c(z_c) t_c(h)` with native template `t_c = D_c U h` and a gate `g_c` fitted by
REML splines (gam's own `gaussian_reml_fit_batched`). The error splits exactly into operator
inadequacy `E‖r‖²` (the part of `y_c` not along `t_c`), missing conditioning, and gate fit; and
for native gains `ρ`, `E‖F_ρ − F̂_ρ‖² = ρᵀKρ` with `K_cd = E[e_cᵀe_d]`.

On OLMo-2-0425-1B, layers 2, 8 and 14 (2048 train and 2048 held-out tokens):
- Groups formed by clustering unit gates beat size-matched random groups at every depth and
  group count (at 1024 groups, held-out variance explained 0.50/0.42/0.57 vs 0.15/−0.4/−0.1).
- Operator inadequacy dominates everywhere: 0.45–0.49 of group energy at 64 groups and
  0.12–0.15 at 4096, against 0.01–0.06 for conditioning and ≤ 0.01 for gate fit. Even oracle
  per-token gates cap variance explained at 0.79–0.86. A group's output is a scalar times its
  template only if its units' gates are equal, and they are not, even in two-unit clusters.
- Group errors add constructively (`|Σe|²` exceeds `Σ|e_c|²` by 3–105%).
- The late-layer sign-gated law (section 3) is better than any grouping up to 4096 groups
  at layer 14 (0.918 vs 0.78); early layers have no competing law.
- `ρᵀKρ` equals the executed edited-block error to 5.5e-15 relative.

The exact error split did its job: the failure is the operator family, not missing context or
gate flexibility, so more spline freedom would not have helped.

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
