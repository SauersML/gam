# Letter to the Statistics / Mathematics Review Team

**Re: Manifold sparse-dictionary ("manifold-SAE") — objective correctness, identifiability, and Newton-curvature questions**

You can read code (Rust in `SauersML/gam`: `src/terms/sae_manifold.rs`, `src/solver/arrow_schur.rs`, `src/terms/analytic_penalties.rs`; Python wrapper `gamfit/_sae_manifold.py`) but we are **not** asking you to write or run any — we want **theory**: catch mistakes in the objective/curvature, prove or disprove identifiability claims, and propose better-posed formulations. Where we cite an equation, the corresponding code is pointed out so you can check our implementation matches our math.

---

## 1. What the method is

We decompose a matrix of activations `X ∈ R^{N×D}` (N tokens, D ambient dims) into **K atoms**. Each atom `k` is a low-dimensional **typed manifold**:

- a decoder `B_k ∈ R^{M_k × D}` whose row-space is the atom's ambient subspace,
- a **topology type** `τ_k ∈ {line, circle, sphere, torus, euclidean-patch}`,
- a per-token **latent coordinate** `t_{ik} ∈ R^{d_k}` on that manifold,
- a basis map `Φ_k(t): R^{d_k} → R^{M_k}` (harmonic for circle/torus/sphere, affine/polynomial for euclidean),

so atom k's contribution to token i is `g_k(t_{ik}) = Φ_k(t_{ik}) B_k ∈ R^D`, and the fitted token is `x̂_i = Σ_k a_{ik} g_k(t_{ik})` with gate/assignment weights `a_{ik}` (softmax / IBP-MAP / JumpReLU). This is a **joint** fit (coordinates, gates, decoders are all parameters solved together via a Newton method), **not** EM or a teacher-student scheme.

The headline scientific claim we need to be able to trust: **these typed manifolds are identifiable and reproduce across seeds where ordinary SAEs shatter.** So identifiability and honest uncertainty matter as much as fit quality.

## 2. The objective (this is where we most want your eyes)

We minimize a penalized negative log-likelihood; the inner objective at fixed smoothing parameters ρ is

```
F = ½‖X − X̂‖²                         (data fit, Gaussian)
  + assignment/gate sparsity prior
  + ½ λ_s · decoder smoothness          (Bᵀ S B quadratic)
  + ARD coordinate prior
  + Σ analytic penalties (below)
```

driven by an **arrow–Schur Newton** solve (`run_joint_fit_arrow_schur`), with an **outer REML/Laplace** criterion `ℓ(ρ) = F* + ½ log|H| − occam` selecting ρ (`reml_criterion`). `H` is the joint Hessian (Gauss–Newton / PSD-majorized).

Penalties of theoretical interest:

1. **ARD coordinate prior.** Euclidean axes: Gaussian `½ α t²` (grad `α t`, curvature `α`). **Circle/torus axes:** we replaced the Gaussian (which is discontinuous across the periodic cut and geometrically ill-posed — depends on an arbitrary origin) with a **von Mises energy** `V(t) = (α/κ²)(1 − cos κt)`, `κ = 2π/period`, so `V' = (α/κ) sin κt`, `V'' = α cos κt`. It is C∞ across the cut, `V ≈ ½α t² + O(t⁴)` near 0, and carries the same precision α. **Q1: is this the right periodic analogue of an ARD/Gaussian prior, and is the `−½ n log α` normalizer still correct for it (we keep the Gaussian normalizer)?** Note `V''(t)= α cos κt` is **negative** for |κt| > π/2.

2. **Isometry penalty** (the one causing us the most trouble): for each token,
   `P = ½ μ Σ_n ‖J_nᵀ W_n J_n − G_ref‖²_F`,
   where `J_n[i,a] = ∂g/∂t = Σ_m (∂Φ/∂t)[n,m,a] B[m,i]` is the model Jacobian (D×d), `W_n` a behavioral/identity metric, `G_ref` a reference pullback metric. This pushes the learned coordinate toward an **isometry** (constant-speed / unit pullback metric).

3. **Decoder incoherence** (separability lever, on by default for K≥2): `½ w Σ_{j<k} W_{jk} ‖B_j B_kᵀ‖²_F` over co-activating pairs — pushes cross-atom decoder column-spaces apart.

4. **Nuclear norm** on each `B_k` (embedding-dimension selection), plus SCAD/MCP, block-orthogonality (we believe block-orthogonality targets the *wrong* object — within-atom coordinate axes — and keep it off; please sanity-check that reasoning).

## 3. Where we are, and the concrete failures we need theory on

We have driven the solver from grossly broken to *almost* all-converging. A 9-case test grid (line/circle × d∈{1,2} × isometry∈{0,0.1,1.0}) now passes most cells, but with two persistent issues:

**(A) Isometry curvature makes the Newton/Schur system indefinite.** `P` is a sum of squares of `r(t,B) = JᵀWJ − G_ref`, a function of **both** the coordinate `t` and the decoder `B` (since `J = (∂Φ/∂t)·B`). We assemble the Gauss–Newton majorizer of `P`'s curvature, but we majorize the **coordinate block `H_tt`** and the **decoder block `H_BB`** *independently* and the **cross block `H_tB`** is missing the isometry coupling — so the Schur complement `S = H_BB − H_Bt H_tt⁻¹ H_tB` comes out slightly **non-positive-definite** (we observe Cholesky pivots like `−0.05`), and the Direct log-det in the REML criterion fails. **Q2: For a least-squares penalty `½‖r(t,B)‖²` with `r` bilinear-ish in (t,B), is it ever valid to PSD-majorize the diagonal blocks independently while taking the exact (or zero) cross-block — or must the Gauss–Newton majorizer `AᵀA` (with `A = ∂r/∂(t,B)` the *full* Jacobian) be assembled as one joint block to guarantee a PSD Schur complement? We believe the latter; we want a proof/condition.** (Code: `add_sae_isometry_beta_penalty` and the htt/htbeta assembly in `sae_manifold.rs`; the Schur factor in `arrow_schur.rs`.)

**(B) Two correct-looking fixes conflict.** The von-Mises-ARD fix makes **circle** converge; a separately-landed isometry-cross-block fix makes **euclidean+isometry** converge but **regresses circle**. We have a source where both pass; we suspect the two curvature contributions interact (the clamped `max(V'',0)` periodic curvature plus the isometry GN curvature on the same coordinate block). **Q3: when several penalties contribute curvature to the same coordinate block — some PSD-majorized (ARD `max(α cos κt,0)`), some Gauss–Newton (isometry) — is summing the individually-majorized blocks a valid global majorizer, and does the `max(·,0)` clamp on one term break the descent guarantee or the Laplace approximation?**

**(C) The Laplace/REML criterion uses a majorized, not exact, Hessian.** `ℓ(ρ) = F* + ½ log|H_majorized| − occam`. **Q4: is the model-evidence / REML selection still consistent (or at least sensible) when `log|H|` uses the PSD majorizer rather than the exact Hessian at the optimum? Under what conditions does this bias ρ, and is there a correction?**

**(D) Identifiability / recovery quality.** Two specific worries:
- **Euclidean d=1 gauge freedom.** With an affine basis `Φ(t)=[1,t]`, the map `t ↦ αt+β` can be absorbed by rescaling/shifting the decoder, so the optimum is a non-isolated manifold and the Newton system is rank-deficient (gradient never reaches zero). The Gaussian ARD pins scale but not shift. **Q5: what is the minimal, correct gauge fixing for each topology (we currently consider per-atom centering + unit-variance), and does isometry+ARD already over- or under-determine it?**
- **Convergence ≠ recovery.** `circle d=1 with isometry=0.1` *converges* but recovers the planted manifold at only **R²≈0.47** (vs 0.997 with isometry off). So turning the isometry penalty on *hurts* recovery here. **Q6: is the isometry penalty `½μ‖JᵀWJ − G_ref‖²` actually well-targeted? With a misspecified/identity `G_ref` and `W`, could it be pulling the coordinate off the true isometric parameterization — i.e., is `G_ref` (and the choice to penalize the full pullback metric vs. only its conformal/volume part) the bug?**

## 4. The identifiability tower (please audit)

We claim a three-level identifiability: (i) the **subspace** `row(B_k)` is identified up to the usual sparse-coding permutation/scale, (ii) the **topology type** `τ_k` is selected by REML evidence, (iii) the **coordinate** `t` is identified up to the manifold's isometry group (rotation for circle/sphere, shift+rotation for torus, affine for euclidean) — fixed by the isometry penalty + a gauge choice. We also distinguish **separability** (σ_min of the active-atom tangent matrix) from **specification** (an out-of-class / Level-0 misspecification test) as orthogonal axes. **Q7: is this tower sound? In particular, is "coordinate identified up to isometry group" actually achievable from the objective above, or does it require an explicit quotient/gauge constraint we are missing? And is using σ_min of the stacked active tangents the right separability statistic?**

## 5. What would help most

In priority order: **Q2** (the joint-vs-block Gauss–Newton majorizer — this is our active blocker), **Q6/Q1** (is the isometry penalty / von-Mises prior correctly formulated, since they hurt recovery / could be ill-posed), **Q5/Q7** (gauge & identifiability), then **Q4** (REML validity under majorized Hessian). Concrete counterexamples, a clean PSD-majorizer condition for coupled least-squares blocks, or a corrected isometry/gauge formulation would each directly unblock us.

Code map for reading: objective + assembly `src/terms/sae_manifold.rs` (`run_joint_fit_arrow_schur`, `reml_criterion`, `add_sae_isometry_beta_penalty`, the ARD `ArdAxisPrior`); the penalties' value/grad/curvature `src/terms/analytic_penalties.rs` (`IsometryPenalty`, `DecoderIncoherencePenalty`, nuclear norm); the arrow–Schur Newton + log-det `src/solver/arrow_schur.rs`. Open issue with the empirical failure history: `SauersML/gam#681`.

Thank you — even partial answers to Q2 and Q6 would be high-leverage.
