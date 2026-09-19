# Joint REML over smoothing parameters and the Matérn range: identifiability, charts, tail laws and certificates

**Lane:** `matern-identifiability`, convergence theory team. **Scope:** the joint outer problem over θ = (ρ, ψ), where ρ_m = log λ_m are the smoothing parameters of the four Matérn operator penalties and ψ = log κ is the log inverse range. This is the problem behind the failing tests `mgcv_matern_smooth`, `sklearn_gp_matern_regression`, `mega_batch_k::matern_with_explicit_centers` and `statsmodels_gam_additive`. The report also covers `inla_tensor_product_spde`: it is not a Matérn term, but it fails in the same outer-covariance code.

**Code references** are to the read-only checkout `SP/main_src` at commit `0b0d0120c2`, where `SP` is the session scratchpad. The log is `SP/q1561/all-tests.log`. The scripts are in `SP/theory/matern-identifiability/`, and each writes a matching `.out` file.

The replica (`gamfit_replica.py`) reproduces gamfit's basis, operator penalties, Frobenius normalization and profiled REML. Its data and RNG are **not** those of the tests, so it matches the mechanisms, not the log's numbers.

**Status tags:**
- [P] proven here;
- [N] checked numerically;
- [R] taken from the literature;
- [C] conjecture.

---

## 1. Summary

1. **The raw chart builds in the ill-conditioning; the ρ̃ chart removes that part exactly [P, N].**
   - gamfit normalizes each operator penalty by a ψ-dependent Frobenius norm: S_m = S̃_m/c_m(ψ). So V(ρ, ψ) = Ṽ(ρ − ℓ(ψ), ψ), where ℓ = log c (Theorem 1).
   - The raw Hessian is the sheared matrix Jᵀ∇²Ṽ J, with J = [[I, −ℓ′], [0, 1]], plus a curvature term that vanishes at stationarity.
   - In the replica at the optimum: cond(raw) = 5.3·10³ and cond(ρ̃) = 274, a factor of 19. The identity reproduces the raw Hessian from the ρ̃ Hessian to four digits.
   - The ρ̃ chart can be had exactly by freezing the normalization at the seed.
2. **The joint minimum is isolated and ill-conditioned, not flat [N, P].**
   - The soft eigenvalue is 0.17 in ρ̃ (0.040 raw). The Weyl bar ε_H is about 10⁻⁷, so the minimum is Morse.
   - The Newton–Kantorovich certificate of the certificate-theory lane applies without modification.
   - Nothing should be deflated. The existing `with_criterion_invariance` publication, which has a zero ψ component, is correct.
3. **The ridge comes from the Matérn model and fades as κ grows [R, N].**
   - For a GP with a nugget, Zhang's microergodic direction (dρ, dψ) ∝ (2ν, 1) is within 2–4° of the soft eigenvector.
   - The condition number grows with n, with a preasymptotic slope of 0.29–0.51. The asymptotic slope is d/(2ν + d) = 1/6 (Tang–Zhang–Banerjee 2021).
   - It falls with κ: at n = 180 it is 89 at κ = 0.3 and 15 at κ ≥ 50. The flat-direction information grows about linearly in κL, the number of independent range patches.
   - In gamfit's basis-plus-operator-penalty model, the ridge appears as a ρ̃₂ ↔ ψ compensation, with posterior correlation +0.94 and soft vector ≈ (0.02, 0.99, 0.17) in (ρ̃₀, ρ̃₂, ψ).
4. **The four penalties hold a second range parameter [P].**
   - The identity (κ̃² + ω²)³ = κ̃⁶ + 3κ̃⁴ω² + 3κ̃²ω⁴ + ω⁶ shows that the span of the penalties contains the Matérn-κ̃ RKHS norm for *every* κ̃ (Proposition 3).
   - So the model carries a range in the basis (κ) and a range in the penalty ratios (κ̃).
   - S̃₂ is full rank. Hence r_j = 0 for j = 0, 1, 3, and the lower faces λ_j = 0 are genuine domain faces.
   - In the replica, both railed coordinates are **strict KKT face optima**: ∂V/∂λ₁|₀ = 1.7·10³ > 0 and ∂V/∂λ₃|₀ = 1.5·10⁶ > 0.
   - The "rails at −21" in the log are these faces, seen through the ρ-chart and cut by the resolvability box.
5. **ψ tail laws (Theorem 3).**
   - *κ → 0, in the ρ̃ chart:*
     - V = ½(p − s)ψ + Σ_j v_j e^{jψ}, which is analytic in κ [P].
     - When V is bounded, V − V₀ ∝ κ^q with an integer q. The measured q is **2** in all five configurations tested; the local exponents are 1.986 and 1.998, and the error shrinks by about e² per unit ψ [N].
     - This agrees with the Gu–Wang–Berger rate κ^{2 min(ν,1)} [R].
     - The face s = κ ≥ 0 is part of the mathematics, not a hand bound.
   - *κ → ∞:* V carries terms of the form poly(κ)·e^{−√5κd} from every gap d between data and centers. These are beyond all orders in 1/κ. So **there is no finite-order tail law, and nothing can be certified at ψ = +∞** [P, N].
   - *Raw chart at fixed ρ:* κ → 0 drives λ̃_m = λ_m/c_m(κ) → ∞, because c_m ~ κ^{p_m}. The ψ tail is therefore a mixed rail. This is why "psi coordinate … no exponential tail law" is true in the chart gamfit uses.
6. **`BfgsCostStallExit` is an unfinished descent, not convergence [P, N].**
   - In mgcv, gamfit's own printed predicted decrease is ½gᵀH⁻¹g = 0.347. The stall floor is τ = 10⁻⁶(1 + |V|) = 1.57·10⁻⁴.
   - Theorem 5: steepest-descent-like progress on a condition-K quadratic allows a per-step decrease below τ while the gap is Kτ/4. That gives K_eff = 4Δ/τ ≈ 8.8·10³, the same order as the replica's raw condition number of 5.3·10³.
   - The same arithmetic gives K_eff ≈ 343 for sklearn and 1.2·10⁴ for mega_batch_k.
   - A diagonal-only estimate of the remaining decrease is 0.015, so at least 95% of the 0.347 lies along the coupled (ρ, ψ) direction.
   - The gap is statistically material: it is 125× τ_stat = 1/(2n), and the stall point is 0.83 posterior sd from the optimum.
7. **The |Pg| test (0.331 against 0.0181) is chart-dependent and meaningless under this conditioning.** The only chart-free measures are Δ = ½gᵀH⁻¹g together with the Kantorovich quantity.
8. **The fix is exact-Hessian trust-region Newton in the ρ̃ chart, with faces in λ ≥ 0 and t = e^{−ρ̃} (§6).**
   - Remove the gradient-only planner override (`spatial_optimization.rs:5468`, 1779–1792).
   - Remove the ψ and ρ step caps (`:6`, 1870–1872, 5867–5869, 6587).
   - Remove the ρ box (`:40`).
   - Remove the stall guard as a terminator (`bridges.rs:300–301, 385`; `run_plan.rs:2839`).
   - Remove the relative cost floor (`run.rs:8628, 4581`).
   - In the replica, exact-Hessian TR-Newton reaches its basin minimum in 16–19 iterations from starts where the gamfit-like BFGS stalls 0.024 above it.
9. **inla_tensor fails for a different reason.**
   - The outer covariance inverts over a coordinate that is already certified as railed. Its curvature there, O(e^{−ρ}), is at noise level (−8.7·10⁻⁸ against a bar of 6.0·10⁻⁸).
   - That coordinate's contribution to the smoothing correction is O(e^{(q−2)ρ}) → 0 for the generic rail q = 1.
   - So the fix is to remove it by *rail status*, which is a Schur complement on the free block, not by an eigenvalue threshold (Proposition 7).

---

## 2. Setup and notation

**Model.** The Gaussian identity-link model with a Matérn(ν) term, as gamfit builds it for `matern(x, nu, k)` (in 1-D for the replica):

- **Kernel.** φ(r) = (1 + a + a²/3)e^{−a} with a = √5κr for ν = 5/2. For ν = 3/2, φ(r) = (1 + a)e^{−a} with a = √3κr.
- **Basis.** X(κ) = [1 | K(x, C; κ)Z]. Here C holds k centers and Z (k × (k−1)) is an orthonormal basis of 1^⊥ (the `CenterSumToZero` chart).
- **Operator penalties** (`build_matern_collocation_operator_matrices`, `matern_kernel.rs:3806`). Define D_m[i, j] = ∂_h^m φ(|h|) at h = c_i − c_j, for m = 0..3. With s = √5κ and the signed derivative polynomials:
  - p₀ = 1 + a + a²/3;
  - p₁ = −a/3 − a²/3;
  - p₂ = −1/3 − a/3 + a²/3;
  - p₃ = a − a²/3;

  we have ∂_h^m φ = (s·sgn h)^m p_m(a)e^{−a}.

  The raw penalty is S̃_m(κ) = ZᵀD_mᵀD_mZ, padded with a zero row and column for the intercept. The normalized penalty is S_m = S̃_m/c_m with c_m(κ) = ‖S̃_m‖_F (`normalize_penalty_candidate`, `matern_kernel.rs:3766`).
- **Criterion.** REML with σ² profiled out:

  V(θ) = ½(n − M_p) log D_p + ½ log|A| − ½ log|S_λ|₊,

  where:
  - A = XᵀX + S_λ;
  - S_λ = Σ_m e^{ρ_m}S_m;
  - D_p = ‖y − Xβ̂‖² + β̂ᵀS_λβ̂;
  - M_p = 1.
- **Coordinates.** θ = (ρ₀, …, ρ₃, ψ). The raw chart is gamfit's chart. The **ρ̃ chart** is ρ̃_m = ρ_m − ℓ_m(ψ) with ℓ_m = log c_m, so that e^{ρ̃_m} multiplies the *un*-normalized S̃_m.

**Resolution quantities** (all derived, none chosen):

| Symbol | Meaning | Source |
|---|---|---|
| ε_V | Rigorous bound on the computed-V error. | fp-error-analysis Theorem 2 (Cholesky route) plus the D_p channel. |
| ε_H | Bound on ‖Ĥ − H‖₂. | fp-error-analysis §3.4, Weyl. |
| band_V | The implemented ε_V (`decrement_bands.rs`). | fp-error-analysis §6.3. |
| τ_stat = 1/(2n) | Statistical resolution of V. | boundary-probability Proposition 6 and (3.5). |
| Δ = ½gᵀH⁻¹g | Quadratic-model gap on the free face. In the log it is printed as "Newton decrement" / predicted decrease, confirmed by `newton_predicted_decrease_is_curvature_scaled` at `run_plan_tests.rs:411`. | Here. |
| τ = rel_tol·(1 + \|V\|) | gamfit's cost-stall floor and `objective_tol`. rel_tol = 10⁻⁶ for these tests. | `run.rs:8628, 4581`. |

**GP comparison model** (used only in Proposition 2). y = 1β + f + ε, with f ~ GP(0, σ_f²M_κ) and ε ~ N(0, σ²I). Here λ = σ²/σ_f², so ρ = log σ² − log σ_f². The microergodic parameter is m = log σ_f² + 2νψ.

---

## 3. Results with proofs

### 3.1 Theorem 1 (normalization factorization and the chart shear) [P, N]

**Statement.** Let c_m(ψ) > 0 be C² and ℓ = log c. Define Ṽ(ρ̃, ψ) as the criterion computed with the raw penalties S̃_m(ψ). Then:

(a) **Factorization.** V(ρ, ψ) = Ṽ(ρ − ℓ(ψ), ψ).

(b) **Gradient.** ∇_ρV = ∇_ρ̃Ṽ and ∂_ψV = ∂_ψṼ − ℓ′ᵀ∇_ρ̃Ṽ.

(c) **Hessian.** With J = ∂(ρ̃, ψ)/∂(ρ, ψ) = [[I, −ℓ′], [0, 1]]:

  ∇²V = Jᵀ ∇²Ṽ J + diag(0, …, 0, −ℓ″ᵀ∇_ρ̃Ṽ).

(d) **Chart invariance of Ṽ.** Ṽ does not change under any invertible reparameterization of the coefficients α → Tα of the kernel part.
  - The quadratic form αᵀS̃_mα = Σ_k (f^{(m)}(c_k))² is the squared collocation norm of the m-th derivative of the fitted function f = Σα_jφ(|· − c_j|).
  - That form depends only on f, not on how f is written.
  - Therefore Ṽ depends on κ **only through the function space span{φ(|· − c_j|; κ)} ∩ {α ⊥ 1}** and through the collocation functionals, which are κ-free.

(e) **Conditioning.** cond(∇²V) ≤ cond(J)²·cond(∇²Ṽ) at a stationary point. Since det J = 1, cond(J) = σ_max(J)² = 1 + a²/2 + a√(1 + a²/4), where a = ‖ℓ′_F‖ over the free coordinates.

**Proof.**
- (a): S_λ = Σ_m e^{ρ_m}S̃_m/c_m = Σ_m e^{ρ_m − ℓ_m}S̃_m, and every term of V depends on ρ only through S_λ.
- (b), (c): the chain rule for the map (ρ, ψ) ↦ (ρ − ℓ(ψ), ψ). Its Jacobian is J and its only nonzero second derivative is ∂²ρ̃/∂ψ² = −ℓ″. The second-order term contracts that with ∇_ρ̃Ṽ.
- (d):
  - Under α → Tα, the design X → XT′ and the penalties S̃ → T′ᵀS̃T′, with T′ = blockdiag(1, T).
  - Then log|A| and log|S_λ|₊ shift by the same 2 log|det T′|, which cancels in V.
  - D_p is invariant, because β̂ transforms contravariantly.
  - The collocation identity is the definition of D_m.
- (e): Congruence gives cond(JᵀH̃J) ≤ cond(J)²·cond(H̃). The singular values of a unipotent 2 × 2 block [[1, −a], [0, 1]] are the roots of σ² + σ⁻² = 2 + a². ∎

**Numerical check** (`replica_charts.py`; free coordinates (ρ₀, ρ₂, ψ) at the replica optimum, where ρ₁ and ρ₃ sit on their λ = 0 faces).

The measured slopes are ℓ′ = (2.955, 2.844, 5.731, 5.840). The ρ̃ Hessian is

  H̃ = [[1.763, 0.178, −1.212], [0.178, 1.471, −7.68], [−1.212, −7.68, 45.57]].

Then JᵀH̃J = [[1.763, 0.178, −7.442], [0.178, 1.471, −16.636], [−7.442, −16.636, 210.50]]. The directly computed raw Hessian is [[1.763, 0.178, −7.44], [0.178, 1.471, −16.64], [−7.44, −16.64, 210.5]]. They agree to all printed digits.

| | Eigenvalues | cond | Jacobi-scaled cond | Soft eigenvector |
|---|---|---|---|---|
| Raw | 0.0401, 1.615, 212.1 | 5.29·10³ | 157 | (0.27, 0.96, 0.085) |
| ρ̃ | 0.171, 1.730, 46.90 | 274 | 32 | (0.017, 0.986, 0.167) |

Here cond(J) = 43.6, and the observed amplification of 19 is well inside the cond(J)² = 1.9·10³ bound.

**Corollaries.**
- **(i) Newton is affine-invariant; first-order methods are not.** Exact Newton steps, and TR/ARC steps in the Hessian norm, are unchanged by the *linear* part of the shear. BFGS and projected steepest descent with a Euclidean or |Pg| test are not.
  - So the shear is primarily an explanation for why the BFGS path fails.
  - For Newton it still matters in two ways:
    - it inflates the Lipschitz constant of H through the ψ-dependence of ℓ′(ψ);
    - away from stationarity it adds the indefinite term −ℓ″ᵀ∇_ρ̃Ṽ to H_ψψ.
- **(ii) The ρ̃ chart is available exactly and cheaply.** Freeze the normalization at the seed: S_m(ψ) := S̃_m(ψ)/c_m(ψ₀). Then V_frozen(ρ, ψ) = Ṽ(ρ − ℓ(ψ₀), ψ), a constant translation of ρ̃.
  - The scale convention that normalization exists to provide (unit-strength penalties at the seed) is kept.
  - The shear, and the ℓ″ term, are gone.
  - No constant is introduced: every ψ₀ gives the same criterion up to relabelling ρ.

### 3.2 Proposition 2 (the GP microergodic ridge and the conditioning as a function of (n, κ)) [R, N]

**Statement** (GP comparison model, d = 1, fixed ν, bounded domain [0, L]).

(a) [R: Zhang 2004 Thm 2] The Gaussian measures for (σ_f², κ) and (σ_f²′, κ′) are equivalent iff σ_f²κ^{2ν} = σ_f²′κ′^{2ν}. With the nugget variance fixed, the likelihood is therefore asymptotically flat along dm = 0, which in the gamfit chart is **(dρ, dψ) ∝ (2ν, 1)**.

(b) [R: Kaufman–Shaby 2013 Thm 1; Tang–Zhang–Banerjee 2021 Thms 2.4, 2.7; Chen–Simpson–Ying 2000] The Fisher information for m grows like:
  - n without a nugget;
  - n^{d/(2ν+d)} = n^{1/6} with a nugget (ν = 5/2, d = 1).

  The information along the flat direction stays bounded as n → ∞ at fixed κL.

(c) [N; heuristic law] Under infill:

  cond_{(ρ,ψ)} ≍ I_mm(n, κ)/I_flat(κL),

  and I_flat grows about linearly in κL, the number of effectively independent range patches, until κL ~ n/const. Under increasing domain (L ∝ n) the condition number is O(1) [R: Mardia–Marshall 1984; Bachoc 2014].

**Why (a) is a ridge in gamfit's coordinates.**
- λ = σ²/σ_f² gives ρ = log σ² − log σ_f².
- With σ² profiled, m = log σ_f² + 2νψ = −ρ + 2νψ + const.
- Setting dm = 0 gives dρ = 2ν dψ.
- The high-frequency form of the spectral density, f(ω) ∝ σ_f²κ^{2ν}|ω|^{−2ν−d}(1 + O(κ²/ω²)), is the reason only m is identified (Stein 1999, ch. 4).
- Along the ridge, predictions are asymptotically invariant (Stein 1999, ch. 3–4; Kaufman–Shaby 2013 §3).

**Numerical check** (`gp_reml_information.py`, `gp_reml_efficient.py`): exact REML Fisher information I_ij = ½tr(PV_iPV_j), with σ_ε = 0.08 and uniform x on [0, 1].

| κ | cond at n = 50 → 800 | Log-log slope | Angle(soft, (2ν, 1)) |
|---|---|---|---|
| 2.2 | 45 → 102 | +0.29 | 2.3–4° |
| 5 | 29 → 85 | +0.38 | 2.3–4° |
| 15 | 15 → 60 | +0.51 | 2.3–4° |
| 5, no nugget | 61 → 325 | +0.82 (I_mm slope +1.06) | — |

The efficient information for m has slopes of +0.37 to +0.43 with a nugget, which is preasymptotic relative to 1/6. Without a nugget the slopes are +0.61 to +1.05. The orthogonal (flat) efficient information has slopes of +0.06 to +0.18.

At n = 180, as a function of κ:

| κ | 0.3 | 1 | 2.2 | 5 | 15 | 50 | 150 |
|---|---|---|---|---|---|---|---|
| cond | 89 | 79 | 70 | 54 | 32 | 15 | 15.5 |
| I_flat,orth | 0.77 | 3.9 | 8.7 | 19 | 52 | 121 | 141 |

**What carries over to gamfit, and what does not.**
- gamfit's model is not the GP. It is a rank-(k−1) kernel basis with four normalized collocation penalties.
- The ridge survives as a *soft, isolated* direction: ρ̃₂ ↔ ψ, with ratio 0.986/0.167 ≈ 5.9, close to 2ν = 5, and posterior correlation +0.94. Proposition 3 explains why it is ρ̃₂ rather than a common scale.
- It does **not** become exactly flat, because the finite basis is not an equivalent-measure family.
- So the right certificate is the Morse one (Theorem 4(i)), not a quotient.

### 3.3 Proposition 3 (the penalty family contains a second range; the lower faces) [P, N]

**Statement (d = 1, ν = 5/2).**

(a) For every κ̃ > 0,

  ‖f‖²_{H(κ̃)} ∝ ∫(κ̃² + ω²)³|f̂(ω)|² dω = κ̃⁶‖f‖² + 3κ̃⁴‖f′‖² + 3κ̃²‖f″‖² + ‖f‴‖².

  gamfit's four penalties are collocation discretizations of ‖D^mf‖², m = 0..3, on the kernel span. So the penalty λ_m ∝ τ(κ̃⁶, 3κ̃⁴, 3κ̃², 1) approximates the Matérn-κ̃ RKHS norm, with κ̃ independent of the basis κ.

(b) If S̃₂ is full rank on the centered space, then range(S̃_j) ⊆ range(S̃₂) for all j. In the notation of boundary-probability Proposition 5 this gives r_j = h_j = 0 for j ≠ 2. So V stays bounded as λ_j → 0, and **λ_j = 0 is a domain face** with the KKT condition ∂V/∂λ_j|_{λ_j=0} ≥ 0.

**Proof.**
- (a): Expand the binomial and apply Plancherel. The spectral density of Matérn-ν in d = 1 is ∝ (κ̃² + ω²)^{−(ν+1/2)}, and ν + ½ = 3.
- (b): This is boundary-probability Proposition 5 with r_j = 0: the directions penalized only by S_j are the zero space. ∎

**Consequence.**
- There are two range parameters: κ in the basis and κ̃ in the penalty ratios. The one-parameter penalty curve of (a) spans the same direction as moving κ̃ ∝ e^{ψ}.
- So the criterion can trade ψ against the lower-order weights, which is the observed ρ̃₂ ↔ ψ compensation.
- The optimum can also switch off penalties (λ_j = 0) rather than choose κ̃, which is the observed railing.
- In the SPDE reading of Lindgren–Rue–Lindström 2011, a sparse subset of the (κ̃² − Δ)^{ν+1/2} expansion is being selected.

**Numerical check** (`replica_lower_faces.py`, at the replica optimum ψ = −0.0716):
- The numerical ranks of the normalized S_m on the 19-dimensional centered space are 14, 18, 19 and 19. S₀ is the square of an ill-conditioned kernel Gram, with λ_min = 4·10⁻¹⁶.
- S₂ is well conditioned, with eigenvalues from 1.5·10⁻² to 0.96, so (b) applies.
- At λ₁ = λ₃ = 0, V = 42.026247, which equals the replica optimum.
- The face derivatives are ∂V/∂λ₁ = 1.666·10³ and ∂V/∂λ₃ = 1.463·10⁶. Both are > 0, so the face KKT conditions hold strictly.
- In ρ, the same derivatives read ∂V/∂ρ_j = λ_j∂V/∂λ_j. At ρ₁ = −21 that is 1.3·10⁻⁶; at ρ₃ = −21 it is 1.1·10⁻³.
- This is why the ρ-chart iterates drift: ρ₃ wanders to −29 in the unconstrained replica, and the log shows rails at −21 to −22.8.

### 3.4 Theorem 3 (ψ tail laws) [P, N]

**(a) κ → 0 in the ρ̃ chart [P].** Fix ρ̃ and assume D_p(0⁺) > 0, i.e. y is not interpolated by the limit span. Let p = ord_{κ=0} det A(κ) and s = ord_{κ=0} det S̃_λ(κ) restricted to the centered block. Then there are real v_j with

  Ṽ(ρ̃, ψ) = ½(p − s)ψ + Σ_{j≥0} v_j e^{jψ},

and the series converges for e^ψ < R, with R > 0. If V is bounded as ψ → −∞ (p = s), let q be the first j ≥ 1 with v_j ≠ 0. Then:

  V − V₀ = v_q e^{qψ}(1 + O(e^ψ)),
  ∂_ψV = q·v_q e^{qψ}(1 + O(e^ψ)),
  ∂²_ψV/∂_ψV → q,
  d log|∂_ψV|/dψ → q.

*Proof.*
1. For fixed h, a = √5κ|h| is linear in κ. The function φ(a) = (1 + a + a²/3)e^{−a} and the derivative forms (s·sgn h)^m p_m(a)e^{−a} are entire in a.
2. So every entry of X(κ) and S̃_m(κ) is entire in κ.
3. det A and det S̃_λ are therefore entire and not identically zero. For κ > 0, A ≻ 0, and S̃_λ ≻ 0 on the centered block. So det A = κ^p(a₀ + O(κ)) and det S̃_λ = κ^s(b₀ + O(κ)), with a₀, b₀ ≠ 0.
4. D_p(κ) = yᵀ(I − XA⁻¹Xᵀ)y is a ratio of entire functions (Cramer's rule). It is bounded on (0, κ₀), since 0 ≤ D_p ≤ ‖y‖². A ratio of entire functions bounded on a real interval next to 0 has a removable singularity there. With D_p(0⁺) > 0, log D_p is analytic at 0.
5. Adding the three terms of V gives the expansion.
6. The statements about ∂_ψ follow from ∂_ψ = κ∂_κ applied term by term. ∎

**Measured q [N]** (`replica_tail_laws.py` at mpmath dps = 100; `replica_tail_generic.py`; the ψ-FD step 10⁻²⁵ is for verification only).

At ρ̃ = (−5.70, −24.65, −20.03, −44.49):

| ψ | V | ∂_ψV | Local exponent q_loc |
|---|---|---|---|
| −3 | 379.323 | −26.7 | |
| −4 | 384.457 | −9.95·10⁻² | |
| −5 | 384.47422 | −2.02·10⁻⁴ | 2.28 |
| −6 | | 2.06·10⁻⁵ | 1.87 |
| −7 | | 3.17·10⁻⁶ | 1.986 |
| −8 | | 4.35·10⁻⁷ | 1.998 |
| −9 | 384.4742080 | 5.90·10⁻⁸ | |

- The deviations |q_loc − 2| (0.28, 0.13, 0.014, 0.002) shrink by a factor of 2–9 per unit ψ, roughly e². This suggests the next correction is O(κ²) relative, i.e. an even series.
- Four more configurations all converge to q_loc = 2.000 at ψ = −6 … −10:
  - two ρ̃ vectors on the same data;
  - one ρ̃ vector on different data (y = cos 9x + eˣ + N(0, 0.3²)), run at two starting points.
- In one of these the transient first shows 6.08 and 4.47, which is a switch of the leading order as the mass penalty on the x² direction takes over.
- The drop of V between ψ = −3 and −4 is that takeover: the kernel part's constant scales like κ⁻², which rescales the mass penalty.

**Why q = 2 is plausible [C, partial argument].**
- Expanding the kernel gives φ(a) = 1 − a²/6 + a⁴/24 − a⁵/45 + O(a⁶). There are no a¹ or a³ terms, so the first odd power is a^{2ν} = a⁵.
- Every matrix entry is therefore a series in κ², plus odd terms starting at κ⁵.
- If the leading orders p and s and the leading D_p numerator and denominator are all set by the κ²-part, every relative correction up to order 5 is even. That forces q ∈ {2, 4}, and q = 2 generically.
- This agrees with Gu, Wang and Berger's (2018) Table 1 rate κ^{2 min(ν,1)} for the GP marginal likelihood. For ν = 3/2 the same argument gives an even leading order with an odd κ³ correction.
- A proof that the κ²-part sets the orders is open (§7).

**Certificate at the κ = 0 face [P, given q = 2].**
- Use s = κ ≥ 0, which is a domain constraint. In s, ∂_sV|₀ = v₁ = 0 and ∂²_sV|₀ = 2v₂.
- A face point (ρ̃*, s = 0) is a certified strict local minimum iff:
  - the free-block Kantorovich test of certificate-theory Theorem 1 holds for V₀(ρ̃) = V(ρ̃, s = 0), and
  - v₂ > band.
- v₂ comes from the exact second-order perturbation of the matrix series; it is not a finite difference.
- In ψ, the same conditions appear as ∂_ψV > 0 and ∂²_ψV/∂_ψV → 2 on the tail, with deviation O(e^{Δψ}). The sign of ∂_ψV > 0 above says V decreases toward the face.

**(b) κ → ∞ [P, N].** The kernel entries at distinct points are poly(κd)e^{−√5κd}, where d is the data–center or center–center distance. At coincident points (h = 0), the values are φ(0) = 1, φ′(0) = 0, φ″(0) = −5κ²/3 and φ‴(0) = 0. Hence:

  V = V_alg(κ⁻¹; ρ̃) + Σ_d P_d(κ)e^{−√5κd},

where V_alg carries the coincident-node terms, including a λ̃₂κ⁴·(25/9) diagonal from S̃₂.

- Each exponential term has all derivatives in 1/κ equal to zero at 1/κ = 0, so it is invisible to any finite-order expansion.
- Its local exponent is d log|g|/dψ ≈ −√5κd, which is unbounded.
- On the range where some √5κd_min = O(log(1/ε_V)), the local exponent and the ratio H/g are neither constant nor monotone.
- In the replica (complex step, double precision) at ψ = 3, …, 10: g = 28.5, 208, 66.6, 6.88, −0.505, −8.69, −9.37, −3.24.
- **Conclusion:** no finite-data tail law certifies ψ = +∞. The profile V*(ψ) rises steeply (42.0 at ψ = 0; 354 at ψ = 5), so this end is never an optimum in the tests. Iterates heading there must end in a typed failure.

**(c) The raw chart [P].**
- c_m(κ) = ‖S̃_m‖_F → 0 with orders p_m ≥ 1 as κ → 0. For m = 0, D₀ → 11ᵀ and 11ᵀZ = 0, so c₀ = O(κ²). For m ≥ 1 the factor s^m adds κ^m.
- At fixed raw ρ, λ̃_m = e^{ρ_m}/c_m → ∞, so the raw ψ → −∞ tail is a *mixture* of upper rails in every penalty.
- Its ψ-derivative mixes q_m-laws with different exponents, and no single exponential law exists. The refusal at `run.rs:6446–6455` ("psi coordinate … no exponential tail law") is therefore correct in the raw chart.
- In the ρ̃ chart (a) gives an integer law.

**Flat-limit function space [R].** As κ → 0 the span, suitably rescaled, tends to span{1, x, x²} ⊕ {Σα_j|x − c_j|⁵ : α ⊥ P₂}. This is polyharmonic, with |r|³ for ν = 3/2 (Song–Riddle–Fasshauer–Hickernell 2012; Driscoll–Fornberg 2002; Schaback 2005). V₀ is the REML of that limiting spline model, which is why it is finite.

### 3.5 Theorem 4 (flat valley versus ill-conditioned; certificates for degenerate minima) [P, R]

Let θ* be a critical point of V restricted to its face. Let Ĥ be the computed free-block Hessian with ‖Ĥ − H‖₂ ≤ ε_H, and let μ̂₁ ≤ … be its eigenvalues.

**(i) Isolated but ill-conditioned (Morse).** If μ̂₁ > ε_H, then H(θ*) ≻ 0 by Weyl. The H-norm Newton–Kantorovich test of certificate-theory Theorem 1 certifies a unique strict local minimum in the Ĥ-ball of radius r₋.
- Ill-conditioning enters *only* through the Lipschitz constant L_H of the Hessian, measured in the H-norm. It does not enter through cond(H).
- So cond = 10⁴ is not an obstacle to certification.

**(ii) Flat valley (Morse–Bott) with a known invariance.** Suppose the critical set is a k-manifold M whose tangent space T is known analytically. An example is a published criterion invariance (`with_criterion_invariance`, `spatial_optimization.rs:2068`, or certificate-theory Proposition 6).
- If H is nondegenerate on the normal space N = T^⊥, the Łojasiewicz exponent is ½ (Feehan 2020; Feehan–Maridakis 2020). V − V* ≍ dist(θ, M)² and ‖∇V‖ ≍ dist(θ, M).
- Certify by applying Theorem 1 to the restriction to N: use g_N, H_NN, and the pseudo-inverse decrement ½g_NᵀH_NN⁻¹g_N.
- The tangential components must vanish *exactly* by the invariance. So their computed values must be within ε_g, and this is a checkable consistency test.

**(iii) Rails.** On a coordinate going to a face, compactify: t = e^{−ρ̃} for the upper face (boundary-asymptotics R8; compactification Theorem 1), λ ≥ 0 for the lower face when r_j = h_j (boundary-probability C5), and s = κ for the ψ-face (Theorem 3(a)). Then certify with face KKT plus the free-block Kantorovich test (certificate-theory Theorem 4).

**(iv) Degenerate minimum without a known invariance.** Suppose some computed eigenvalues satisfy |μ̂| ≤ ε_H on a subspace U, and no analytic invariance explains U. Then:

- (P) No finite-precision test decides whether θ* is a minimum. V restricted to θ* + U can have either sign of curvature at every order resolvable above ε_V.
  - *Proof:* the perturbation ±ε_H·uuᵀ on U is consistent with all computed data, and it flips the sign of the second-order term.
  - A cubic or quartic term at order ε_V/r³ is likewise unresolvable within radius r ≤ (ε_V/L)^{1/3}. ∎
- (P) What *can* be certified is a fit-level statement. Let F be the fitted values, or any functional the user reports.
  - If ‖∂F/∂θ·u‖ ≤ δ_F for u ∈ U, uniformly on the U-directions of the resolved ball, then within the ball every point of θ* + U gives the same fit to δ_F·r.
  - This is the finite-sample analogue of Stein's and Kaufman–Shaby's prediction invariance along the microergodic ridge.
  - It should be reported as "hyperparameters unidentified along U; fit certified", a typed outcome distinct from a certified minimum. It is **not** a convergence certificate for θ, and it is not needed in the Matérn tests.

**Application [N].**
- In the replica the joint Matérn optimum has μ₁ = 0.171 in ρ̃ and 0.040 in raw coordinates.
- The replica's criterion noise is 3·10⁻⁹ to 10⁻⁸ (`replica_noise.py`; Cholesky and LU solves agree to 6·10⁻⁷ in V at κ = 0.3, where cond(X) = 3.5·10¹⁰). So ε_H is at least 10⁵× smaller than μ₁.
- **It is case (i).** The ψ direction must not be deflated. Publishing a ψ component in the invariance, which was considered earlier in this lane, is **rejected**. The soft direction is a genuine curvature of 0.17 (posterior sd 2.4 e-folds in ρ̃₂), not a symmetry.

### 3.6 Theorem 5 (a cost-stall window is not a certificate; the decrement is) [P, R]

**Statement.**

(a) [R: Nocedal–Wright 2006 Thm 3.3; Akaike 1959] Let f = ½xᵀQx with cond(Q) = K. Steepest descent with exact line search satisfies

  f_{k+1} − f* ≤ ((K − 1)/(K + 1))²(f_k − f*),

and this is attained for the worst starting vector. The per-step decrease is then

  (f_k − f_{k+1}) = [4K/(K + 1)²](f_k − f*) ≈ (4/K)(f_k − f*).

So W consecutive decreases ≤ τ are consistent with a gap f_k − f* ≈ Kτ/4, for every window length W.

(b) [P] Let V be C³ with H ⪰ μI near θ* and Hessian Lipschitz L in the H-norm. Let δ = ‖H⁻¹g‖_H = √(2Δ). Then

  V − V* = Δ·(1 + O(Lδ)).

The quadratic model's gap Δ = ½gᵀH⁻¹g is therefore the chart-free first-order estimate of the remaining decrease. Certificate-theory Theorem 1 turns it into a rigorous statement once h = L‖H⁻¹g‖_H ≤ ½.

(c) [P] BFGS with memory resets, cap-truncated steps and projected components is not better than (a) in the worst case. After a reset its step is the steepest-descent step. On directions of negative curvature, which the replica has with μ ≈ −5·10⁻³ in the ρ₀ tail at the stall, the secant update cannot model the curvature and discards the pair.

*Proof.*
- (a): The Kantorovich inequality, as in Nocedal–Wright Thm 3.3. Its worst case is a start with equal H-weight on the extreme eigenvectors. The decrease formula follows by algebra.
- (b): Taylor-expand around θ* in the H-norm. The cubic remainder is bounded by (L/6)‖θ − θ*‖³_H, and ‖θ − θ*‖_H = δ(1 + O(Lδ)) by the Newton–Kantorovich radii.
- (c): The BFGS update is skipped when sᵀy ≤ 0 (`replica_optimizers.py`, line 61; opt lib.rs has the same guard). Resets set H⁻¹ ∝ I. ∎

**Consequence: the effective condition number printed by every failure.** K_eff := 4Δ/τ.

| Test | Δ (log) | τ | K_eff | n | τ_stat = 1/(2n) | Δ/τ_stat |
|---|---|---|---|---|---|---|
| mgcv_matern_smooth (line 15831) | 0.3474 | 1.570·10⁻⁴ | 8.8·10³ | 180 | 2.78·10⁻³ | 125 |
| sklearn_gp_matern (33534) | 5.97·10⁻³ | 6.96·10⁻⁵ | 343 | 150 | 3.33·10⁻³ | 1.8 |
| mega_batch_k matern (14600) | 0.912 | 2.963·10⁻⁴ | 1.2·10⁴ | 200 | 2.5·10⁻³ | 365 |
| statsmodels additive (15518) | 1.459·10⁻² | 1.826·10⁻⁴ | (line search) | 250 | 2.0·10⁻³ | 7.3 |

- mgcv's K_eff is within a factor of 2 of the replica's raw condition number of 5.3·10³.
- Every Δ exceeds τ_stat as well as ε_V, so **none of these points is converged by any derived criterion.**
- By boundary-probability Proposition 6, mgcv's stall point is √(2·0.347) = 0.83 posterior sd from the optimum in the worst direction. The admissible distance is η = n^{−1/2} = 0.075.

### 3.7 Forensics of the mgcv `BfgsCostStallExit` (129 iterations, |Pg| = 0.331 against 0.0181) [P, N]

**Log line 15831:**
- V = −155.9986;
- checkpoint θ = (−4.9937, −21.1714, −22.7791, −10.9017, ψ = 0.78733);
- tail-snap probes:
  - k = 0: g = 0.1384, H = 1.676, ratio 12.1;
  - k = 3: g = 0.2949, H = 4.872, ratio 16.5;
  - k = 4: refused, "psi coordinate";
- printed predicted decrease 0.3474 against τ = 1.570·10⁻⁴.

**How the 0.331 decomposes.** √(0.1384² + 0.2949²) = 0.326, so the remaining projected-gradient mass is |g_ψ| ≈ 0.06, assuming the railed components are projected out.

**Where the missing decrease lies.**
- The diagonal Newton estimates are g₀²/(2H₀₀) = 5.7·10⁻³ and g₃²/(2H₃₃) = 8.9·10⁻³.
- Even with a generous ψ term, the diagonal estimate totals about 0.015. The full ½gᵀH⁻¹g = 0.347.
- So **at least 95% of the predicted decrease lies along the coupled soft (ρ, ψ) direction.** In the replica that is the ρ̃₂ ↔ ψ direction, which is invisible to per-coordinate tests.

**Chain of causes.** Each link is checked in the replica with `replica_optimizers.py` and `replica_stall_diag.py`.

1. **Planner.** The exact outer Hessian is available (`analytic_outer_hessian_available` at `spatial_optimization.rs:1779`). It is withheld by `suppress_outer_hessian_for_nfree` (1670, 1864), and `with_prefer_gradient_only(true)` (5468; comments at 1784–1792) forces BFGS ahead of the ARC arm of `capability::plan`.
2. **Chart.** The raw chart adds the shear of Theorem 1, a ×19 increase in cond in the replica.
3. **Faces.** Lower faces in the ρ-chart are exponentially flat (Proposition 3: ∂V/∂ρ_j = λ_j c_j) and are cut by the box `joint_rho_resolvability_domain` (40). The iterates therefore sit on box corners with O(1) curvature coupling to the interior coordinates.
4. **Negative curvature in the ρ tails.** The replica's stall point has eigenvalues (−4.9·10⁻³, 4.1·10⁻³, 1.20, 2.12, 175.4), and the soft or negative direction is ≈ e₀. BFGS cannot represent it.
5. **The stall floor τ = rel_tol·(1 + |V|).** It is magic and depends on units. It fires once six consecutive decreases are below τ, which by Theorem 5(a) happens while the gap is ≈ K_eff τ/4.

**Replica evidence** (projected BFGS mimicking gamfit: box ρ ∈ [−21, 23], ψ ∈ [−3, 6]; caps 5 and ln 2 with uniform scaling; Armijo with ≤ 50 halvings; window 6; rel_tol 10⁻⁶):

| Start | Raw BFGS | ρ̃ BFGS | Exact-Hessian TR-Newton (raw) |
|---|---|---|---|
| (0,0,0,0,1.5) | cost_stall, 40 it, V = 49.20090, **0.0245 above** its basin min 49.17643 | cost_stall, 34 it, V = 42.02625, \|Pg\| 9.4·10⁻⁴ (V within 2·10⁻⁶ of V\*: benign) | 19 it, V = 49.17643 |
| (−3,−3,−3,−3,0.8) | 64 it, V = 42.02737 | 38 it, 49.17645 | 100 it (crude projection), 42.02735, \|Pg\| 6·10⁻⁵ |
| (2,−5,−5,2,2.5) | 86 it, 42.02736, \|Pg\| 1.1·10⁻⁵ | 38 it, 42.02625 | 16 it, 49.17643 |

- **Removing the caps does not fix it.** Without them the stall comes after 36 iterations with |Pg| = 3.1·10⁻², and the structure is the same. The caps are not the cause, though they are still unprincipled (§6).
- **The box costs 1.1·10⁻³ in V** (42.0274 against 42.0262). That is below τ_stat for n = 180 but far above ε_V.
- **Multimodality.** The replica has two ψ-basins: 42.026 at ψ = −0.07 and 49.176 at ψ = 2.58, with ΔV = 7.15. Each local method converges to one of them depending on the start. The profile (`replica_profile.py`, verification only) is multi-branched: branches switch at ψ ≈ 1.25, where ρ̃₃ ≈ −20 becomes active.

**The tail-snap refusals are correct.** The ratio band `TAIL_SNAP_CURVATURE_BAND = (0.25, 4.0)` (`run.rs:6356`) tests H/g ≈ q, which is the signature of an exponential tail. Ratios of 12.1 and 16.5 say those coordinates are curvature-dominated, i.e. interior. ψ = 0.787 is interior as well. A ψ tail law would not have rescued this fit, and it does not rescue sklearn (ψ = 0.789) or mega_batch_k (ψ = 0.699) either.

### 3.8 statsmodels line-search failure and inla_tensor covariance

**statsmodels (line 15518).**
- The run ends in line_search_failed MaxAttempts after 50 halvings ("bracket never closed") and 162 iterations, with |Pg| = 3.72·10⁻² against 2.51·10⁻².
- ρ₃ is railed at +21.63 (upper) and ρ₄ at −18.41 (lower), and the Matérn ψ = −0.568.
- 2Δ = 2.9·10⁻² ≫ ε_V, so by fp-error-analysis Proposition 10 (the Armijo stall law) **this is not floating-point failure**.
- The mechanism is the projection kink. Along a BFGS direction that pushes railed coordinates into the box, the projected path φ(α) = V(P(θ + αd)) is only piecewise smooth, and the Wolfe curvature condition can fail on every piece.
- A trust-region step on the free face with faces in the compactified charts needs no bracket.
- Matérn ν = 3/2 here with a linear truth (0.5·x₃): Theorem 3(a) predicts a q = 2 law if ψ → −∞ becomes optimal, which is future-proofing only.

**Proposition 7 (rail coordinates in the smoothing-parameter correction) [P].**

*Setting.* This is the correction term J_ρΣ_ρJ_ρᵀ of the Wood–Pya–Säfken (2016) corrected covariance. Here J_ρ = ∂β̂/∂ρ, Σ_ρ = H_ρ⁻¹, and coordinate j is on an upper rail with the generic tail law V = V_∞ + v_q e^{−qρ_j} (boundary-asymptotics R3).

*Claim.* The column J_{ρ,j} = O(e^{−ρ_j}) and Σ_{ρ,jj} ≍ e^{qρ_j}/(q²v_q). So the j-contribution to the correction is O(e^{(q−2)ρ_j}), which → 0 for q = 1. Couplings to free coordinates enter through the Schur complement and are smaller still.

*Proof.*
- ∂β̂/∂ρ_j = −A⁻¹λ_jS_jβ̂.
- As λ_j → ∞, the range(S_j) part of β̂ is O(λ_j⁻¹), so λ_jS_jβ̂ = O(1).
- A⁻¹ maps range(S_j) with norm O(λ_j⁻¹), by a Schur complement on the range and null split of S_j.
- The curvature is H_jj = q²v_q e^{−qρ_j}(1 + o(1)). ∎

*Consequence.*
- At inla_tensor (line 15629) the code notes that the point is "certified on an infinite-smoothing rail", then still forms `OuterHessianInverse` over all coordinates (`smoothing_correction.rs:346, 2024`).
- It meets a noise-level eigenvalue, −8.659·10⁻⁸ against a Weyl bar of 6.046·10⁻⁸. The chain-rule term is 4.2·10⁻²¹ and 0 null directions are deflated.
- The railed coordinate must be removed by status: invert H_FF on the free block F only, and set its correction contribution to its limit, which is 0 for q = 1.
- An eigenvalue threshold is the wrong tool. Its sign is set by rounding and tail cancellation (the fp and boundary-asymptotics lanes explain why the band is exceeded).
- For q ≥ 2 rails, or a face at t = 0 in the compactified chart, the correct treatment is the boundary law of the boundary-probability lane.

---

## 4. Numerical checks

All scripts are in `SP/theory/matern-identifiability/` and use the venv at `SP/theory/venv`. Complex-step derivatives (h = 10⁻³⁰) serve as the exact first derivatives. Hessians are 4th-order differences of complex-step gradients. Finite differences and mpmath are used here only as verification tools.

| # | Script | Claim | Result |
|---|---|---|---|
| 1 | `gp_reml_information.py` | Prop. 2 (a)–(c) | Soft vector 2.3–4° from (2ν, 1); cond slopes +0.29/+0.38/+0.51 at κ = 2.2/5/15; no nugget +0.82; cond(κ) at n = 180: 89 → 15; I_flat,orth 0.77 → 141 |
| 2 | `gp_reml_efficient.py` | Prop. 2 (b) rates | Efficient I_m slope +0.37 to +0.43 with nugget (theory 1/6, preasymptotic); +0.61 to +1.05 without; flat slope +0.06 to +0.18 |
| 3 | `replica_charts.py` | Thm 1 | c_m = (3.04, 30.5, 47.5, 711); ℓ′ = (2.955, 2.844, 5.731, 5.840); cond 5.29·10³ raw against 274 ρ̃; JᵀH̃J reproduces raw H |
| 4 | `replica_noise.py` | ε_V, ε_H scale | cond(X) = 3.5·10¹⁰, 8.7·10⁷, 1.9·10⁶, 3.8·10⁴, 2.2·10², 10 at κ = 0.3, 1, 2.2, 5, 15, 50; criterion noise 3·10⁻⁹ to 10⁻⁸; Cholesky against solve 6·10⁻⁷ at worst |
| 5 | `replica_optimum.py` | Global replica optimum | V = 42.02625 at ψ = −0.0716 (κ = 0.931), ρ₀ = −4.598, ρ₂ = −16.541, ρ₁ and ρ₃ → −∞ |
| 6 | `replica_lower_faces.py` | Prop. 3 (b) | ranks 14/18/19/19; face V = 42.026247; ∂V/∂λ₁ = 1.67·10³, ∂V/∂λ₃ = 1.46·10⁶ (strict KKT) |
| 7 | `replica_optimizers.py` | §3.7, Thm 5 | Table in §3.7 |
| 8 | `replica_stall_diag.py` | §3.7 | Stall g = (−0.006, 0.0041, 0.0083, 0.0045, −0.0404); eigenvalues (−4.9·10⁻³, 4.1·10⁻³, 1.20, 2.12, 175.4); the no-cap run also stalls |
| 9 | `replica_profile.py` | Multimodality, ridge slope | V*(ψ): 43.885, 43.214, 42.407, 42.040, 42.913, 45.311 at ψ = −1.5, …, 1.0; second branch 48.3–49.2 at ψ = 2–2.5; 354 at ψ = 5; dρ̃₂*/dψ ≈ 5.9 |
| 10 | `replica_tail_laws.py` | Thm 3 (a), (b) | q_loc → 2 (1.986, 1.998) at ψ = −7, −8; +∞ side erratic up to ψ = 10 |
| 11 | `replica_tail_generic.py` | Thm 3 (a) generic | q_loc → 2.000 in four further configurations |

**Caveat.** The replica is structural. It uses gamfit's formulas with its own data (seed 456, 20 quantile centers). The log's numbers come from gamfit's own data. The agreement of K_eff (8.8·10³ against 5.3·10³) is therefore of order of magnitude, not exact.

---

## 5. Literature (precise citations)

**Identifiability of Matérn parameters**
- **Zhang, H. (2004).** Inconsistent estimation and asymptotically equal interpolations in model-based geostatistics. *JASA* 99(465):250–261. **Theorem 2**: equivalence iff σ²κ^{2ν} is equal (d ≤ 3). Used in Prop. 2(a).
- **Stein, M. L. (1999).** *Interpolation of Spatial Data: Some Theory for Kriging.* Springer. **Ch. 4** (equivalence of Gaussian measures, spectral tails) and **Ch. 6** (likelihood-based estimation, microergodicity). Used in Prop. 2 and Thm 4(iv).
- **Kaufman, C. G. & Shaby, B. A. (2013).** The role of the range parameter for estimation and prediction in geostatistics. *Biometrika* 100(2):473–484, doi:10.1093/biomet/ass079. **Thm 1** (asymptotic normality of the microergodic parameter at a misspecified range) and §3 (prediction insensitivity). Used in Prop. 2(b) and Thm 4(iv).
- **Tang, W., Zhang, L. & Banerjee, S. (2021).** On identifiability and consistency of the nugget in Gaussian spatial process models. *JRSS-B* 83(5):1044–1070, doi:10.1111/rssb.12472. **Thms 2.4, 2.7**: rates with a nugget, n^{−d/(2(2ν+d))} for the microergodic parameter. Used in Prop. 2(b).
- **Chen, H.-S., Simpson, D. G. & Ying, Z. (2000).** Infill asymptotics for a stochastic process model with measurement error. *Statistica Sinica* 10:141–156. The n^{1/4} rate for ν = ½ with error. Used in Prop. 2(b).
- **Du, J., Zhang, H. & Mandrekar, V. S. (2009).** Fixed-domain asymptotic properties of tapered maximum likelihood estimators. *Ann. Statist.* 37(6A):3330–3361. The microergodic CLT for Matérn. Background for Prop. 2(b).
- **Bachoc, F. (2014).** Asymptotic analysis of the role of spatial sampling for covariance parameter estimation of Gaussian processes. *J. Multivariate Anal.* 125:1–35. Increasing-domain consistency and asymptotic normality. Used in Prop. 2(c).
- **Mardia, K. V. & Marshall, R. J. (1984).** Maximum likelihood estimation of models for residual covariance in spatial regression. *Biometrika* 71(1):135–146. Increasing-domain asymptotics. Used in Prop. 2(c).
- **Gu, M., Wang, X. & Berger, J. O. (2018).** Robust Gaussian stochastic process emulation. *Ann. Statist.* 46(6A):3038–3066, doi:10.1214/17-AOS1648. **Assumption 3.2 and Table 1**: the expansion of the correlation as range → ∞, with rate κ^{2 min(ν,1)}. **Lemma 3.3**: the marginal likelihood is O(1). Consistent with Thm 3(a), q = 2 for ν = 5/2. Their reference prior is not adopted, because it changes the estimand.

**SPDE link**
- **Lindgren, F., Rue, H. & Lindström, J. (2011).** An explicit link between Gaussian fields and Gaussian Markov random fields: the SPDE approach. *JRSS-B* 73(4):423–498. The (κ² − Δ)^{α/2} representation. Used in Prop. 3.

**Flat limits of RBFs**
- **Driscoll, T. A. & Fornberg, B. (2002).** Interpolation in the limit of increasingly flat radial basis functions. *Comput. Math. Appl.* 43(3–5):413–422.
- **Schaback, R. (2005).** Multivariate interpolation by polynomials and radial basis functions. *Constr. Approx.* 21:293–317.
- **Song, G., Riddle, J., Fasshauer, G. E. & Hickernell, F. J. (2012).** Multivariate interpolation with increasingly flat radial basis functions of finite smoothness. *Adv. Comput. Math.* 36(3):485–501. For finite-smoothness kernels the flat limit is a polyharmonic spline. Used in Thm 3 (flat-limit space).

**Degenerate critical points**
- **Bott, R. (1954).** Nondegenerate critical manifolds. *Ann. of Math.* 60(2):248–261. Used in Thm 4(ii).
- **Łojasiewicz, S. (1963).** Une propriété topologique des sous-ensembles analytiques réels. *Les Équations aux Dérivées Partielles*, Éditions du CNRS, 87–89.
- **Feehan, P. M. N. & Maridakis, M. (2020).** Łojasiewicz–Simon gradient inequalities for analytic and Morse–Bott functions on Banach spaces. *J. reine angew. Math.* 765:35–67.
- **Feehan, P. M. N. (2020).** On the Morse–Bott property of analytic functions on Banach spaces with Łojasiewicz exponent one half. *Calc. Var. PDE* 59:87. Morse–Bott ⇔ exponent ½. Used in Thm 4(ii).
- **Absil, P.-A., Mahony, R. & Andrews, B. (2005).** Convergence of the iterates of descent methods for analytic cost functions. *SIAM J. Optim.* 16(2):531–547. Single-limit convergence on analytic criteria. It excludes nothing on noncompact charts, which is why the faces are compactified.

**Optimization**
- **Nocedal, J. & Wright, S. J. (2006).** *Numerical Optimization*, 2nd ed., Springer. **Thm 3.3** (steepest-descent rate), **Thm 4.9** (global convergence of trust region to stationarity), **§4.3** (Moré–Sorensen). Used in Thm 5 and §6.
- **Akaike, H. (1959).** On a successive transformation of probability distribution and its application to the analysis of the optimum gradient method. *Ann. Inst. Statist. Math.* 11:1–16. Worst-case zig-zag of steepest descent. Used in Thm 5(a).
- **Moré, J. J. & Sorensen, D. C. (1983).** Computing a trust region step. *SIAM J. Sci. Stat. Comput.* 4(3):553–572.
- **Conn, A. R., Gould, N. I. M. & Toint, Ph. L. (2000).** *Trust-Region Methods.* SIAM. **Thm 6.4.6**: convergence to second-order critical points with an exact Hessian.
- **Ortega, J. M. & Rheinboldt, W. C. (1970).** *Iterative Solution of Nonlinear Equations in Several Variables.* Academic Press, **§12.6** (Newton–Kantorovich). Used via certificate-theory Theorem 1.
- **Hager, W. W. & Zhang, H. (2005).** A new conjugate gradient method with guaranteed descent and an efficient line search. *SIAM J. Optim.* 16(1):170–192. Approximate Wolfe conditions near the noise floor; context for §3.8.

**GAM smoothing-parameter uncertainty**
- **Wood, S. N., Pya, N. & Säfken, B. (2016).** Smoothing parameter and model selection for general smooth models. *JASA* 111(516):1548–1575. The corrected covariance J_ρΣ_ρJ_ρᵀ. Used in Prop. 7.
- **Wood, S. N. (2011).** Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B* 73(1):3–36. The REML structure. Context only.

---

## 6. Consequences for gamfit

File:line references are to `SP/main_src` at `0b0d0120c2`.

### 6.1 Diagnosis

| Mechanism | Where | SPEC class | Theorem | Action |
|---|---|---|---|---|
| Gradient-only override of an available exact Hessian | `drivers/spatial_optimization.rs:5468` (`.with_prefer_gradient_only(true)`), 1779–1792 (planner gating and comment), 1670 and 1864 (`suppress_outer_hessian_for_nfree`) | Fallback / planner forcing | Thm 5, §3.7 | **Delete** |
| ψ-dependent Frobenius normalization of Matérn penalties | `gam-terms/src/basis/matern_kernel.rs:3766` (`normalize_penalty_candidate`), 2341 (`normalize_penalty`); `workspace_cache.rs:373–400` (`build_matern_operator_penalty_candidates`) | Adds a shear; not a SPEC breach itself | Thm 1 | **Freeze at ψ₀** |
| ψ step cap ln 2; ρ cap 5 | `spatial_optimization.rs:6` (`SPATIAL_PSI_BFGS_STEP_CAP`), 1872, 5869; ρ `Some(5.0)` at 1870, 5867, 6587; uniform cap scaling in opt `lib.rs:8061–8088` | Caps | §3.7 (not the cause, but magic) | **Delete** |
| ρ box | `spatial_optimization.rs:40` (`joint_rho_resolvability_domain`, around `precision_box`) | Hand bound | Prop. 3; compactification lane | **Delete**; replace with faces |
| Cost-stall guard as a terminator | `rho_optimizer/bridges.rs:300` (`COST_STALL_WINDOW = 6`), 301 (`ARC_COST_STALL_WINDOW = 3`), 385 (`CostStallGuard`), 2208 (`fold_accepted_iterate`); `run_plan.rs:1581, 1863, 2477` (construction), 2839 (`BfgsCostStallExit`) | Magic window; non-certificate stop | Thm 5 | **Delete as a terminator** (at most telemetry) |
| Relative cost floor | `run.rs:8628` (`outer_rel_cost_floor`), 4581 (`objective_tol = floor·(1 + \|cost\|)`), `COST_STALL_REL_TOL_FLOOR` (`bridges.rs:302`) | Magic, unit-dependent | Thm 5; boundary-probability §3.6 | **Replace** with band_V and τ_stat |
| Decrement compared with objective_tol | `run.rs:6284` (`certify_interior_stationarity`) | Right quantity, wrong threshold | Thm 5(b) | **Keep Δ**; change the threshold (§6.4) |
| ψ tail-snap refusal and ratio band | `run.rs:6356` (`TAIL_SNAP_CURVATURE_BAND`), 6419 (`try_tail_snap_to_rail`), 6446–6455 (ψ refusal), 6492 onward; `ASYMPTOTE_PROBE_COUNT = 18` at 5757 | Magic band; probing | Thm 3; certificate-theory §6 | **Delete** the band and probes; build the Thm 3(a) face in s = κ |
| Outer covariance over railed coordinates | `gam-solve/src/estimate/smoothing_correction.rs:346, 2024` (`OuterHessianInverse`) | Threshold decision on noise | Prop. 7 | **Schur complement on the free block by rail status** |
| Criterion invariance with zero ψ component | `spatial_optimization.rs:2068` (`with_criterion_invariance`) | Correct | Thm 4(ii), (iv) | **Keep**; do not add a ψ component |
| Two-basin range selection | `spatial_optimization.rs:6753` (`select_isotropic_matern_range_basin`) | Comparison of two certified local fits; not a grid | Thm 4; §7 item 4 | Keep as-is. It is not a global certificate (open problem) |

### 6.2 Delete

1. **`spatial_optimization.rs:5468`**: remove `.with_prefer_gradient_only(true)` in `exact_joint_multistart_outer_problem` (5393).
   - Also remove the gating `analytic_outer_hessian_available && !suppress_outer_hessian_for_nfree` at 1779, and `suppress_outer_hessian_for_nfree` at 1670 and 1864, so that `capability::plan` selects the exact-Hessian arm whenever the terminal-analytic joint Hessian exists.
   - Delete the comment block at 1784–1792 that justifies BFGS first.
2. **Caps.** Remove `SPATIAL_PSI_BFGS_STEP_CAP` (`:6`) and its uses at 1872 and 5869. Remove the ρ `Some(5.0)` at 1870, 5867 and 6587, and the uniform cap rescaling in opt `lib.rs:8061–8088`.
   - With a trust region, the step length comes from the ratio test.
   - The radius update factors (for example ×2 and ×¼) affect speed only. They never enter a certificate, and any factors in (0, 1) and (1, ∞) keep global convergence (Conn–Gould–Toint 2000 Thm 6.4.6). So they are algorithm parameters, not tolerances.
3. **Box.** Remove `joint_rho_resolvability_domain` (`:40`) and the `precision_box` [ln ε, ln 1/ε] as the domain. Replace them with the faces of §6.3 step 0.
4. **Stall guard.** `CostStallGuard` (`bridges.rs:385`) must not terminate the solve. Delete `BfgsCostStallExit` (`run_plan.rs:2839`) as an outcome, the windows at `bridges.rs:300–301`, and the floor at 302. The guard may remain as telemetry.
5. **Relative floor.** Delete `outer_rel_cost_floor` (`run.rs:8628`) wherever it gates certification or stopping (4581, 6284). This matches boundary-probability C3.
6. **ψ tail-snap machinery.** Delete the curvature band (`run.rs:6356`), the ψ probes (5757) and the ratio tests (6492 onward). The refusal at 6446–6455 is correct but becomes unreachable, because ψ faces are handled analytically (§6.3 step 5).

### 6.3 Build: certified joint (ρ̃, ψ) solve (Algorithm M)

**Step 0: charts.**
- *Freeze the normalization.* In `normalize_penalty_candidate` (`matern_kernel.rs:3766`) and at the operator-candidate assembly (`workspace_cache.rs:373–400`), compute c_m once at the seed ψ₀ and store it with the term. For every ψ, use S_m(ψ) = S̃_m(ψ)/c_m(ψ₀).
  - The exact ψ-derivatives of S_m are then ∂^kS̃_m/∂ψ^k / c_m(ψ₀). The ℓ′ and ℓ″ terms disappear from the gradient and the Hessian (Thm 1(b), (c)).
- *Faces:*
  - upper: t_j = e^{−ρ̃_j} ≥ 0 (boundary-asymptotics R8; compactification Thm 1);
  - lower, when r_j = h_j: λ_j ≥ 0. For Matérn, r_j = 0 for every penalty contained in the range of a full-rank one (Prop. 3(b); boundary-probability C5);
  - ψ-lower face: s = κ ≥ 0 (Thm 3(a)).
  - All of these are domain constraints. None is a hand bound.

**Step 1: oracle.** Evaluate the exact V, g and H, including the ∂X/∂ψ and ∂S̃/∂ψ terms, with the existing analytic joint Hessian, together with band_V and ε_H from fp-error-analysis §3.4.

**Step 2: step.**
- Take a trust-region Newton step on the free face. Use a Moré–Sorensen subproblem in the norm ‖·‖_{D}, with D the diagonal of |H|, or equivalently an ARC step with exact H.
- Use the noise-aware ratio r = (ΔV + 2band_V)/(pred + 2band_V) (literature-survey §6.4; inexact-oracle lane).
- Negative curvature is used, not discarded. That is exactly the replica's ρ₀ tail direction.

**Step 3: active set.**
- A coordinate on a face is released iff its face KKT multiplier has the wrong sign beyond its band:
  - ∂V/∂λ_j|₀ < −band for lower faces;
  - ∂V/∂t_j|₀ < −band for upper faces;
  - ∂²V/∂s²|₀ < −band for the ψ-face when q = 2.
- A free coordinate is fixed to its face when the TR step reaches the face with multiplier sign consistent.
- This is finite identification (Nocedal–Wright §16.7). It needs no probing.

**Step 4: termination.** There are exactly three outcomes.
- **T1 (certified local minimum).** Δ_F + band_V ≤ τ_stat (boundary-probability (3.5)), certificate-theory Theorem 1 passes on the free block (h ≤ ½), and every face coordinate satisfies strict KKT beyond its band (certificate-theory Theorem 4).
  - With exact Newton, polishing from Δ ≈ τ_stat to Δ ≈ band_V costs one or two quadratic-rate steps. We recommend doing so, so that the reported fit is at the floating-point floor.
- **T2 (typed numerical failure).** band_V > τ_stat: the evaluation cannot resolve the statistics (boundary-probability C3).
- **T3 (typed derivative inconsistency).** The trust radius collapses to the point where pred ≤ 2band_V, yet Δ_F + band_V > τ_stat.
  - For C² V this cannot happen with correct derivatives: as the radius r → 0, pred → g·p = O(r) while the actual decrease matches to O(r²).
  - So it signals a gradient or Hessian bug, or an unmodelled kink, and must fail loudly. It is not retried.
- There is no iteration cap. With exact H on the compactified domain the sublevel sets are compact, and TR converges globally to a second-order point (Conn–Gould–Toint Thm 6.4.6). The local rate is quadratic.
- If ψ → +∞ persists (g_ψ < 0 with V decreasing as κ grows), the outcome is a typed "ψ → ∞ limit is not certifiable" (Thm 3(b)).

**Step 5: ψ face (future-proofing; not needed for the current failures, where ψ is interior at 0.57–0.79).**
- If the iterates approach ψ → −∞ with ∂_ψV > 0, switch ψ to s = κ.
- Certify at s = 0 with:
  - V₀(ρ̃), the limit REML, evaluated from the κ-series of X and S̃ at order 0;
  - v₂ = ½∂²_sV|₀ from the order-2 terms of the same series;
  - T1 on the (ρ̃, s) face with v₂ > band.
- Until the series oracle exists, the analytic check is ∂²_ψV/∂_ψV → q = 2, with deviation O(e^{Δψ}) between successive *exact* evaluations. It is exact, not a finite difference. The band 0.25–4 is not used.
- Candidates: the nearly linear truths (mega_batch_k sin t on [0, 1]; statsmodels x₃ linear).

**Step 6: covariance** (`smoothing_correction.rs:346, 2024`).
- Partition θ into certified-face coordinates A and free coordinates F.
- Form Σ_FF = (H_FF)⁻¹ by Cholesky, and use J_F Σ_FF J_Fᵀ.
- Coordinates in A contribute their analytic limit: 0 for q = 1 rails (Prop. 7), and the boundary-probability law otherwise.
- There is no eigenvalue thresholding and no deflation beyond published exact invariances (Thm 4(ii)).

### 6.4 The certificate

A joint Matérn fit is certified iff all of the following hold:

(i) The chart is ρ̃ (frozen normalization) and the faces are as in Step 0.

(ii) Δ_F = ½g_FᵀH_FF⁻¹g_F satisfies Δ_F + band_V ≤ τ_stat = 1/(2n_eff).

(iii) The H_FF-norm Kantorovich test holds, h = L_H·‖H_FF⁻¹g_F‖_{H_FF} ≤ ½, with L_H from certificate-theory Props. 3 and 4. Extending those bounds to ψ needs a ψ-analyticity radius (§7 item 2). Until it exists, the Hessian-change bound ‖H(θ + p) − H(θ)‖ over the final step, computed exactly, is a checked surrogate.

(iv) For every face coordinate, the KKT multiplier exceeds its band strictly.

(v) λ_min(Ĥ_FF) > ε_H, which is the Morse case (Thm 4(i)). If it fails on a subspace not explained by a published invariance, the outcome is the typed "unidentified along U" of Thm 4(iv), never a silent pass.

Every quantity is derived: band_V and ε_H from floating-point error, τ_stat from the statistical resolution, L_H analytically, and the bands of the face multipliers from fp-error-analysis Theorem 9.

### 6.5 How each tolerance is derived

| Tolerance | Derivation | Replaces |
|---|---|---|
| band_V (ε_V) | Cholesky componentwise bound (fp-error-analysis Thm 2), plus the D_p channel scaled by (n − M_p)/2, plus the penalty log-det (Thm 4 there). Replica scale 10⁻⁸ to 10⁻⁶. | `outer_rel_cost_floor`, `COST_STALL_REL_TOL_FLOOR` |
| ε_H | Weyl band of the computed Hessian (fp-error-analysis §3.4) | ad hoc curvature floors |
| τ_stat | 1/(2n_eff) (boundary-probability Prop. 6) | rel_tol·(1 + \|V\|) |
| Face bands | fp-error-analysis Thm 9 and boundary-asymptotics R12 | box edges, `TAIL_SNAP_CURVATURE_BAND` |
| q (ψ-face law) | Thm 3(a): the integer is structural (measured 2); no fit | "no exponential tail law" refusal |
| Trust-region factors | Not tolerances; they do not enter the certificate | caps 5 and ln 2 |

### 6.6 Failing clusters addressed

| Test (log line) | Cause here | Fix (§6.2 / §6.3 items) |
|---|---|---|
| mgcv_matern_smooth (15831): `BfgsCostStallExit`, 129 it, \|Pg\| 0.331 vs 0.0181, Δ = 0.347 | Gradient-only plan; raw-chart shear; ρ-box lower faces; magic τ (Thm 5, K_eff 8.8·10³) | Delete 1, 2, 3, 4, 5; Build 0, 2, 3, 4 |
| sklearn_gp_matern_regression (33534): \|Pg\| 2.79·10⁻² vs 1.51·10⁻², Δ = 5.97·10⁻³ | Same, K_eff 343; Δ = 1.8 τ_stat, so one or two Newton steps finish it | Delete 1, 4, 5; Build 0, 2, 4 |
| mega_batch_k::matern_with_explicit_centers (14600): \|Pg\| 3.62 vs 6.5·10⁻², 160 it, Δ = 0.912 | Same, K_eff 1.2·10⁴; lower face #0 in the ρ-chart | Delete 1–5; Build 0, 2, 3, 4 (and 5 if ψ → −∞ becomes active under the nearly linear truth) |
| statsmodels_gam_additive (15518): line_search_failed MaxAttempts, rails ±, Δ = 1.46·10⁻² | Projected-kink Wolfe failure (not fp: 2Δ ≫ ε_V) | Delete 1, 3; Build 0 (faces), 2 (TR, no bracket), 3, 4 |
| inla_tensor_product_spde (15629): `OuterHessianInverse` negative curvature −8.66·10⁻⁸ vs bar 6.05·10⁻⁸ | Covariance inverted over a certified rail (Prop. 7) | Build 6 |

---

## 7. Open problems

1. **Prove q = 2 for ν = 5/2** (v₁ = 0 in Thm 3(a)), and find q for general half-integer ν. The measured value is 2 in five configurations, and the partial argument of §3.4 reduces it to showing that the even κ²-part of the matrix series sets the leading orders of det A, det S̃_λ and D_p. For non-half-integer ν, the Gu–Wang–Berger log and fractional terms make the face variable non-analytic.
2. **A computable radius and remainder for the κ-series**, i.e. a ψ-analyticity radius δ_ψ for certificate-theory Proposition 4. This is needed for a rigorous L_H in the ψ direction and for turning Step 5 into a finite-order certificate. The kernel entries are entire in κ, but the radius of log D_p and the log-determinants is set by the nearest complex κ where A is singular or D_p = 0. That needs a Rouché-type bound.
3. **Certification at κ → ∞.** V has terms beyond all orders in 1/κ (Thm 3(b)). Is there a data-dependent change of variable, for example s = e^{−√5κd_min}, that makes the end analytic? Or can one prove from the design alone that this end is never a minimizer?
4. **Global multimodality in ψ without grid search.** The replica has two certified basins with ΔV = 7.15. `select_isotropic_matern_range_basin` compares two seeds, and nothing certifies that no third basin exists. Candidate approaches are homotopy in the data (y → t·y) or in κ, and interval bounds on V*(ψ) from a Lipschitz bound on the profile. Both are open.
5. **Preasymptotic rates.** The measured slopes of 0.29–0.51 are far above the asymptotic 1/6. A finite-k analogue of the equivalence of measures for the basis-plus-operator-penalty model would give cond(n, κ, k) directly and replace the GP heuristic of Prop. 2(c).
6. **Faces that depend on κ.** The ranks of S̃_m, and hence r_j − h_j in boundary-probability Proposition 5, change as κ → 0, since c_m ~ κ^{p_m}. A joint (λ, κ) corner theory is needed if both a lower λ-face and the κ = 0 face are active at once.
7. **Non-Morse–Bott degeneracies** (Thm 4(iv)). Beyond the fit-level certificate, there is no finite-precision certificate of minimality when the degeneracy is not an exact, published invariance. Is there a practical class, such as analytic V with known Łojasiewicz exponent θ > ½, for which a certified bound on V − V* follows from ‖g‖ and θ?
