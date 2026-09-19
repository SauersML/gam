# Certifying a local minimizer of the outer criterion V(ρ) from inexact analytic data

Lane: `certificate-theory` (convergence theory team).
Scripts: `SP/theory/certificate-theory/{reml_toy.py, check_derivs.py, check_certificates.py, check_classifier.py, check_smale.py}` (SP = the session scratchpad). All numerics below were run in `SP/theory/venv`.
Code references are to the read-only checkout `SP/main_src` (crate `gam-solve`, directory `crates/gam-solve/src/rho_optimizer/`, abbreviated `ro/`), and to the `opt` crate at `/root/.cargo/git/checkouts/opt-4a38fa79856f3ac9/53ce029`.

Every claim is tagged as **[proven]** (a proof or derivation is given here, or it is a cited theorem applied under verified hypotheses), **[checked]** (verified numerically on the toy REML problem) or **[conjectured]**.

---

## 1. Summary

1. **What a certificate must assert.** A certificate is a *finite, a-priori* test on computed quantities (Ṽ, g̃, H̃, and one extra bound) whose truth implies the following: there is a **unique** critical point ρ\* within radius r₋ of the returned point, it is a **strict** local minimizer (H(ρ\*) ≻ 0), and r₋ is reported in **posterior-sd units** (the H-norm).
   - The current terminal test, the Newton-decrement verdict ½λ̂² + band_λ² ≤ band_f (`ro/decrement_bands.rs:166`, `opt` `newton_decrement_verdict`), does **not** assert this. Small λ at a PD Hessian does not imply a nearby minimizer.
   - On every exponential tail, λ → 0 with no minimizer at finite ρ. On the toy tail, λ² = 6·10⁻¹² < 2·band_f at ρ₄ = 34.5, so the verdict would certify a point whose minimizer is at ρ₄ = ∞ **[proven + checked]**.
2. **The strongest cheap certificate is an H-norm (affine-invariant) Newton–Kantorovich test.** The test is h = L_H·(λ̂+δ_λ)/(1−η)² ≤ ½, where L_H is a Lipschitz constant of the Hessian measured in the H̃-norm over the ball of radius 2λ̂.
   - When it holds: a unique minimizer exists within r₋ = (μ − √(μ² − 2L_Hλ))/L_H, it is strict, Newton converges quadratically from the point, and the step budget follows from a derived scalar recursion **[proven]**.
   - Theorem 1 (§3.3) states it.
3. **L_H can be bounded analytically, with no FD and no autodiff, for Gaussian REML. Three results make this possible:**
   - **(a) Uniform third-derivative bound [proven, checked].** |D³V[u,u,u]| ≤ K‖u‖∞³ with K = 3τ + 3r_ov + ½(13π + 9π²/ν + 2π³/ν²), where τ = tr(H⁻¹S_λ) and π = νP/D_p (Prop. 3).
   - **(b) Holomorphy [proven].** V extends holomorphically to the polydisc |z_j| ≤ ln(3/2) around ρ̃ and is bounded there by M = ½τ ln 2 + ½r_ov ln 2 + (ν/2)ln(1 + P/RSS) (Prop. 4). The Cauchy estimates then bound every higher derivative.
   - **(c) Banach polarization in the H̃-norm (a Hilbert norm).** It turns diagonal bounds into operator-norm bounds for free (Lemma 2).
   - **Recommended computable form.** L_H = ‖T̂(ρ̃)‖_F + R·24M s⁴/(δ − sR)⁴, where T̂ is the exact third tensor normalized by the Cholesky factor of H̃⁻¹ and s is the largest marginal posterior sd of a ρ_j.
   - **Result on the toy.** It certifies at 10⁻³ sd from the minimizer (h = 0.20; r₋ = 1.13·10⁻³ against a true distance of 1.00·10⁻³). The uniform bound (a) alone is about 650× looser **[checked]**.
4. **The other frameworks, compared:**
   - **Smale's α-theory** needs only point data (g, H) plus M, not T. The test is α = λ·γ ≤ α₀ = (13−3√17)/4 ≈ 0.158, with γ from the Cauchy bound. It is weaker: it certifies only at ≲ 6·10⁻⁵ sd on the toy. That is still enough at the polish target, since ½λ² ≤ band_f ⇒ λ ≲ 10⁻⁵ **[proven, checked]**. Smale is therefore the natural certificate where T is expensive (LAML), once M is available.
   - **Krawczyk / interval Newton** is the same test in a box norm and adds nothing here **[proven]**.
   - **Self-concordance does not hold for V in ρ**: the self-concordance constant is M_eff = 1/(2λ) → ∞ on every exponential tail. The polish budget (`ro/newton_polish.rs:357-380`) assumes Boyd–Vandenberghe's M = 1, and that assumption is invalid **[proven]**.
   - **SOSC plus a Lipschitz-Hessian ball argument** needs Lλ/μ² < 3/8, which is strictly weaker than Kantorovich's ½ **[proven]**.
5. **Boundary faces (ρ_j = +∞).** In the compactified coordinate t_j = e^{−ρ_j}, Gaussian REML is **real-analytic up to and across t_j = 0**, and ∂_{t_j}V(0) = c_j, the analytic pencil constant (Prop. 5) **[proven, checked to 7 digits]**.
   - A face point is certified by KKT with strict complementarity plus SOSC on the free face. The test is c_j − δ_c − r·G_c > 0 together with the reduced Kantorovich test (Theorem 4) **[proven]**.
   - With no box, such a face is identified in finitely many projected steps in t.
6. **Finding: `rail_face.rs` refuses valid faces.** `certify_rail_face` requires λ_min(C) > margin (`ro/rail_face.rs:369-383`). That condition is **sufficient, not necessary**. For disjoint penalty ranges the exact first-order condition is c_j > 0 for every j in the face. For overlapping ranges it is min over the simplex of ½tr(S(d)⁺C) > 0.
   - **Counterexample [checked]:** a single-penalty face with eig(C) ∋ −1710 has c = 5687 > 0 and is the true optimum. The refusal text "releasing the face would not raise the criterion" is false there.
7. **Degenerate minima.**
   - Exact structural degeneracy is Morse–Bott, for example linearly dependent penalties. On the toy, V − V\* = ½λ_⊥²(1 + O(dist)), which is Łojasiewicz exponent ½ (Feehan 2020) **[checked]**.
   - The fix is to quotient it out *before* optimizing, by the rank of the Gram matrix ⟨S_j, S_k⟩_F. Near-degeneracy (small μ) is not a separate case: the H-norm certificate absorbs it through s and μ.
   - Non-Morse–Bott degenerate critical points cannot be certified from finite-order point data. This is conjectured in general and is an open problem.
8. **An exact regime classifier [proven, checked].** h₀ = |D³V[p,p,p]|/λ² along the Newton step is affine-invariant. It equals:
   - → 0 at a Morse minimum;
   - exactly 1 on any exponential tail;
   - (k+2)/(k+1) on an algebraic tail ρ^{−k};
   - exactly 2 on a logarithmic ramp.

   The step observables (ΔV/λ², λ₊²/λ²) are (0.632, 0.368) on exponential tails, (ln 2, 1) on logarithmic ramps, and (½, → 0) in the quadratic regime.
9. **The multinomial failure** ("½λ̂² = 4.7·10⁻⁵ > band_f = 3.95·10⁻¹¹ after 2 of 2 steps; the decrement stopped contracting"):
   - band_f is derived (it is roundoff resolution), and the refusal to certify is correct.
   - The *stop* is not derived. The observed step (ΔV/λ² = 0.65, λ² ratio 1.027) is outside every quadratic regime (h₀ ≳ 2). The point is neither noise-limited (ΔV ≈ 6·10⁻⁵ ≈ 10⁶·band_f) nor Morse.
   - Reading: a weakly identified direction (curvature ≈ 0.023, about 6.6 sd) far from the Kantorovich region, **[conjectured]** heading to a face or a flat valley.
   - The budget of 2 came from the invalid self-concordance formula.
   - Fix: keep descending while each step's decrease is resolvable (> band_f). Certify by Theorem 1 or Theorem 4 when their tests pass. Delete the contraction stop.
10. **One unified certificate replaces the ladder** (§6.2):
    - (i) a structural Morse–Bott quotient;
    - (ii) descent in compactified coordinates while the decrease is resolvable;
    - (iii) a free-face H-norm Kantorovich certificate, or Smale where only M is available;
    - (iv) face KKT with analytic c_j;
    - (v) polish to band_f with the Kantorovich step recursion;
    - (vi) report r₋ in posterior sd.

    All tolerances in it are band_f, band_λ, η and δ_c, which are floating-point bands, plus the analytic constants L_H, M and G_c. There are no hand constants.

---

## 2. Setup and notation

- **Coordinates.** ρ ∈ ℝ^q are the outer coordinates: log smoothing parameters ρ_j = log λ_j, and possibly other hyperparameters ψ. S_λ = Σ_j e^{ρ_j}S_j with S_j ⪰ 0, and A_j := e^{ρ_j}S_j.
- **Gaussian profiled REML** (the model the proofs cover exactly):
  - V(ρ) = ½ log|H_β| − ½ log|S_λ|₊ + (ν/2) log D_p(ρ);
  - H_β = XᵀX + S_λ, D_p = yᵀy − bᵀH_β⁻¹b = RSS + P with b = Xᵀy;
  - P = β̂ᵀS_λβ̂ and ν = n − M_p, where M_p = dim null S_λ is constant in ρ.
  - LAML replaces XᵀX with XᵀW(β̂)X and D_p with the penalized deviance; §7 lists what changes.
- **Derivatives.** g = ∇V, H = ∇²V, T = D³V. Tilde quantities (Ṽ, g̃, H̃) are computed in floating point.
- **Bands** (as in `ro/decrement_bands.rs:67-143`):
  - |Ṽ − V| ≤ band_f;
  - ‖g̃ − g‖_{H̃⁻¹} ≤ δ_λ, which is band_λ;
  - ‖H̃^{-1/2}(H(ρ̃) − H̃)H̃^{-1/2}‖₂ ≤ η (Weyl, from `hessian: growth·‖H̃‖_F`, line 141, divided by λ_min(H̃)).
- **Local norm.** ‖u‖ := ‖u‖_{H̃} = √(uᵀH̃u), with dual ‖g‖\* = √(gᵀH̃⁻¹g). Because H̃ ≈ H is the Laplace precision of ρ, ‖u‖ is a distance **in posterior standard deviations**.
- **Newton quantities.** p = −H̃⁻¹g̃ is the Newton step and λ̂ = ‖g̃‖\* = √(g̃ᵀH̃⁻¹g̃) is the decrement.
- **Marginal sd.** s := max_j √((H̃⁻¹)_jj) is the largest marginal posterior sd of a coordinate. It satisfies ‖u‖∞ ≤ s‖u‖ (Cauchy–Schwarz).
- **Normalized penalty matrices.** B_j := H_β^{-1/2}A_jH_β^{-1/2} ⪰ 0 with Σ_jB_j ⪯ I, since H_β ⪰ S_λ. Write τ := tr(H_β⁻¹S_λ) = Σ_j tr B_j, the penalty's effective degrees of freedom, with τ ≤ rank S_λ. Also π := νP/D_p, and r_ov is the rank of the part of S_λ on which penalties overlap (0 for block-disjoint penalties).
- **Face coordinates.** t_j = e^{−ρ_j} ∈ [0,∞) is the compactified coordinate of a smoothing parameter. The constraint t_j ≥ 0 is the domain constraint λ_j ∈ (0, ∞], not a box.
- **Normalized third tensor.** T̂ = T[L·, L·, L·], where LLᵀ = H̃⁻¹. It is the third tensor in H̃-orthonormal coordinates.

### Definition (certificate)

A *certificate at ρ̃* is a predicate on computed data such that, if it evaluates to true, then under exact arithmetic within the stated bands:

- **(C1)** there is ρ\* with ∇V(ρ\*) = 0 and ‖ρ\* − ρ̃‖ ≤ r₋;
- **(C2)** ρ\* is the only critical point in ‖ρ − ρ̃‖ < r₊;
- **(C3)** H(ρ\*) ≻ 0, so ρ\* is a strict local minimizer.

A *face certificate* replaces (C1)–(C3) with the KKT analogues in the compactified coordinates.

---

## 3. Results

### 3.1 Why the decrement verdict is not a certificate

**Proposition 1 [proven, checked].** Let V(ρ) = V_∞ + c·e^{−kρ} + (terms independent of ρ) in some direction, with c, k > 0. Then:

- The pure Newton step is Δρ = 1/k.
- λ² = c·e^{−kρ}, so λ₊²/λ² = e^{−1}.
- ΔV/λ² = 1 − e^{−1}.
- h₀ := |V'''|·|p|³/λ² = 1.

All four hold for every k and c. The verdict ½λ² + band_λ² ≤ band_f is eventually satisfied, at ρ ≈ ρ₀ + ln(λ₀²/(2 band_f))/k, although no critical point exists at finite ρ.

*Proof.* Take V' = −kce^{−kρ}, V'' = k²ce^{−kρ} and V''' = −k³ce^{−kρ}. Then p = −V'/V'' = 1/k and λ² = V'²/V'' = ce^{−kρ}. The value drop over one step is ce^{−kρ}(1 − e^{−1}). Finally h₀ = k³ce^{−kρ}k^{−3}/(ce^{−kρ}) = 1. ∎

**Numerical check.** On the toy REML tail (`check_certificates.py`, Check 3, with x3_effect none and linear), every step has ΔV/λ² = 0.632, a λ² ratio of 0.3679 and t = 1.

- At iteration 18, λ² = 5.97·10⁻¹², so ½λ² = 3·10⁻¹² < 3.95·10⁻¹¹. The decrement rung would certify it.
- The limit is the face ρ₄ = ∞. For the coupled 'linear' case it is the two-coordinate face ρ₃, ρ₄ = ∞.

A certificate therefore needs one more piece of information beyond (g, H): a bound on how H varies. That is the content of every theorem below.

### 3.2 Two analytic bounds on higher derivatives (Gaussian REML)

**Lemma 1 (derivative identities) [proven].** Write B(u) := Σ_j u_jB_j and B(u^k) := Σ_j u_j^kB_j, evaluated at the current point. Since ∂_jA_k = δ_jkA_k:

- D log|H_β|[u] = tr B(u)
- D² log|H_β|[u,u] = tr B(u²) − tr B(u)²
- D³ log|H_β|[u³] = tr B(u³) − 3 tr(B(u)B(u²)) + 2 tr B(u)³
- D⁴ log|H_β|[u⁴] = tr B(u⁴) − 4tr(B(u)B(u³)) − 3tr B(u²)² + 12 tr(B(u)²B(u²)) − 6 tr B(u)⁴

For the data term, let a(v) := β̂ᵀA(v)β̂, m(u,v) := β̂ᵀA(u)H_β⁻¹A(v)β̂ and q(u) := β̂ᵀA(u)H_β⁻¹A(u)H_β⁻¹A(u)β̂. Then:

- D_p' = a(u), by the envelope theorem
- D_p'' = a(u²) − 2m(u,u)
- D_p''' = a(u³) − 6m(u²,u) + 6q(u)

*Proof.* Differentiate H_β⁻¹ with d(H⁻¹) = −H⁻¹(dH)H⁻¹, using dA(v)[u] = A(uv). ∎

These are implemented exactly in `reml_toy.py:11-71`. The check against high-precision FD (test-only) gives relative errors of 1.1·10⁻⁹ for g, 4.5·10⁻¹⁰ for H and 6.1·10⁻¹⁰ for T over 20 random ρ **[checked]**.

**Proposition 3 (uniform third-derivative bound) [proven, checked].** For Gaussian REML, at every ρ:

  |D³V(ρ)[u,u,u]| ≤ K(ρ)·‖u‖∞³,   K = 3τ + 3r_ov + ½(13π + 9π²/ν + 2π³/ν²).

*Proof.*

1. Since B_j ⪰ 0 and ΣB_j ⪯ I, ‖B(u)‖₂ ≤ ‖u‖∞ and tr|B(u^k)| ≤ ‖u‖∞^k τ. Apply this term by term in Lemma 1:
   - |tr B(u³)| ≤ τ‖u‖∞³;
   - |tr(B(u)B(u²))| ≤ ‖B(u)‖·tr B(|u|²) ≤ τ‖u‖∞³;
   - |tr B(u)³| ≤ ‖B(u)‖²·tr|B(u)| ≤ τ‖u‖∞³.

   So |D³ ½log|H_β|| ≤ ½(1+3+2)τ‖u‖∞³ = 3τ‖u‖∞³.
2. For block-disjoint penalties, −½log|S_λ|₊ = −½Σ r_jρ_j + const is linear, so its third derivative vanishes. On overlapping blocks the same argument with B_j^S (for which ΣB_j^S = I on the range) gives ≤ 3r_ov.
3. Data term. By Cauchy–Schwarz on A_j ⪰ 0: a(v) ≤ ‖v‖∞P, m(u,v) ≤ ‖u‖∞‖v‖∞P and q(u) ≤ ‖u‖∞³P. Hence |D_p'| ≤ P‖u‖∞, |D_p''| ≤ 3P‖u‖∞² and |D_p'''| ≤ 13P‖u‖∞³.
4. Write (log D)''' = D'''/D − 3D''D'/D² + 2D'³/D³. Multiplying by ν/2 gives ½(13π + 9π²/ν + 2π³/ν²). ∎

Numerically, the maximum over 4000 random directions of |T[u³]|/(K‖u‖∞³) is 0.055 ≤ 1 (`check_derivs.py`) **[checked]**.

Along a ball of radius R in ‖·‖∞, the constants inflate at most as τ ≤ τ(ρ̃)e^{2R} and π ≤ π(ρ̃)e^{4R}, and τ ≤ rank S always holds. Prop. 3 is valid everywhere, but in the H̃-norm it becomes K s³, which is large when a coordinate is weakly identified (s large). This is why Prop. 4 plus the exact T̂ is used instead.

**Proposition 4 (holomorphic extension and Cauchy bound) [proven].** Let δ := ln(3/2). For Gaussian REML with RSS > 0, V extends holomorphically to the polydisc Δ = {ρ̃ + z : |z_j| ≤ δ}. On it,

  |V(ρ̃+z) − V(ρ̃) − ℓ(z)| ≤ M := ½τ ln 2 + ½ r_ov ln 2 + (ν/2) ln(1 + P/RSS),

where ℓ is the linear part of −½log|S_λ|₊ (it is exactly linear for disjoint penalties) and τ, P, RSS are evaluated at ρ̃. Consequently, for real x with ‖x − ρ̃‖∞ = σ < δ and every k ≥ 2,

  |D^kV(x)[u^k]| ≤ k!·M·‖u‖∞^k / (δ − σ)^k.

*Proof.*

1. **Log-determinant part.** H_β(ρ̃+z) = H̃_β^{1/2}(I + E)H̃_β^{1/2} with E = Σ_j(e^{z_j} − 1)B_j. For |z_j| ≤ δ, |e^{z_j} − 1| ≤ e^{δ} − 1 = ½. For unit x, y, Cauchy–Schwarz on B_j ⪰ 0 gives
   |xᴴEy| ≤ ½Σ_j(xᴴB_jx)^{1/2}(yᴴB_jy)^{1/2} ≤ ½(Σ xᴴB_jx)^{1/2}(Σ yᴴB_jy)^{1/2} ≤ ½.
   So ‖E‖ ≤ ½, and log det(I+E) = Σ_i log(1+e_i) is holomorphic, where e_i are the eigenvalues of E, with |e_i| ≤ ½. Using |log(1+e)| ≤ −log(1−|e|) ≤ 2 ln 2·|e| for |e| ≤ ½, together with Σ|e_i| ≤ ‖E‖_\* ≤ Σ_j|e^{z_j} − 1| tr B_j ≤ ½τ, gives |½ log det(I+E)| ≤ ½τ ln 2.
2. **Data part.** With w = H̃_β^{1/2}β̂ we have D_p(z) − D_p = wᵀ(I+E)⁻¹Ew = wᵀEw − wᵀE(I+E)⁻¹Ew. Here |wᵀEw| ≤ ½P, and ‖Ew‖ ≤ ½√P gives |wᵀE(I+E)⁻¹Ew| ≤ 2·P/4. So |D_p(z) − D_p| ≤ P < D_p = RSS + P, which means D_p(z) ≠ 0 and
   |log D_p(z) − log D_p| ≤ −log(1 − P/D_p) = ln(1 + P/RSS).
3. **Overlapping part.** The overlapping |S_λ|₊ part is the same argument as step 1 with ΣB_j^S = I.
4. **Cauchy.** Apply the one-variable Cauchy estimate to ζ ↦ V(x+ζu) − V(ρ̃) − ℓ(x − ρ̃ + ζu) on |ζ| ≤ (δ−σ)/‖u‖∞. That disc stays inside Δ, and for k ≥ 2 the subtracted terms contribute nothing. ∎

**Lemma 2 (polarization in a Hilbert norm) [proven; Banach 1938].** For a symmetric k-linear form A on a real Hilbert space, sup_{‖u_i‖ ≤ 1}|A[u_1,…,u_k]| = sup_{‖u‖ ≤ 1}|A[u,…,u]|. Because ‖·‖_{H̃} is a Hilbert norm, and ‖u‖∞ ≤ s‖u‖, Prop. 4 gives the operator-norm bound

  ‖D^kV(x)‖_{H̃} ≤ k!·M·s^k/(δ − σ)^k.

In particular, the Hessian is Lipschitz on the H̃-ball of radius R (with sR < δ) with constant

  **L_H(R) := ‖T̂(ρ̃)‖_F + R·24·M·s⁴/(δ − sR)⁴.**   (★)

This uses ‖T(ρ̃)‖_op ≤ ‖T̂‖_F, and sup_ball‖T‖ ≤ ‖T(ρ̃)‖ + R·sup_ball‖D⁴V‖.

### 3.3 Theorem 1: H-norm Newton–Kantorovich certificate

**Theorem 1 [proven; Kantorovich 1948, Ortega–Rheinboldt 1970 §12.6, Deuflhard 2004 Thm 2.1 (affine-covariant form)].** Assume the following:

- H̃ ≻ 0 is symmetric;
- ‖H̃^{-1/2}(H(ρ̃) − H̃)H̃^{-1/2}‖ ≤ η < 1, and set μ := 1 − η;
- ‖g̃ − g(ρ̃)‖\* ≤ δ_λ, and set λ := λ̂ + δ_λ;
- R := 2λ/μ, and L := L_H(R) from (★), with sR < δ.

If

  **h := L·λ/μ² ≤ ½,**

then:

- **(C1)** There is a critical point ρ\* with ‖ρ\* − ρ̃‖ ≤ r₋ := (μ − √(μ² − 2Lλ))/L ≤ R.
- **(C2)** It is the only critical point in the ball of radius min(r₊, R), where r₊ := (μ + √(μ² − 2Lλ))/L.
- **(C3)** H(ρ\*) ⪰ √(μ² − 2Lλ)·H̃ ≻ 0 when h < ½. So ρ\* is a strict local minimizer.
- **(C4)** Newton from ρ̃ converges to ρ\*, with ‖ρ_k − ρ\*‖ ≤ (2h)^{2^k}·λ/(2^k·μ·h) for h < ½ (Kantorovich error estimate).

*Proof.* Work in coordinates v = H̃^{1/2}ρ, in which ‖·‖ is Euclidean.

1. **Bounds at ρ̃.** Let F := ∇V. Then ‖F'(ρ̃)⁻¹‖ ≤ 1/μ by Weyl, and the Newton step satisfies ‖F'(ρ̃)⁻¹F(ρ̃)‖ ≤ λ/μ. F' is L-Lipschitz on the ball of radius R by Lemma 2.
2. **Kantorovich.** Apply Kantorovich's theorem (Ortega–Rheinboldt 12.6.2) with β = 1/μ, η_K = λ/μ and K = L. The condition is h_K = βKη_K = Lλ/μ² ≤ ½. It gives existence in t\* = (1 − √(1 − 2h))/(βK) = r₋ and uniqueness in t\*\* = r₊. Since t\* = 2η_K/(1 + √(1−2h)) ≤ 2η_K = R, the ball on which L was bounded contains t\*. This is self-consistent, because L was computed for exactly this R.
3. **Positive definiteness at ρ\*.** H(ρ\*) ⪰ H(ρ̃) − L·r₋·H̃ ⪰ (μ − Lr₋)H̃ = √(μ² − 2Lλ)·H̃.
4. **Minimality.** A critical point with PD Hessian is a strict local minimizer (SOSC; Nocedal–Wright 2006, Thm 2.4).
5. **Convergence rate.** (C4) is the classical majorant estimate. ∎

**Corollary 1 (derived contraction and step budget) [proven].** Take η = 0 (the Hessian at the new point is recomputed) and h = Lλ ≤ ½. Then the next decrement satisfies

  λ₊ ≤ Lλ²/(2√(1 − Lλ)),  hence  λ₊²/λ² ≤ h²/(4(1−h)) ≤ 1/8.

*Proof.* g(ρ₊) = ∫₀¹(H(ρ̃+sp) − H(ρ̃))p ds, so ‖g(ρ₊)‖\* ≤ L‖p‖²/2 = Lλ²/2. Also H(ρ₊) ⪰ (1 − Lλ)H̃, so the new dual norm inflates by at most (1 − Lλ)^{-1/2}. ∎

Iterating θ_{k+1} = θ_k²/(2√(1−θ_k)) with θ₀ = h gives the *derived* Newton budget: the smallest k with ½(θ_k/L)² + band_λ² ≤ band_f. This is a scalar loop. It replaces the self-concordance budget, and a failure to contract by the factor h²/(4(1−h)) while h ≤ ½ is a proven inconsistency of the evaluation (a bug signal), not a property of the problem.

**Numerical check (Check 2; interior case, x3_effect 'strong').** ρ\* = (4.23, 7.82, −2.14, 4.27), with eig H = (0.27, 0.30, 1.74, 1.97).

| distance from ρ\* (sd) | λ | L_H (★) | h | verdict | r₋ (true distance) |
|---|---|---|---|---|---|
| 1, 0.5, 0.3 | — | ∞ (sR ≥ δ) | ∞ | refused | — |
| 0.1 | 0.100 | 3.0·10⁹ | 3·10⁸ | refused | — |
| 0.03 | 0.030 | 2.2·10⁴ | 670 | refused | — |
| 10⁻³ | 1.0·10⁻³ | 201 | 0.201 | **certified** | 1.128·10⁻³ (1.000·10⁻³) |
| 10⁻⁶ | 1.0·10⁻⁶ | 0.89 | 9·10⁻⁷ | **certified** | 1.004·10⁻⁶ (1.000·10⁻⁶) |

On this problem ‖T̂‖_F = 0.70, whereas the uniform bound gives K·s³ = 651. Newton from 0.02 sd produces λ = 2.0·10⁻², 5.4·10⁻⁵, 4.6·10⁻¹⁰, 4·10⁻¹³, which is quadratic.

The Cauchy fourth-order remainder dominates L_H for λ ≳ 10⁻²: the certificate is conservative by about one Newton step, not more. Because convergence is quadratic, looseness of L by a factor c costs about log₂(1 + log c/log(1/h)) extra steps. The true local L here is O(1), so the certified region is loose, but the cost is ≤ 1 step **[checked]**.

### 3.4 Theorem 2: Smale's α-test with an analytic γ

**Theorem 2 [proven; Smale 1986, Blum–Cucker–Shub–Smale 1998 Ch. 8].** Let:

- β := λ/μ;
- γ := sup_{k≥2}‖F'(ρ̃)⁻¹D^kF(ρ̃)/k!‖^{1/(k−1)}.

If α := βγ ≤ α₀ = (13 − 3√17)/4 ≈ 0.1577, then Newton from ρ̃ converges quadratically to a zero ρ\* with ‖ρ\* − ρ̃‖ ≤ 2β.

**Point-only bound for γ [proven].** With F = ∇V, D^kF = D^{k+1}V. By Prop. 4 and Lemma 2 at σ = 0:

  γ ≤ sup_{k≥2} [ (k+1)·M·s^{k+1} / (μ·δ^{k+1}) ]^{1/(k−1)}.

This needs only g̃, H̃ and the scalar M. **No third tensor is needed.** Strict minimality follows as in Theorem 1 (C3), using the Lipschitz bound on the 2β ball.

**Numerical check (Check 7, `check_smale.py`).** γ = 2.5·10³, so the test certifies at 10⁻⁶ sd (α = 2.5·10⁻³) but not at 10⁻⁴ sd (α = 0.25). Kantorovich with the exact T̂ certifies from 10⁻³ sd. At the polish target, ½λ² ≤ band_f ≈ 4·10⁻¹¹ ⇒ λ ≤ 9·10⁻⁶, so α ≈ 0.02 there. **Smale therefore suffices wherever Newton has already polished to band_f** **[checked]**. That is the practical route for LAML, where T needs fifth likelihood derivatives but M might be bounded more cheaply (§7).

### 3.5 Krawczyk / interval Newton

**Proposition (ball Krawczyk) [proven; Krawczyk 1969, Moore 1977, Rump 2010 §§8, 13].** Take the operator K(X) = ρ̃ − Yg̃ + (I − YH(X))(X − ρ̃) with Y = H̃⁻¹ on the H̃-ball X of radius R. The inclusion K(X) ⊂ int X holds if ‖Yg̃‖ + δ_λ + (η + L R)R < R. When it holds, X contains a unique zero.

Minimizing over R gives exactly (1 − η)² > 2Lλ, i.e. **the same test as Theorem 1**, with no extra power.

The interval (box) version with ‖·‖∞ is strictly weaker here. The box would have to contain the weakly identified directions, and those are exactly the ones inflated by s. For this problem Krawczyk offers no advantage over Theorem 1. Its value, rigorous rounding by interval arithmetic, is already provided by the band analysis in `decrement_bands.rs`.

### 3.6 Self-concordance

**Proposition 2 [proven].** V is not self-concordant in ρ for any constant. On the exponential tail of Prop. 1,

  |V'''|/(V'')^{3/2} = 1/√(c e^{−kρ}) = 1/λ → ∞.

So M_eff = 1/(2λ). On an algebraic tail the ratio is (k+2)/((k+1)λ). Moreover, V is not convex on ℝ^q in general: the penguins cluster has λ_min(H) = −23. Self-concordance (Nesterov–Nemirovski 1994, Def. 2.1.1; Nesterov 2018 §5.1) is a convex-analysis notion and does not apply.

**Consequence.** The budget in `ro/newton_polish.rs:357-380` uses λ₊ ≤ 2λ² for λ ≤ ¼ (Boyd–Vandenberghe 2004, §9.6.3, which assumes M = 1). That is precisely Theorem 1 with an **assumed** L_H = 2 in the H-norm.

- On a Morse minimum the true L can be larger or smaller.
- On a tail L_H·λ = h₀ = 1 identically, so the inequality λ₊ ≤ 2λ² is simply false: λ₊ = e^{−½}λ.

In the multinomial failure, the formula gave ratio = ln(8·3.95·10⁻¹¹)/ln(2·0.0120) = 5.87, hence ⌈log₂ 5.87 − 1⌉ = **2 steps**. That is the "2 of 2" in the log, and it is an assumption, not a derivation.

### 3.7 SOSC with a Lipschitz Hessian (ball argument)

**Proposition [proven].** Suppose H(ρ̃) ⪰ μH̃, ‖g‖\* ≤ λ and the Hessian is L-Lipschitz on the ball. Then

  V(ρ̃+u) ≥ V(ρ̃) − λ‖u‖ + ½μ‖u‖² − (L/6)‖u‖³.

If λ < 3μ²/(8L), the right-hand side exceeds V(ρ̃) on the sphere of radius r = 3μ/(2L) (where ½μr − Lr²/6 attains its maximum 3μ²/(8L)). A local minimizer therefore exists inside the ball, by compactness.

This needs Lλ/μ² < 3/8, which is **strictly stronger** than Kantorovich's ½, and it gives no uniqueness. It is dominated by Theorem 1 and is listed only for completeness.

### 3.8 Boundary faces: the compactified KKT certificate

**Proposition 5 (analytic compactification) [proven for Gaussian REML with block-disjoint penalties; checked].**

*Setup.* Fix a face coordinate j, and let Q be an orthonormal basis of range S_j and Z of its complement. Write S_j = QΣQᵀ with Σ ≻ 0, and K := XᵀX + S_{rest}.

*Claim.* With t = e^{−ρ_j}, the function Ṽ(t, ρ_F) := V(ρ_F, −log t) extends real-analytically to a neighbourhood of t = 0, including t < 0. Its value at 0 is the limit-model criterion V_∞(ρ_F) (as `ro/rail_face.rs:30-42` states), and

  ∂_tṼ(0, ρ_F) = c_j = ½ tr(Σ⁻¹[Schur_Z(K) − g_Qg_Qᵀ/φ̂]),

where φ̂ = D_face/ν and g_Q = Qᵀ(b − Kβ̂_∞) is the limit score. The code's C restricted to one penalty is C = Schur_Z(K) − g_Qg_Qᵀ/φ̂ (`ro/rail_face.rs:50-53`, with Schur_Z(S_R) = 0 for a disjoint rest).

*Proof.*

1. **Log-determinants.** |K + S_j/t| = |K_ZZ|·|Schur_Z(K) + Σ/t| = |K_ZZ|·t^{−r}·|Σ + t·Schur_Z(K)|. Also |S_λ|₊ contains the factor t^{−r}|Σ|. The factors t^{−r} cancel, leaving ½log|K_ZZ| + ½log|Σ + tSchur_Z(K)| − ½log|Σ|, which is analytic for |t| < 1/‖Σ^{-1/2}Schur_ZΣ^{-1/2}‖.
2. **Data term.** D_p(t) = yᵀy − bᵀ(K + S_j/t)⁻¹b is analytic in t by the same block inversion, and D_p(0) = D_face > 0.
3. **Derivative at 0.** Differentiating at t = 0 gives ½tr(Σ⁻¹Schur_Z(K)) − (ν/2)g_QᵀΣ⁻¹g_Q/D_face = c_j.
4. **Link to ρ.** Since ∂V/∂ρ_j = −t ∂_tṼ, we get −e^{ρ_j}∂V/∂ρ_j → c_j as ρ_j → ∞. ∎

**Numerical check (Check 4).** On the wiggliness face ρ₄ → ∞ of the toy:

- analytic c = 5687.1633;
- measured −e^{ρ}∂V/∂ρ = 5354.55 at ρ₄ = 10, 5687.317 at 20 and 5687.160 at 30.

On the joint face (ρ₃, ρ₄ → ∞): analytic c = (72.954, 44953.996), measured (72.955, 44953.997) at ρ = 30.

**Theorem 4 (face certificate) [proven; KKT/SOSC with strict complementarity, Nocedal–Wright 2006 Thm 12.6].** Let A be a face (a set of coordinates at t = 0) with block-disjoint penalty ranges, and F its free coordinates. Suppose:

- (i) Theorem 1 (or 2) applied to V_∞ on the free coordinates certifies a strict minimizer ρ\*_F within r₋ of ρ̃_F, in the H̃_FF-norm;
- (ii) for each j ∈ A, c_j(ρ̃_F) − δ_c − r₋·G_c > 0, where G_c ≥ sup over the ball of ‖∇_{ρ_F}c_j‖\*, and δ_c is the floating-point band of c_j.

Then (t = 0, ρ\*_F) is a strict local minimizer of Ṽ on the closed domain t ≥ 0, i.e. of V on the compactified space.

*Proof.* Ṽ is C² near (0, ρ\*_F) by Prop. 5. KKT holds with multipliers c_j(ρ\*_F) > 0, and ∇_FV_∞ = 0. Strict complementarity makes the critical cone {d : d_A = 0}, on which ∇²Ṽ = ∇²_FFV_∞ ≻ 0 by (C3). SOSC (Nocedal–Wright Thm 12.6) then applies. ∎

- **The tolerance δ_c** is the same backward-error band the code already derives for C (`ro/rail_face.rs:371-375`: q·ε·‖C‖·(1 + κ)), propagated through ½tr(Σ⁻¹·): δ_c = ½ q ε ‖C‖(1+κ) tr Σ⁻¹.
- **G_c** is an open analytic bound (§7). In practice r₋ ≲ 10⁻⁶ sd after polish, so condition (ii) is decided by c_j ≫ δ_c.

**Finite identification (no box needed) [proven; Burke–Moré 1988 Thm 3.x, Calamai–Moré 1987].** In t-coordinates the face is a nondegenerate KKT point (c_j > 0). A projected-gradient or projected-Newton iteration therefore reaches t_j = 0 **exactly, in finitely many steps**. In 1-D, when Ṽ ≈ ct + ½dt² with c > 0 and d > 0, the Newton step t − (c + dt)/d = −c/d is < 0 for every t, so the projection lands on 0 in one step.

Contrast this with ρ-coordinates. Newton there never arrives: λ² shrinks by e⁻¹ per step, and it would take ln(λ₀²/2band_f) ≈ 14 steps to reach ρ ≈ 26.7 in the multinomial example, past the 22.7 box edge.

**Remark (C ≻ 0 is sufficient, not necessary) [proven, checked].**

- *Disjoint ranges.* Leaving the face along coordinate j changes V by t_jc_j + O(t²). The exact first-order condition for a face with disjoint ranges is therefore **c_j > 0 for all j ∈ A**.
- *Overlapping ranges.* The directional derivative f(d) = ½tr(S(d)⁺C), with S(d) = Σ d_jS_j on the released space, is positively homogeneous of degree 1 but not linear (e.g. d₁d₂/(d₁+d₂) for identical blocks). The condition is **min_{d∈simplex} f(d) > 0**, which is a smooth convex-in-structure problem of dimension |A| − 1.
- *Why C ≻ 0 is too strong.* C ≻ 0 implies both conditions, but it is equivalent to them only when the released penalty can take *every* PD shape, which a fixed S_j cannot.
- *Toy counterexample.* A single-penalty face with eig C = (−1710, 35, 160, 166, 204, 212) has c = 5687 > 0. Its limit is the true optimum: the unconstrained minimizer sends ρ₄ → ∞. `certify_rail_face` (`ro/rail_face.rs:376-383`) refuses it with the false statement "releasing the face would not raise the criterion".

### 3.9 Degenerate minima

**Proposition 6 (structural Morse–Bott) [proven; checked].** Suppose the S_j are linearly dependent, e.g. S_a = S_b. Then V depends on ρ only through S_λ, so V = W∘φ with φ(ρ) = Σe^{ρ_j}S_j. When W has a nondegenerate minimum on the image cone, the critical set is a smooth submanifold of dimension q − rank φ', and the Hessian's kernel equals its tangent space. That is the Morse–Bott condition (Bott 1954).

Feehan (2020, Thm 1 and its converse, arXiv:1803.11319) shows that Morse–Bott holds iff the Łojasiewicz gradient inequality holds with the optimal exponent ½, i.e. V − V\* ≤ C‖∇V‖². Newton restricted to a slice then converges quadratically.

**Numerical check (Check 5).** With a duplicated cyclic-wiggliness penalty:

- eig H = (−4.6·10⁻¹⁰, 0.27, 0.30, 0.98, 1.75), with kernel v = (0, −0.707, 0, 0, 0.707);
- g·v = 1.6·10⁻²⁵ and T[v,v,v] = 7·10⁻¹⁶;
- (V − V\*)/(½λ_⊥²) = 0.987, 0.9987, 0.9999 at offsets 10⁻¹, 10⁻², 10⁻³.

**Certificate.** Quotient the degeneracy **before** optimizing:

1. Form G_jk = ⟨S_j, S_k⟩_F (with q×q entries).
2. Take rank(G) with a derived floating-point threshold: an eigenvalue band of q·ε·‖G‖.
3. Reparametrize onto a slice. For proportional or duplicate penalties, merge them. For general dependencies, use the image cone (§7).

Then apply Theorems 1 and 4 in the quotient. Near-degeneracy (μ small, s large) needs nothing special: the H-norm certificate measures everything in posterior sd, and the certified radius is correspondingly larger in ρ-units.

**Non-Morse–Bott degenerate critical points [conjectured].** Examples are V − V\* ∝ ρ⁴, or a cusp. Point data of order ≤ k cannot certify minimality: the (k+1)-jet can reverse it. Such points are also statistically meaningless, since the Laplace approximation of ρ fails there. The consistent verdict is "not certified; degenerate of order k", reported with the smallest H eigenvalue and |T̂[v,v,v]|.

### 3.10 Proposition 7: the regime classifier h₀

**Proposition 7 [proven for the model families; checked on REML].** Define h₀ := |D³V[p,p,p]|/λ², where p is the Newton step and λ² = pᵀHp. It is affine-invariant and equals L_H(0)·λ along p. For 1-D model functions under a pure Newton step:

| regime | h₀ | ΔV/λ² | λ₊²/λ² |
|---|---|---|---|
| Morse minimum (dist → 0) | → 0 (∝ λ) | → ½ | → 0 |
| exponential tail c e^{−kρ} (any k, c) | 1 | 1 − e⁻¹ = 0.632 | e⁻¹ = 0.368 |
| algebraic tail ρ^{−2} | 4/3 | 0.656 | 0.5625 |
| algebraic tail ρ^{−k} | (k+2)/(k+1) | → ln 2 as k → 0 | ((k+1)/(k+2))^k |
| logarithmic ramp −a log s | 2 | ln 2 = 0.693 | 1 |
| **multinomial failure, step 2** | ≳ 2 (inferred) | **0.650** | **1.027** |

*Proof.* These are direct computations. For the ramp: V' = −a/s, V'' = a/s², V''' = −2a/s³, p = s, λ² = a, and h₀ = 2as³/(s³a) = 2. The value drop per step is a ln 2. ∎

The table is reproduced exactly in `check_classifier.py` (Check 6a). On the toy REML tail, h₀ = 1.0009, 1.0003, 1.0001, 1.0000, … over 10 Newton iterates. Near the interior minimum, h₀/λ ≈ 0.02–0.29, i.e. h₀ → 0 like λ (Check 6b) **[checked]**.

Two uses:

- **Certification.** Since h ≥ h₀ ≥ 1 on every tail, Theorem 1 **never certifies an interior minimum on a tail**, whatever band_f is. This is exactly the false positive of §3.1 removed.
- **Diagnostics.** The step observables (ΔV/λ², λ₊²/λ²) are exact function and derivative evaluations, not FD. They tell a tail (→ go to a face in t) from a ramp or valley (→ keep descending) from the quadratic regime (→ certify).

### 3.11 The multinomial failure, analysed

The log is at `SP/q1561/all-tests.log`, test `families::quality_vs_statsmodels_ordinal_mnlogit::gam_multinomial_recovers_true_class_simplex` (`tests/quality/families/quality_vs_statsmodels_ordinal_mnlogit.rs:151`). The reported state is:

- ρ = (4.2535, 12.2882, −1.7712), V = 266.2455, |Pg| = 1.477·10⁻³;
- λ̂² = 1.452·10⁻⁴, then 9.175·10⁻⁵, then 9.426·10⁻⁵, with decreases ΔV of [3.914·10⁻⁵, 5.965·10⁻⁵];
- band_f = 3.95·10⁻¹¹, ARC with 84 iterations, and no rails;
- releasing coordinate 2 was tested and raised V by +0.135 or +40.1.

1. **Is band_f derived? Yes [proven by reading the code].** It is `growth·(|fixed_β| + |log|H|| + |log|S|| + |kkt|) + ½·logdet_forward_error + |inner residual energy|` (`ro/decrement_bands.rs:110-128`), with growth = γ_m for m = n + p² (Higham 2002, §3.1 and Thm 10.3 for the Cholesky logdet). It is the resolution of Ṽ, not a statistical tolerance.
   - The gate `ObjectiveNotResolvable` at `:131-137` compares it against `outer_rel_cost_floor` (`ro/run.rs:8628`), which contains the constant floor `COST_STALL_REL_TOL_FLOOR`. That gate is a magic-constant check, although it did not fire here.
2. **Is it noise-limited? No [proven from the logged numbers].** Each step decreased V by 4–6·10⁻⁵ ≈ 10⁶·band_f. The decrease is fully resolved.
3. **Is it degenerate or quadratic? Neither, and it is far from the quadratic region [proven from the numbers; interpretation conjectured].**
   - In the Morse region, Corollary 1 forces λ₊²/λ² ≤ 1/8 and ΔV/λ² → ½. The observed ratio is 1.027 with ΔV/λ² = 0.650. That sits at the logarithmic end of Prop. 7's table (h₀ ≈ 2, or larger since the ratio exceeds 1), so L_H ≳ 2/λ ≈ 200 in the H-norm.
   - The effective curvature along the step is |g|²/λ² ≈ (1.477·10⁻³)²/9.4·10⁻⁵ = 0.023, a posterior sd of about 6.6 in ρ: a weakly identified direction.
   - A pure *uncoupled* exponential tail would have λ² constant ratio e⁻¹. That is not observed either.
   - **Conjecture:** the direction is a flat valley or tail onset in which curvature decays faster than g². λ² ratio > 1 means the curvature is falling towards an inflection. It is heading either to a face (ρ₂ = 12.29 is the candidate) or through a nonconvex region, which is consistent with ARC being used.
4. **What went wrong in the code?** Two things, neither of them the certificate.
   - (a) The budget of 2 steps came from the invalid self-concordance formula (§3.6).
   - (b) The polish stopped because "λ² did not contract" (`ro/newton_polish.rs:106-127`). Contraction is guaranteed **only** inside the Kantorovich region (Corollary 1). Outside it, λ² can legitimately stay flat (ramp) or grow (approaching an inflection) while V decreases resolvably.
   - The correct action is to keep descending (a globalized Newton/ARC step) as long as the step's certified decrease exceeds band_f, and then certify by Theorem 1/2 (interior) or Theorem 4 (face in t).
   - Statistically the current point is already within λ̂ ≈ 0.0097 posterior sd of where Newton would go. That is irrelevant for inference, but the SPEC requires a certified optimum, and continuing is cheap and principled.

---

## 4. Numerical checks

All checks are in `SP/theory/certificate-theory/`, run with `SP/theory/venv/bin/python <script>`. The toy problem is Gaussian REML with n = 200 and 4 or 5 penalties:

- a ridge on x1;
- a cyclic Fourier wiggliness penalty on x2;
- a null-space penalty on s(x3);
- a cosine-basis wiggliness penalty on s(x3).

`check_derivs.py:7-30` builds it.

| # | Script | What is checked | Result |
|---|---|---|---|
| 1 | `check_derivs.py` | exact g, H, T (Lemma 1) against FD (test only); Prop. 3 bound | rel. err 1.1·10⁻⁹ / 4.5·10⁻¹⁰ / 6.1·10⁻¹⁰; max ratio 0.055 ≤ 1 |
| 2 | `check_certificates.py` (interior) | Theorem 1 with (★) at 1…10⁻⁶ sd | certified from 10⁻³ sd (h = 0.20, r₋ = 1.128·10⁻³ against a true 1.000·10⁻³); quadratic Newton λ: 2·10⁻² → 5·10⁻⁵ → 5·10⁻¹⁰ |
| 3 | `check_certificates.py` (tail, 'none' / 'linear') | Prop. 1; decrement-verdict false positive | ΔV/λ² = 0.632, ratio 0.368, t = 1 at every step; λ² = 6·10⁻¹² at ρ₄ = 34.5; h = ∞ (Theorem 1 refuses) |
| 4 | `check_certificates.py` (face) | Prop. 5, Theorem 4, the C ≻ 0 false negative | analytic c = 5687.163, measured 5687.160 at ρ = 30; face h = 4·10⁻¹⁶; eig C ∋ −1710 |
| 4b | same (joint face) | two-coordinate face | c = (72.954, 44953.996), measured (72.955, 44953.997); h = 1.2·10⁻¹⁶ |
| 5 | same (Morse–Bott) | Prop. 6 | zero eigenvalue with an exact flat direction; (V − V\*)/(½λ_⊥²) → 1 (0.987, 0.9987, 0.9999) |
| 6 | `check_classifier.py` | Prop. 7 | model table exact; REML tail h₀ = 1.000; interior h₀ ∝ λ |
| 7 | `check_smale.py` | Theorem 2 with analytic γ | γ = 2.5·10³; certifies at 10⁻⁶ sd (α = 2.5·10⁻³), not at 10⁻⁴ sd; Kantorovich certifies from 10⁻³ sd |

Limitations: every check uses Gaussian REML. The LAML and non-Gaussian cases were not checked numerically.

---

## 5. Literature (precise citations)

- **Kantorovich, L. V. (1948).** "Functional analysis and applied mathematics." *Uspekhi Mat. Nauk* 3(6), 89–185. This is the original Newton–Kantorovich theorem.
- **Ortega, J. M., & Rheinboldt, W. C. (1970).** *Iterative Solution of Nonlinear Equations in Several Variables.* Academic Press. **Thm 12.6.2** (Newton–Kantorovich, with the radii t\*, t\*\*) and §12.6 error estimates. These are used in Theorem 1.
- **Deuflhard, P. (2004).** *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms.* Springer SSCM 35. **Thm 2.1** (affine-covariant Newton–Mysovskikh/Kantorovich) and Ch. 3 (affine-conjugate form for minimization, in the energy norm ‖·‖_H). This is the justification for measuring everything in the H̃-norm.
- **Deuflhard, P., & Heindl, G. (1979).** "Affine invariant convergence theorems for Newton's method and extensions to related methods." *SIAM J. Numer. Anal.* 16(1), 1–10.
- **Smale, S. (1986).** "Newton's method estimates from data at one point." In *The Merging of Disciplines*, Springer, 185–196. Also **Blum, L., Cucker, F., Shub, M., & Smale, S. (1998).** *Complexity and Real Computation*, Springer, **Ch. 8** (α-theory, α₀ = (13 − 3√17)/4). And **Hauenstein, J. D., & Sottile, F. (2012).** "alphaCertified: certifying solutions to polynomial systems." *ACM TOMS* 38(4), Art. 28 (α-test with rigorous rounding).
- **Krawczyk, R. (1969).** "Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken." *Computing* 4, 187–201. **Moore, R. E. (1977).** "A test for existence of solutions to nonlinear systems." *SIAM J. Numer. Anal.* 14(4), 611–615. **Rump, S. M. (2010).** "Verification methods: Rigorous results using floating-point arithmetic." *Acta Numerica* 19, 287–449 (§§8, 13: Krawczyk operator and verification of nonlinear systems).
- **Nesterov, Y., & Nemirovski, A. (1994).** *Interior-Point Polynomial Algorithms in Convex Programming.* SIAM (Def. 2.1.1, self-concordance). **Nesterov, Y. (2018).** *Lectures on Convex Optimization*, 2nd ed., Springer, §5.1–5.2 (Thm 5.1.13 local Newton region; Thm 5.2.1). **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*, CUP, **§9.6.3** (λ ≤ ¼ ⇒ λ₊ ≤ 2λ², under M = 1; the code's citation at `newton_polish.rs:361`).
- **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization*, 2nd ed., Springer. **Thm 2.4** (SOSC, unconstrained) and **Thm 12.6** (SOSC with strict complementarity for constrained problems).
- **Burke, J. V., & Moré, J. J. (1988).** "On the identification of active constraints." *SIAM J. Numer. Anal.* 25(5), 1197–1211. **Calamai, P. H., & Moré, J. J. (1987).** "Projected gradient methods for linearly constrained problems." *Math. Programming* 39, 93–116. These give finite identification of a nondegenerate face.
- **Banach, S. (1938).** "Über homogene Polynome in (L²)." *Studia Math.* 7, 36–44 (for symmetric multilinear forms on Hilbert space, the norm equals the norm on the diagonal). This is Lemma 2.
- **Bott, R. (1954).** "Nondegenerate critical manifolds." *Ann. of Math.* 60, 248–261. **Łojasiewicz, S. (1963).** "Une propriété topologique des sous-ensembles analytiques réels." *Colloques internationaux du CNRS* 117, 87–89. **Kurdyka, K. (1998).** "On gradients of functions definable in o-minimal structures." *Ann. Inst. Fourier* 48(3), 769–783. **Feehan, P. M. N. (2020).** "On the Morse–Bott property of analytic functions on Banach spaces with Łojasiewicz exponent one half." *Calc. Var. PDE* 59, Art. 87 (arXiv:1803.11319): **Morse–Bott ⇔ Łojasiewicz exponent ½** for analytic functions. **Feehan, P. M. N., & Maridakis, M. (2020).** "Łojasiewicz–Simon gradient inequalities for analytic and Morse–Bott functions on Banach spaces." *J. reine angew. Math.* 765, 35–67.
- **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM: §3.1 (γ_m) and **Thm 10.3** (Cholesky backward error). These are the source of band_f.
- Context only (their constructions are not adopted): **Wood, S. N. (2011).** *JRSS-B* 73(1), 3–36 (Laplace approximate marginal likelihood for GAMs; the REML derivative structure of Lemma 1). **Zhang, H. (2004).** *JASA* 99, 250–261 (Matérn microergodicity: why ψ is weakly identified).

---

## 6. Consequences for gamfit

### 6.1 Diagnosis of every rung and band

| Item | Location | Theory | Verdict |
|---|---|---|---|
| NewtonDecrement verdict ½λ̂² + band_λ² ≤ band_f | `ro/decrement_bands.rs:166`, `opt` lib.rs ~2654 | rounding bands from γ_m, Higham Thm 10.3, Weyl | **principled bands, but not a certificate on its own** (§3.1 false positive on tails). Keep it as the *polish target*; add Theorem 1/2 as the *certificate* |
| band_f construction | `ro/decrement_bands.rs:110-128` | FP resolution of Ṽ | **principled** |
| ObjectiveNotResolvable gate | `ro/decrement_bands.rs:131-137` → `ro/run.rs:8628` | uses COST_STALL_REL_TOL_FLOOR | **magic**. Delete it; band_f stands on its own |
| SolverBand `max(tol, scale√ε)·(1+τ)` | `ro/run.rs:8774` (`outer_engine_gradient_band`), `:8702` | none (not affine-invariant, declared tolerance) | **heuristic**. Delete |
| CertificateScoreRelative τ(1+\|V\|) | `ro/run.rs` (StationarityBoundSource, 3204-3210) | none | **heuristic**. Delete |
| CurvatureResolvability √(2hτ) | `ro/run.rs:3211-3213`, `:8646` | decrement↔gradient conversion is right, but τ = rel_cost_floor | **half-derived**. Replace τ with band_f; it then equals the decrement rung |
| GradientReproducibility (2× spread) | `ro/run.rs:4678, 4695` | an empirical noise estimate with factor 2 | **heuristic**. Replace with the analytic δ_λ band |
| FixedPointResidual | `ro/run.rs:3217-3228` | config.tolerance | **heuristic** (no gradient on that route) |
| polish step budget (self-concordance, M = 1) | `ro/newton_polish.rs:357-380` | Boyd–Vandenberghe §9.6.3 with an assumed L = 2 | **invalid assumption** (§3.6). Replace with the Corollary 1 recursion |
| "decrement stopped contracting" stop | `ro/newton_polish.rs` (was :106-127) | holds only inside the Kantorovich region | **deleted**: the walk is bounded by each kept step's decrease `> band_f` and `V` bounded below; the λ₊ ≤ 2λ² test now only orders faces before Newton (§3.11) |
| `rail_face` C ≻ 0 test | `ro/rail_face.rs:369-383` (docs `:62-76`) | sufficient, not necessary | **wrong as a gate** (false negatives, §3.8). Replace with c_j > δ_c (disjoint) / simplex min (overlapping) |
| per-coordinate c_j | `ro/rail_face.rs:385-434` | Prop. 5, exact | **principled**. Promote it to *the* face test |
| LARGE_STEP_DELTA 1.0, PROBE_DELTA 1.0 | `ro/run.rs:4937-4995, 6833, 7114` | value probes | **heuristic**. Delete (Theorem 4 needs no probes) |
| asymptote window 12, MIN_TAIL_SAMPLES 3, EXP4_* | `ro/asymptote_certificate.rs:81, 86, 221, 226, 230` | curve fitting of tails | **heuristic**. Delete; replace with Prop. 5 + Theorem 4 |
| ASYMPTOTE_* (1e-4, 18, 0.5, 6), TAIL_SNAP_DRIFT_REL | `ro/run.rs:5744, 5757, 5769-5770, 5456, 5896, 6417` | same | **heuristic**. Delete |
| TAIL_SNAP_CURVATURE_BAND (0.25, 4.0) | `ro/run.rs:6356, 6492` | a guess at the exponential-tail curvature ratio. The exact ratio is Prop. 7 (h₀ = 1, contraction e⁻¹) | **heuristic**. Delete |
| FACE_LAW_ERROR_SLACK 4.0, ORDER_BAND 0.5, DOMAIN_MARGIN 1e-6 | `ro/run.rs:6037, 6050, 6154` | none | **magic**. Deleted with `falsify_face_law`; the analytic face proof spends no criterion evaluation |
| CERTIFY_RESUME_PROGRESS_REL | `ro/run.rs:7327` | none | **heuristic** |
| LOG_STRENGTH box as a proxy for ∞; RepresentabilityFace rung | `ro/run.rs:8495, 8507`; `ro/rail.rs:31-230` | violates the SPEC (hand box) | **delete**. Replace with t-coordinates (the domain t ≥ 0) |

### 6.2 The unified certificate (build this; it replaces the ladder)

All of this outer-optimizer work belongs in the `opt` crate (SPEC). gamfit supplies the oracle: Ṽ, g̃, H̃, the bands, T̂ or M, the face data C/c_j, and the Gram matrix of the S_j.

**Step 0: structural quotient** (before optimizing; Prop. 6).

1. Compute G_jk = ⟨S_j, S_k⟩_F.
2. If λ_min(G) ≤ q·ε·‖G‖, merge the dependent penalties (duplicates or proportional ones). The general dependent case is §7.

This removes exact Morse–Bott degeneracy.

**Step 1: coordinates.** Every smoothing coordinate lives on [−∞, +∞]. Use ρ_j in the interior. Switch coordinate j to t_j = e^{−ρ_j} with the domain t_j ≥ 0 as soon as the step observables show a tail (Prop. 7: h₀ ≈ 1, λ² ratio ≈ e⁻¹, ΔV/λ² ≈ 0.632 along a direction dominated by j). Handle the lower end with t'_j = e^{ρ_j} against the unpenalized limit model. There is no box.

**Step 2: descent while the decrease is resolvable.** Take ARC / trust-region / damped Newton steps, projected at t = 0. The stop rule is:

- continue while the accepted step's decrease exceeds band_f;
- stop *Unresolved* only when no step along the model direction decreases V by more than band_f while ½λ² > band_f. The existing "backtrack unresolved" logic at `ro/newton_polish.rs:116-120, 391-404` is the right form.

There is no contraction test and no a-priori budget outside the Kantorovich region. Termination is guaranteed because V is bounded below on the compactification and every step removes ≥ band_f (see the open problem in §7 for LAML).

**Step 3: interior certificate** (Theorem 1, or Theorem 2 where T̂ is unavailable).

1. At the current point, form λ = λ̂ + δ_λ, μ = 1 − η and R = 2λ/μ.
2. Compute s = max_j √((H̃⁻¹)_jj) and M from Prop. 4.
3. Form L = ‖T̂‖_F + R·24Ms⁴/(δ − sR)⁴ (★), with δ = ln(3/2).
4. **Certified** iff sR < δ and h = Lλ/μ² ≤ ½ (or α = γλ/μ ≤ α₀ with γ from §3.4).
5. Report r₋ = (μ − √(μ² − 2Lλ))/L in posterior sd.

**Step 4: face certificate** (Theorem 4). Faces enter in t-coordinates and are identified in finitely many projected steps.

1. Run Step 3 on V_∞ over the free coordinates.
2. For each face coordinate, check c_j > δ_c + r₋·G_c.
3. For overlapping penalties, check min over the simplex of ½tr(S(d)⁺C) > δ.
4. Delete the C ≻ 0 gate (`ro/rail_face.rs:376-383`).

**Step 5: polish to band_f** (the accuracy target, a floating-point quantity). Once Step 3 or 4 has certified, iterate Newton for the Corollary 1 budget:

- iterate θ_{k+1} = θ_k²/(2√(1 − θ_k)) from θ₀ = h;
- stop at the first k with ½(θ_k/L)² + band_λ² ≤ band_f;
- if the observed λ² fails λ₊² ≤ λ²h²/(4(1−h)) while h ≤ ½, refuse with the type **EvaluationInconsistent**. That is a proven bug signal: bands or derivatives are wrong.

**Step 6: report.** Report the verdict (Interior / Face(A) / Unresolved / Degenerate(k)), h or α, r₋ (sd), the face multipliers c_j, and the classifier h₀.

**Tolerances, and where each comes from.**

| Tolerance | Source |
|---|---|
| band_f, band_λ, η | FP error analysis, already derived in `decrement_bands.rs` |
| δ_c | the eigenvalue/trace backward error at `rail_face.rs:374` |
| rank threshold q·ε·‖G‖ | Weyl |
| L, M, s, δ = ln(3/2) | analytic, from Prop. 4, with the ½ in e^δ − 1 = ½ |
| ½ in h ≤ ½ and α₀ | the theorems' own constants |

There are no hand constants. δ = ln(3/2) is a free choice of analyticity radius inside (0, ln 2). Any value is valid, and one can optimize it per point in closed form by minimizing (★) over δ. That optimization is not a tuning knob, because every choice is rigorous.

### 6.3 Which failing clusters each recommendation addresses

| Cluster (from `SP/q1561/all-tests.log`, "did not certify") | Mechanism | Recommendation |
|---|---|---|
| Binomial logit BFGS StepSizeTooSmall with a coordinate at the box top (ρ₂ = 22.73; \|Pg\| = 2.28·10⁻⁵ vs 7.30·10⁻⁶) | an exponential tail driven into a hand box; the gradient-norm rung is not affine-invariant | Step 1 (t-coordinates, delete the box: `run.rs:8495, 8507`, `rail.rs`) + Step 4 face certificate + delete the SolverBand rung |
| Prostate binomial (\|Pg\| = 9.6·10⁻⁶ vs 7.3·10⁻⁶) | interior point judged by an ∞-norm gradient band | Step 3 (Kantorovich/Smale in the H-norm) + Step 5 polish; delete SolverBand/CertificateScoreRelative |
| Iso-kappa Matérn ("asymptote-rail declined", "tail-snap declined: curvature tie"; 0.331 vs 0.0181) | heuristic tail tests (TAIL_SNAP_CURVATURE_BAND, FACE_LAW_*) plus the C ≻ 0 false negative; weak ψ identification | Step 4 with c_j (fixes the false-negative class) + delete the asymptote/tail-snap constants; the ψ coordinate needs Step 3 with an analyticity radius in ψ (open, §7) |
| Multinomial ARC "decrement stopped contracting" | the SC budget (2 steps) + the contraction stop outside the quadratic region | delete `newton_polish.rs:106-127` and `:357-380`; Step 2 continuation; Step 3 or Step 4 (a candidate face at ρ₂ = 12.29) |
| Penguins (λ_min(H) = −23) | not a minimum | no certificate applies (correctly refused); the optimizer must take the negative-curvature step (outside this lane) |
| Weibull-AFT (decaying ĉ, not a tail) | interior, weakly identified | Step 3 |
| Timeouts (transformation-normal, Box-Cox, Yeo-Johnson, frailty, competing risks, gamlss-LS) | not a certificate failure per se | Step 5's derived budget bounds polish work; the classifier stops the futile tail-Newton loops (≈ 14 extra steps per tail coordinate in ρ) |

---

## 7. Open problems

1. **LAML bounds.** For non-Gaussian families H_β = XᵀW(β̂(ρ))X + S_λ, and β̂ moves with ρ.
   - T needs up to fifth likelihood derivatives in η (log|H| → W → W' … W'''). That is analytic but expensive.
   - A holomorphic-extension bound M (Prop. 4) needs a polydisc on which the inner mode β̂(z) extends. By the implicit function theorem it exists; a *quantitative* radius should follow from self-concordance-type relative bounds of the inner likelihood (|w'/w| ≤ 1 for logit and Poisson canonical links) **[conjectured]**.
   - With M, Theorem 2 certifies at the polish target without T.
2. **G_c**, a bound on ∇_{ρ_F}c_j over the ball (Theorem 4 (ii)). c_j is analytic in (t, ρ_F). A joint polydisc bound in the style of Prop. 4 would close it. Until then, c_j ≫ δ_c at r₋ ~ 10⁻⁶ sd decides every practical case.
3. **Overlapping faces.** The simplex condition min_{d∈Δ} ½tr(S(d)⁺C) > 0 is exact. It still needs (a) an algorithm with a certificate (the function is concave in d? to be checked) and (b) a proof that Ṽ is continuous on the compactification at corners where overlapping penalties meet (homogeneous functions such as t₁t₂/(t₁+t₂) are continuous but not differentiable).
4. **General dependent penalties** (Step 0 beyond duplicates). Quotienting onto the image cone of λ ↦ S_λ, when that cone is not generated by a subset of the S_j.
5. **Non-Morse–Bott degenerate critical points.** Characterize when a finite jet decides minimality for REML (analytic, so finitely determined generically?) **[open]**.
6. **Tighter remainders.** The Cauchy fourth-order term in (★) dominates for λ ≳ 10⁻². A sharper bound would widen the certified region by an order of magnitude. Two candidates:
   - an exact fourth tensor along p only, using the D⁴ log|H| formula of Lemma 1, which is cheap;
   - optimizing δ per point.
7. **Non-penalty hyperparameters (ψ, κ in Matérn; transformation parameters).** They need their own analyticity radius for Prop. 4 and their own compactification if they have limits. The Matérn identifiability lane's results should feed the δ_ψ used here.
8. **The multinomial point itself.** Whether the step-2 direction ends at a face (ρ₂ → ∞), in an interior weakly identified minimum, or crosses a nonconvex region cannot be settled without running the fit. The prediction: Step 2 descent followed by Step 3 or Step 4 certifies it in O(10) additional evaluations **[conjectured]**.
