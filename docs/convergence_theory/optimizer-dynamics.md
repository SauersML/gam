# Optimizer dynamics on the outer REML/LAML problem

Lane: `optimizer-dynamics` (convergence theory team). Status labels used throughout:
**[proven]** (a derivation is given here or a cited theorem applies with its hypotheses checked),
**[checked]** (verified numerically by a script listed in §4), **[conjectured]** (plausible,
not proven, and stated as an open problem in §7 where it matters).

Scripts and outputs: `SP/theory/optimizer-dynamics/` (SP = session scratchpad). Files:
`od_lib.py`, `demo_p1_p2.py` → `p1_p2.out`, `demo_p3_reml.py` → `p3_seed{1,2}.out`,
`demo_p4_inexact.py` → `p4.out`, `demo_p5_theta.py` → `p5_seed{1,2}.out`,
`demo_p6_hybrid.py` → `p6.out`.

---

## 1. Summary

1. **The outer objective has uniformly bounded derivatives in the ρ = log λ chart, but BFGS cannot use that.**
   - **[proven, Gaussian REML with φ fixed/absorbed]** In ρ, the criterion V has globally bounded gradient, Hessian *and third derivative* (T1): L_H ≤ ½(13‖z‖² + 6(r + r_S)). A first-order method sees a very benign problem.
   - Its difficulty is flat directions: V → V_∞ + c e^{−ρ_k} as ρ_k → ∞ (a null or oversmoothed term).
   - BFGS on such a direction drives κ(B⁻¹) ~ e^{ρ_k} and sᵀy → 0, and cos θ collapses **[checked, P1: cos 2.3e−3, cond 6.8e9 at ρ=22.5; 1.8e15 unboxed]**.
2. **StepSizeTooSmall and MaxAttempts are forced by floating point; neither is a tuning problem [proven, T2/T4].**
   - A cushioned Armijo test with value error δ and gradient error η accepts every α ≤ ᾱ = 2[(1−c1)|ĝᵀd| − η‖d‖]/(L‖d‖²).
   - Backtracking can therefore fail only in three ways:
     - (a) the value noise exceeds the cushion;
     - (b) the gradient is inconsistent with the value to a relative level of about 1;
     - (c) cos θ ≲ L·tol_x/(2(1−c1)ρ_bt‖g‖).
   - Separately, progress along −g is value-resolvable only while ‖g‖ > 2√(L ε_f).
   - For the binomial failure (V ≈ 306, ε_f = 6.8e−11) this floor is 1.65e−5·√L. That is *above* the certification bound 7.3e−6, so **no line-search method can certify that fit in double precision**.
   - "Bracket never closed" is the Armijo condition holding at every expansion. By NW Lemma 3.1 this means an unbounded or linear ray, or a flat exponential tail.
3. **Exact-Hessian adaptive cubic regularization (ARC, CGT 2011) resolves gradients far below the line-search floor.**
   - With an exact Hessian, rejections are explainable by value noise only when ‖g‖ ≤ 3 b^{2/3}(L_H/6)^{1/3}/(1−η1), where b = band_f **[proven, T4]**.
   - For the binomial case that is ≈ 5e−7, far below 7.3e−6. Exact-Hessian ARC can certify where BFGS provably cannot.
4. **Every "reject floor" with an analytic Hessian proves that the oracle is inconsistent [proven, T3].**
   - CGT Lemma 5.2 bounds σ_k for a consistent oracle.
   - The survival dim-6 case had 44 rejections down to radius 1e−12 with |g| = 0.27. At Δ = 1e−6 the predicted decrease is ≥ 1.36e−7, so a rejection needs value error ≥ 6e−8 or gradient error ≥ 45% of |g|.
   - This is a bug in the (V, g, H) oracle, not an optimizer problem.
5. **"Newton decrement stopped contracting" rests on a false premise [proven].**
   - The polish budget assumes self-concordance (Boyd–Vandenberghe §9.6.3, `newton_polish.rs:367-372`), but V is not self-concordant: the deviance atom is sigmoidal, with f″ = 0 while f‴ ≠ 0.
   - On exponential or algebraic tails the Newton decrement contracts only *linearly* **[proven, T5]**:
     - q = 1/e for an exponential tail;
     - q = ((p+1)/(p+2))^p for an algebraic tail;
     - q = 1 (no contraction at all) for a log-divergent, unbounded ray.
   - In the x1+cc(x2) failure the refused Newton steps each lowered V by 1.5·10⁶ × band_f.
6. **Łojasiewicz theory holds because V is real-analytic in ρ [proven sketch, §3.6].**
   - Monotone ARC iterates have finite length, hence converge to a *single* point or diverge to ∞ (to a face).
   - Rates depend on the Łojasiewicz exponent θ:
     - θ < 2/3: superlinear;
     - θ = 2/3: linear;
     - θ > 2/3: O(k^{−2/(3θ−2)}).
   - The degenerate valley has θ = 3/4 and is *observed* linear at rate 0.7218 (λ² ratio 0.2714) **[checked, P2]**.
7. **Faces (λ_k → ∞) must be certified at the face, not approached along the ρ tail [proven, T7; checked, P3/P6].**
   - In w = e^{−ρ_k}, V is analytic at w = 0 and ∂V/∂w(0) = f_w(0) = ½[tr(S₂⁻¹K) − cᵀS₂⁻¹c], the REML score-test statistic.
   - A positive f_w(0) together with a reduced interior certificate is a strict local minimum on the closed orthant. This replaces all "rail", "asymptote", "tail probe" and resolvability-box logic.
8. **The global θ = e^{−ρ/2} chart is a trap, even though the SPEC allows it [checked, P5].**
   - V is even in θ and θ = 0 is always critical, with H_θθ(0) = 2f_w(0).
   - But third derivatives grow like θ_s^{−3}. Exact ARC in θ did not converge in 40 iterations for either seed (σ up to ~10⁶, ρ₁ crawling from 0 to 0.50).
   - Use the ρ chart globally and the w = 0 face only as an exact candidate and certificate.
9. **Recommended replacement: RA-ARC (§3.9).**
   - One algorithm with one seed. It combines exact-Hessian ARC in ρ, exact face candidates with score-test multipliers, a rejection audit that turns oracle inconsistency into a typed error, and a Kantorovich terminal certificate.
   - Every tolerance comes from `decrement_bands.rs` FP bands.
   - Prototype (P6): face certificates in 10 evaluations (seeds 2 and 5) and an interior certificate in 29 evaluations (seed 1). Box-BFGS instead "converged" at the rail to a fake |g| = 3.6e−6.
10. **Most of the driver's heuristic layer can be deleted.** This covers the BFGS/EFS/hybrid arms, the seed cascade, parsimony seeds, stall windows, asymptote probes (which include SPEC-violating finite differences), rail margins, polish budgets and the first-order fallback (§6.2 inventory).
    - Two things must be built:
      - (i) exact Hessians where they are currently withheld;
      - (ii) the gradient-residual term in the bands.

---

## 2. Setup and notation

- ρ ∈ ℝ^K, λ_k = e^{ρ_k}. Penalized Hessian: H(ρ) = XᵀWX + Σ_k λ_k S_k, with rank r = p for the parametric block.
- Gaussian REML with φ absorbed (the base case for proofs):

  V(ρ) = ½[ D(ρ) + log|H| − log|S_λ|₊ ],  D = yᵀy − yᵀX H⁻¹ Xᵀy.

  General GLM LAML replaces D with the penalized deviance at β̂(ρ) and XᵀX with XᵀW(β̂)X.
- Whitened pieces:
  - A_k = H^{−1/2} λ_k S_k H^{−1/2}. Then 0 ≼ A_k and ΣA_k ≼ I.
  - ζ = H^{−1/2}Xᵀy, with ‖ζ‖² = ‖z‖² ≤ yᵀy.
  - A(v) = Σ v_k A_k. Then ‖A(v)‖ ≤ ‖v‖_∞ and tr|A(v)| ≤ ‖v‖_∞·r.
- Along a direction u: B1 = A(u), B2 = A(u∘u), B3 = A(u∘u∘u).
- Oracle errors: |V̂ − V| ≤ δ, ‖ĝ − ∇V‖ ≤ η, ‖Ĥ − ∇²V‖ ≤ ζ_H.
- FP bands, as in `gam/src/.../decrement_bands.rs`:
  - γ_m = mε/(1 − mε) with m = n + p²;
  - band_f = B_channels + B_factor + |E_r|, where E_r = ½rᵀH_β⁻¹r and r is the inner residual;
  - band_g_k = γ_m(|fixed_β_k| + |logdet_H_k| + |logdet_S_k|) + |kkt_k|;
  - band_H = γ_m‖H‖_F.
- Newton decrement: λ² = gᵀH⁻¹g. Tail telemetry:
  - r = ΔV/(½λ²) (actual over predicted decrease for a unit Newton step);
  - q = λ²₊/λ² (decrement contraction).
- Line-search constants: c1 (Armijo), c2 (curvature), ρ_bt (backtrack factor). ARC constants: η1 < η2 (ratio thresholds), γ1 < 1 < γ2 (σ factors).
- Compactified chart: w = e^{−ρ_k} ∈ [0, ∞). A face is a set F with w_F = 0, i.e. λ_F = ∞, so the term is removed into the null space of its penalty.

---

## 3. Results

### 3.1 T1 — uniform third-derivative bound in ρ [proven for Gaussian REML; conjectured for LAML/profiled φ]

Differentiating along u, with ∂_u H = H^{1/2}B1H^{1/2} and so on:

- d log|H| = tr B1
- d² log|H| = tr B2 − tr B1²
- d³ log|H| = tr B3 − 3 tr(B1B2) + 2 tr B1³

so |D³ log|H|| ≤ (1 + 3 + 2) r ‖u‖_∞³ = 6r‖u‖_∞³.

For the data term, D = yᵀy − ζᵀζ with ζ depending on ρ through H^{−1/2}:

- D′ = ζᵀB1ζ
- D″ = ζᵀ(B2 − 2B1²)ζ
- D‴ = ζᵀ(B3 − 3(B1B2 + B2B1) + 6B1³)ζ

so |D‴| ≤ (1 + 6 + 6)‖u‖_∞³‖z‖² = 13‖u‖_∞³‖z‖². The penalty log-determinant log|S_λ|₊ has the same structure on range(S), with rank r_S; for disjoint penalty blocks it is linear in ρ.

Summing gives the bounds below, in ℓ_∞ and hence in ℓ₂ since ‖u‖_∞ ≤ ‖u‖₂. A symmetric trilinear form bounded on the diagonal is bounded by the same constant (Banach 1938).

| Quantity | Bound |
|---|---|
| ‖∇V‖ | ≤ ½(‖z‖² + r + r_S) |
| ‖∇²V‖ | ≤ ½(3‖z‖² + 2r + 2r_S) |
| L_H = sup‖∇³V‖ | ≤ ½(13‖z‖² + 6(r + r_S)) |

**Consequences.**
- CGT's assumptions (AF.1 and AF.6, Lipschitz Hessian) hold *globally* in ρ, so the O(ε^{−3/2}) complexity theorem applies without a box.
- The bound is loose: on P3 the observed max|D³V| was 5.4 against a bound of about 4200 **[checked]**. It is used only in the *rejection audit* (§3.3), where a loose L_H makes the audit conservative, never wrong.
- In GLMs, each observation contributes softplus-like atoms σ(t) = log(1 + e^t), with |σ‴| ≤ 1/(6√3) and |σ‴| ≤ σ″. These are generalized self-concordant with ν = 2, M = 1 (Sun–Tran-Dinh 2019), which suggests a T1 analogue for LAML **[conjectured]**.
- V is nonconvex, and it is *not* self-concordant in the Nesterov–Nemirovski sense. At an inflection of the deviance atom f″ = 0 while f‴ ≠ 0, so |f‴| ≤ 2f″^{3/2} fails **[proven]**.

### 3.2 T2 — what line-search failures mean [proven]

**Cushioned Armijo.** Let φ(α) = V(x + αd). With the errors above and an L-Lipschitz ∇V:

V̂(x + αd) − V̂(x) ≤ αĝᵀd + αη‖d‖ + ½Lα²‖d‖² + 2δ.

The opt crate's cushion is ε_f = τ·ε·(1 + |V|) (`opt/src/lib.rs:1968-1971`). The test V̂(x + αd) ≤ V̂(x) + c1αĝᵀd + ε_f therefore holds for every

α ≤ ᾱ = 2[(1−c1)|ĝᵀd| − η‖d‖] / (L‖d‖²),  provided ε_f ≥ 2δ.

Backtracking from α = 1 by ρ_bt reaches ᾱ after ⌈log(1/ᾱ)/log(1/ρ_bt)⌉ trials. **StepSizeTooSmall** (α‖d‖ < tol_x) can therefore occur only if at least one of these holds:

- (a) 2δ > ε_f: value noise exceeds the cushion;
- (b) η‖d‖ ≥ (1−c1)|ĝᵀd|, i.e. η ≳ (1−c1)‖ĝ‖cos θ: the gradient is inconsistent with the values;
- (c) ᾱ‖d‖ < tol_x, i.e. cos θ < L·tol_x/(2(1−c1)ρ_bt‖g‖): the search direction is nearly orthogonal to −g.

For BFGS, cos θ ≥ 1/κ(B) (Zoutendijk/NW Thm 3.2). On a flat tail κ(B) ~ e^{ρ_k} (P1: cond(B⁻¹) = 6.75e9 at ρ₂ = 22.5, cos = 2.3e−3), so (c) is the generic mechanism.

**"After 50 attempts"** is only the label of `BACKTRACKING_MAX_ATTEMPTS = 50` (`lib.rs:3355`); `WOLFE_MAX_ATTEMPTS = 20` (`lib.rs:3354`). The count carries no information. Case (a), (b) or (c) is what exhausts it.

**"MaxAttempts: the bracket never closed."** In a Moré–Thuente/NW Alg. 3.5 search the bracket closes when Armijo fails or φ′ ≥ 0.
- If every expansion α₁ < … < α_N satisfies Armijo, then V(x) − V(x + α_N d) ≥ c1 α_N|φ′(0)|.
- NW Lemma 3.1 guarantees a Wolfe interval only when φ is bounded below along the ray. So either the ray is unbounded below or asymptotically linear, or |φ′(0)| ≤ (V − V_inf)/(c1α_N).
- The second case is the exponential tail: V − V_inf ≈ c e^{−ρ}, so a flat ray is Armijo-feasible for all α.
- Where it exists, the Wolfe interval has width ≥ (1−c2)|φ′(0)|/(L‖d‖²), which shrinks to zero with |φ′(0)| on a tail.

A second source is the non-finite guard: the searches return MaxAttempts after three non-finite evaluations (`func_evals >= 3`, at `lib.rs:12212-12216` in Wolfe and `lib.rs:12500-12504` in backtracking). That is an oracle-domain failure mislabeled as a line-search failure.

**The fallback cascade** at `lib.rs:8975-9070` makes a failed search "succeed" by weakening its own guarantee:
- it relaxes c2 to 0.5 and then 0.1;
- it lowers c1 to 1e−3;
- it switches to nonmonotone GLL with cap 10.

None of these changes addresses (a), (b) or (c).

**Non-convex and unbounded failure modes.**
- Dai (2002) and Mascarenhas (2004) give examples where BFGS with exact or Wolfe line searches fails to converge on nonconvex smooth functions. Powell's 1976 and BNY's 1987 global convergence theorems require convexity (BNY: bounded level set and a convex f, or at least a uniformly positive-definite Hessian on the level set).
- V violates both. It is nonconvex (it has saddles between face and interior basins, P6 seed 1), and level sets are unbounded along flat tails. So **no convergence theorem covers BFGS on this problem [proven, by the failure of hypotheses]**.
- Dennis–Moré superlinear convergence needs a nonsingular limit Hessian. At a face limit the ρ-chart Hessian → 0, so it fails too.

### 3.3 T3 — ARC rejection audit [proven]

Let m(s) = V + ĝᵀs + ½sᵀĤs + (σ/3)‖s‖³ and pred = V − m(s). The actual decrease is

V̂(x) − V̂(x + s) = pred + (ĝ − g)ᵀs + ½sᵀ(Ĥ − H)s + (σ/3)‖s‖³ − R₃ + e₁,

where |R₃| ≤ L_H‖s‖³/6 and |e₁| ≤ 2δ. Hence

ρ̂ − 1 ≥ −[η‖s‖ + ½ζ_H‖s‖² + L_H‖s‖³/6 + 2δ] / pred.

With an exact Hessian (ζ_H = 0), CGT Lemma 5.2, eq. (5.6), gives σ_k ≤ max(σ₀, (3/2)γ₂(C + L_H)) for a consistent oracle. Hence:

**Audit.** Define the slack C(s) = (1−η1)·pred − [2 band_f + band_g‖s‖ + ½band_H‖s‖² + L_H‖s‖³/6]. A rejection with C(s) > 0 is impossible for a consistent oracle, so it certifies an *oracle inconsistency*, which is a typed error. It is never a reason to shrink further.

**Survival dim-6 checkpoint** (`all-tests.log`, reject floor at radius 1e−12).
- Inputs: |g| = 0.2718, λ_min = −6.39e−4, Hessian = Analytic, 44 consecutive rejections, ρ = [2.013, 1.925, 13.563, 5.517, 20.241, −3.693].
- At Δ = 1e−6 the Cauchy decrease alone gives pred ≥ |g|Δ/2 = 1.359e−7.
- A rejection with η1 = 0.1 therefore needs δ ≥ 6.1e−8 (about 10⁴ × band_f) or η ≥ 0.122, which is 45% of |g|.
- **The oracle is inconsistent. This is proven, not conjectured.**
- The existing `RejectFloor` path (the radius clamp at `lib.rs:~903-960`) converts this into "no progress" instead of reporting it.

### 3.4 T4 — resolution floors [proven]

**Line search.** Take the best exact step along −g, which achieves a decrease of ‖g‖²/(2L). The decrease can be adjudicated against value noise only if ‖g‖²/(2L) > 2ε_f, i.e. ‖g‖ > 2√(Lε_f). For BFGS, replace ‖g‖ with cos θ·‖g‖.

**Exact-Hessian ARC.** The model error is at most L_H‖s‖³/6. A rejection can be blamed on value noise b only if pred ≤ (2b + L_H‖s‖³/6)/(1−η1) for the steps tried. Minimizing 2b/s + L_Hs²/6 over s, the rejections are explainable by noise for all s only if

‖g‖ ≤ 3 b^{2/3} (L_H/6)^{1/3} / (1−η1).

| Case | ε_f or b | Line-search floor | ARC floor | Certify bound |
|---|---|---|---|---|
| binomial (V≈306) | 6.8e−11 | 1.65e−5·√L | 5.2e−7·(L_H/5)^{1/3} | 7.3e−6 |
| P4 (V≈190) | 4.25e−11 | 2.09e−5 (L=2.58) | ~4e−7 | — |

The observed binomial stall at |g| = 2.28e−5 matches the line-search floor with L ≈ 1.9 **[consistent; L not measured in the Rust run]**.

### 3.5 T5 — tail telemetry for Newton-type methods [proven; checked]

Here r = ΔV/(½λ²) and q = λ²₊/λ² for a pure Newton step in the flat coordinate.

| Tail model | Newton step | q | r | gap/λ² |
|---|---|---|---|---|
| V_∞ + c e^{−mρ} | 1/m (constant) | 1/e = 0.3679 | 2(1 − 1/e) = 1.2642 | 1 |
| V_∞ + cρ^{−p} | ρ/(p+1) | ((p+1)/(p+2))^p | 2(p+1)/p·(1 − q) | (p+1)/p |
| V₀ − c log ρ (unbounded) | ρ (doubling) | 1 | 2 ln 2 = 1.386 | — |
| c x⁴ (degenerate) | x/3 | (2/3)⁴ = 0.1975 | — | 3/4 |

**Degenerate curved valley** V = (x₁ − x₂²)² + x₂⁴/4.
- The Newton map on the valley parameter a obeys a′ = (5−4a)(7−4a)/(6−4a)².
- The fixed point b = 4a solves b³ − 16b² + 84b − 140 = 0, so b ≈ 3.406.
- This gives contraction c = (6−b)/(7−b) = 0.72178, q = c⁴ = 0.2714 and gap/λ² → 0.844.
- **[checked P2: 0.7218, 0.2714, 0.8440, r = 1.2298]**.

**σ-halving ARC in the ρ chart** (steady state, derived and checked).
- On an exponential tail g, H and the optimal σ all scale like e^{−ρ}. The accepted step size is therefore stationary at s = ln(1/γ_dec) = ln 2, and q = ½.
- **[checked P1: 0.693 per step, q → 0.5]**. ARC marches toward a face at constant speed and never arrives. That is why faces need their own certificate (§3.7).

**x1+cc(x2) failure** (`all-tests.log:1009`).
- Final state: ARC after 84 iterations, V = 266.2455, |Pg| = 1.477e−3 against a bound of 1.352e−6, ρ = [4.2535, 12.2882, −1.7712].
- Polish steps: λ̂² went 1.452e−4 → 9.175e−5 → 9.426e−5; ΔV = 3.914e−5, then 5.965e−5.
- So r = 0.539 then 1.30, and q = 0.632 then 1.027. This matches no exponential tail (q = 0.37). It is consistent with a log-like or pre-asymptotic tail (b ln 2 = 6.4e−5) **[conjectured]**.
- Either way, each refused step lowered V by 1.5·10⁶ × band_f (band_f ≈ 3.95e−11). "Stopped contracting" stopped a productive iteration **[proven from logged numbers]**.

### 3.6 Łojasiewicz/KL analysis [proven sketch; rates derived]

V is real-analytic on ℝ^K for Gaussian REML: it is a composition of exp, the inverse of an SPD matrix, and log det. For LAML it is analytic wherever β̂(ρ) is a nondegenerate inner optimum, by the analytic implicit function theorem. So the Łojasiewicz gradient inequality |V(x) − V(x*)|^θ ≤ C‖∇V(x)‖ holds near every critical point, with θ ∈ [½, 1) (Łojasiewicz 1963/65; Kurdyka 1998 for the o-minimal form).

**Single limit for monotone ARC.**
- Accepted steps satisfy V_k − V_{k+1} ≥ (η1σ_min/6)‖s_k‖³ =: a‖s_k‖³ (CGT Lemma 2.1 with the cubic term).
- They also satisfy ‖g_{k+1}‖ ≤ (L_H/2 + σ_max)‖s_k‖² =: b‖s_k‖² (CGT Lemma 5.2/eq. 5.6 with the first-order subproblem condition).
- Let φ(t) = t^{1−θ}/(1−θ) (the KL desingularizer) and Δφ_k = φ(V_k − V*) − φ(V_{k+1} − V*). Concavity gives Δφ_k ≥ φ′(V_k − V*)(V_k − V_{k+1}) ≥ a‖s_k‖³ / (C b ‖s_{k−1}‖²).
- AM–GM, ‖s_k‖ ≤ (‖s_k‖³/‖s_{k−1}‖²)^{1/3}‖s_{k−1}‖^{2/3}, then gives ‖s_k‖ ≤ (1/3)(Cb/a)Δφ_k + (2/3)‖s_{k−1}‖. Summing yields Σ‖s_k‖ < ∞.
- Hence the iterates either converge to a single critical point or leave every compact set. The same dichotomy appears in Absil–Mahony–Andrews 2005, Thm 3.2, and in the Attouch–Bolte 2009 framework.
- In the ρ chart the second alternative is exactly a face limit ‖ρ‖ → ∞, which is why §3.7 is required.

**Why Absil Thm 4.4 does not apply directly.** Condition (B), m(0) − m(p) ≥ c‖g‖‖p‖, holds for the exact regularized Newton step with c = ½/κ(H + μI). That c → 0 at a singular limit, which is exactly the degenerate case. The KL argument above avoids it.

**Rates.** Write e_k = V_k − V*. Combining the two step inequalities with Łojasiewicz gives e_k − e_{k+1} ≥ c·e_{k+1}^{3θ/2} (consistent with Zhou–Wang–Liang, NeurIPS 2018).
- θ < 2/3: superlinear (θ = ½, a nondegenerate minimum: quadratic).
- θ = 2/3: linear.
- θ > 2/3: e_k = O(k^{−2/(3θ−2)}). For the valley, θ = 3/4 gives worst case O(k^{−8}).
- The ARC *trajectory* on the valley is linear at 0.2714 **[checked P2]**, because along the curve the function behaves like θ = ½ in the trajectory's own parametrization.

**Certification at a singular Hessian** is open (§7). The Kantorovich test (§3.8) needs ‖H⁻¹‖ < ∞. With λ_min(H) ≤ band_H, the only honest outcomes are:
- a typed *degenerate stationary point* result carrying λ² and λ_min;
- or a face test, if the null direction is a penalty direction.

### 3.7 T7 — the compactified face chart [proven for Gaussian; checked]

Partition the coefficients into term a (kept) and term b (the penalty S₂ on block b is full rank on its block; its null part is absorbed in a). With w = e^{−ρ₂}:

V(ρ₁, w) = V_red(ρ₁) + ½[ log|S₂ + wK| − w·cᵀ(S₂ + wK)⁻¹c ] + const,

where

- K = X_bᵀ(I − X_aH_aa⁻¹X_aᵀ)X_b,
- c = X_bᵀ(y − X_aβ̂_a).

This form is analytic at w = 0 (**[checked to 3e−14]**, P3/P6 identity check). The face derivative is

f_w(0) = ½[ tr(S₂⁻¹K) − cᵀS₂⁻¹c ],

the REML score statistic for the variance component λ₂⁻¹ = 0 (Verbyla 1990; Lin 1997).

Chart transforms for w > 0: f_w = −g₂/w, f_ww = (H₂₂ + g₂)/w², f_1w = −H₁₂/w **[checked]**.

**Face certificate [proven].** Let the reduced problem at w_F = 0 have gradient g_r, Hessian H_r and multipliers μ_j = f_{w_j}(0). If
- the reduced decrement satisfies λ_r² ≤ band_λ²,
- H_r ≻ band_H,
- and μ_j > band_μ,

then the point is a strict local minimizer of V on the closed orthant w ≥ 0. Here band_μ = γ_m(|tr(S₂⁻¹K)| + |cᵀS₂⁻¹c|). The argument is second-order sufficiency with strict complementarity: the reduced Hessian covers the free directions and μ > 0 covers the active ones.

V is nonconvex in w, so the face certificate is local, like the interior one.

**θ chart [proven by chain rule; checked P5].** With θ = e^{−ρ/2} (w = θ²), V is even in θ and θ = 0 is always critical. The chart derivatives are:
- g_θ = −2g₂/θ,
- H_θθ = (4H₂₂ + 2g₂)/θ²,
- H_1θ = −2H₁₂/θ,
- H_θθ(0) = 2f_w(0), verified to five digits.

This chart looks attractive because the face becomes an ordinary interior critical point. But the *global* θ chart is badly scaled. The transition scale θ_s ~ 4e−3 (the ρ₂ where the tail begins) makes third derivatives O(θ_s^{−3}). Exact ARC in θ therefore escalates σ to ~2.6e5–2.1e6, oscillates θ in sign, and does not converge in 40 iterations for either seed. ρ₁ ends at 0.50 (seed 2) and 0.80 (seed 1), against optima 3.035 and 3.214.

**Conclusion [checked]:** use the ρ chart globally (uniform T1 constants) and the w = 0 face only as an exact candidate and certificate.

### 3.8 T6 — interior certificates and "stopped contracting" [proven]

**Kantorovich** (Ortega–Rheinboldt 12.6.2). Let β = ‖H(x)⁻¹‖, η = ‖H⁻¹g‖ and h = βL_Hη. If h ≤ ½, a unique zero of ∇V lies within r* = (1 − √(1−2h))/(βL_H) ≤ 2η. Newton from x converges to it quadratically, and the terminal phase needs no values. For L_H use the T1 bound, or better a local bound on the ball (§7).

**Strong-convexity ball (simpler).** If ‖g‖ ≤ λ_min(H)²/(4L_H), the minimizer is within 2‖g‖/λ_min and the gap is ≤ ‖g‖²/λ_min.

**"Newton decrement stopped contracting"** (`newton_polish.rs:118-126`).
- *Inside* the Kantorovich region, a non-contracting decrement contradicts the theorem, so it certifies an oracle inconsistency (typed error).
- *Outside* the region there is no contraction requirement. The step budget at `newton_polish.rs:367-372` uses the self-concordant bound (λ ≤ 0.25 ⇒ quadratic phase), whose premise is false for V (§3.1).
- The test must go. It should be replaced by the Kantorovich test (success) plus the audit (failure).

**ArcUnprogressingStallCheckpoint** (`run.rs:1683`) has the same status. With an exact Hessian and a consistent oracle, CGT Thm 5.3/5.4 rules out an unprogressing ARC (σ is bounded, and each successful step lowers V by ≥ κ‖g‖^{3/2}). The checkpoint therefore detects either the §3.3 inconsistency or the tail marching of §3.5. The audit handles the first; the face candidate handles the second.

### 3.9 RA-ARC: the principled replacement [algorithm; theorem proven by citation]

**Inputs:** an exact oracle (V, g, H) with bands (band_f, band_g, band_H) from `decrement_bands.rs`, the T1 constant L_H, and one seed ρ₀ (the existing deterministic initializer; no cascade). Algorithm parameters, which are not tolerances: η1 = 0.1, η2 = 0.9, γ1 = ½, γ2 = 2, σ₀ = 1. Their values affect only constants, never correctness.

1. **Evaluate** V, g, H at x_k. Record λ_min(H) and λ² = gᵀH⁻¹g (if H ≻ 0).
2. **Interior certificate.** If λ_min(H) > band_H and the Kantorovich condition h ≤ ½ holds, run the terminal Newton phase (step 8) and return CERTIFIED_INTERIOR. Also stop if λ² ≤ band_λ² = 2 band_f, since beyond that the gap is not value-resolvable.
3. **Face candidate.**
   - Let F = { j : g_j < −band_g_j }, the coordinates whose gradient points toward λ_j → ∞.
   - For F ≠ ∅, evaluate V at w_F = 0 exactly, via the reduced model plus ½log|S_F|.
   - If V_face < V_k − band_f, solve the reduced problem **by RA-ARC itself**. P6 seed 3 shows that a pure reduced Newton overflows. Then compute μ_j = f_{w_j}(0):
     - all μ_j > band_μ: return CERTIFIED_FACE;
     - some μ_j < −band_μ: release j, i.e. restart ARC from the face point with w_j > 0 chosen by one cubic step in the w chart (f_w < 0 ⇒ descent);
     - |μ_j| ≤ band_μ: return typed DEGENERATE_FACE with (λ_r², μ).
   - Otherwise continue with the interior step.
4. **Cubic step.** Solve min_s m(s) exactly: eigendecomposition of H, secular equation in μ = σ‖s‖ (`opt/src/lib.rs:7158`), hard case (`lib.rs:7026`), iterated to the FP fixed point (no iteration cap).
5. **Relaxed ratio** (Sun–Nocedal 2023, eq. (7)–(8)): ρ̂ = (V_k − V⁺ + r·band_f)/(pred + r·band_f), with r = 2/(1−η2).
6. **Update.**
   - ρ̂ ≥ η1: accept.
   - ρ̂ ≥ η2: σ ← max(σ_min, γ1σ).
   - ρ̂ < η1: reject and set σ ← γ2σ.
   - σ_min = 0 is allowed (pure Newton) and σ_max is unnecessary: σ is bounded by CGT Lemma 5.2 whenever the oracle is consistent.
7. **Rejection audit.** On every rejection compute C(s) (§3.3). If C(s) > 0, return typed ORACLE_INCONSISTENT carrying (δ̂, η̂) = the measured discrepancies.
8. **Terminal phase.** Take Newton steps from a Kantorovich point until ‖H⁻¹g‖ stops decreasing within band_g. Any non-contraction inside the region is ORACLE_INCONSISTENT.

**Theorem [proven by citation, given T1 and a consistent oracle].**
- (i) σ_k stays bounded.
- (ii) Every accepted step decreases V monotonically. Face candidates are accepted only if they lower V by more than band_f, which preserves this.
- (iii) Within O(ε^{−3/2}) iterations (CGT 2011 Part II, Thm 5.4 with exact H), either ‖g‖ ≤ ε for any ε above the ARC floor of §3.4, or a face candidate is taken.
- (iv) By §3.6, iterates converge to a single critical point or to ∞ in some set F. In the second case g_F < 0 eventually, so step 3 fires and the face test decides.
- (v) Sun–Nocedal Thm 6/7: with bounded noise the iterates reach and stay in the critical region {‖g‖ ≲ (band_f)^{2/3}}.

**Outcomes:** CERTIFIED_INTERIOR, CERTIFIED_FACE, DEGENERATE_STATIONARY, DEGENERATE_FACE, ORACLE_INCONSISTENT. Only the first two are fits. The last three are typed errors carrying the evidence.

**Prototype P6** [checked]:

| Seed | Result | Evaluations | Notes |
|---|---|---|---|
| 2 | FACE at k=1 | 10 | 5 reduced Newton steps; ρ₁=3.0354425913, λ_r²=5.6e−15, μ=16786.2 > 1.3e−8 |
| 1 | INTERIOR at k=13 | 29 | face rejected at every iteration (μ from −1605 to −99737) |
| 3 | FACE | 27 | the pure reduced Newton at k=0 overflowed: the face solve must be globalized |
| 5 | FACE at k=1 | 10 | μ = 25527.7 |

On the same seed-2 problem, box-BFGS "converged" at the rail ρ₂ = 22.254 with |g| = 3.64e−6 < 7.3e−6. That is a *fake certificate*: the point is not stationary in any chart, and the true face optimum has multiplier 16786. ρ-chart ARC needed 36 iterations to certify only because band_f let it walk until e^{−ρ₂} was invisible (ρ₂ = 33.5).

---

## 4. Numerical checks

All scripts use numpy (and mpmath for P5) in `SP/theory/venv`.

| Demo | What | Key numbers |
|---|---|---|
| P1 `demo_p1_p2.py` | boundary-flat V = F₀ + ½(x₁−1)² + e^{−x₂}(1+0.3(x₁−1)²), F₀ = 306 | Box BFGS "converged" at rail x₂ = 22.7 in 36 iterations; cond(B⁻¹) 6.75e9, cos 2.3e−3, sᵀy 4.6e−12. Cushion 6.82e−11; floor 1.65e−5; e^{−x₂} = ε_f at x₂ = 23.41. Unboxed: x₂ = 35.7, cond 1.8e15, cos 7e−5. ARC: step 0.693, q → 0.5, r → 1, gap/λ² → 1. w chart: one-step KKT certificate with multiplier 1.0. |
| P2 `demo_p1_p2.py` | degenerate valley | ARC: contraction 0.7218, q = 0.2714, gap/λ² = 0.8440, r = 1.2298 (theory 0.72178, 0.2714, 0.844). BFGS: 400 iterations to f ~ 1e−79 with cond = ∞; value floor x₂ ≈ 9.7e−4. |
| P3 `demo_p3_reml.py` seed 2 | Gaussian REML, signal + null term (n=300, p=17) | FD checks: gradient 1.4e−9, Hessian 5.5e−11. T1 on [−25,35]²: max\|H\| = 22.0, max\|D³V\| = 5.4 (bound ~4200). Box BFGS fake convergence at ρ₂ = 22.254, \|g\| = 3.64e−6. ARC certified in 36 iterations at ρ₂ = 33.5, band_f = 2.4e−11. Pure Newton tail: step 1.0000, q = 0.3679, r = 1.2642. Face Newton: λ² = 1.5e−22, μ = 16786.21. V(face) = 182.47307781957 vs ARC 182.47307781961. |
| P3 seed 1 | interior optimum | ρ* = [3.21423, 13.36352]; BFGS 13 iterations, ARC 14; face μ = −2885.5; V_int = 190.33967 < V_face = 190.34181. |
| P4 `demo_p4_inexact.py` | CG inner solve, warm start | See the error-order table below. |
| P5 `demo_p5_theta.py` | θ-chart exact ARC (60-digit) | Both seeds hit maxit = 40; σ up to 2.1e6 early, then 6.5e4 → 2e3 while θ → 1e−23 and ρ₁ crawls (0.50 vs 3.035 optimum for seed 2). Final H_θθ = 40780.08 = 2f_w(0) (seed 2); 3395.19 (seed 1, at a *wrong* ρ₁, where the face multiplier is still positive). |
| P6 `demo_p6_hybrid.py` | RA-ARC prototype | Table in §3.9. |

P4 value and gradient errors by inner tolerance:

| tol | \|V err\| | \|g err\| |
|---|---|---|
| 1e−3 | 1.25e−4 | 3.6e−3 |
| 1e−4 | 1.7e−7 | 7.3e−5 |
| 1e−5 | 7.8e−10 | 5.1e−6 |
| 1e−6 | 1.6e−11 | 1.5e−6 |

The rest of P4:
- The ratio \|V err\|/\|g err\|² stays O(10): value error is second order, gradient error first order.
- L = 2.58, floor 2.09e−5.
- Warm-start BFGS reports false convergence with true \|g\| = 3.5e−4, 6.0e−6 and 4.9e−8 at tol 1e−4, 1e−5 and 1e−6. The warm-started residual makes g̃ ≈ 5e−11 at a non-stationary point.
- Cold start at tol 1e−4: StepSizeTooSmall with δ = 1.11e−8 ≫ ε_f = 4.25e−11 and η = 1.33e−4 ≫ (1−c1)\|g\|cos θ = 3.7e−6. These are T2 causes (a) and (b), identified.
- ARC with the inexact oracle: 15 and 19 consecutive rejections, σ up to 4.1e3. With the consistent oracle: 0 rejections, 14 iterations.

**Recommended regression test** (FD is allowed in tests): at the survival dim-6 checkpoint ρ above, compare a central FD of V against g, and a central FD of g against H. T3 predicts a discrepancy ≥ 0.12 in some gradient coordinate or ≥ 6e−8 in V noise.

---

## 5. Literature (precise citations)

- **Cartis, Gould, Toint (2011a)**, "Adaptive cubic regularisation methods for unconstrained optimization. Part I", *Math. Program.* 127(2):245–295. Cor. 2.6 (Cauchy decrease), Lemma 2.1, Thm 4.5 (quadratic local rate), Lemma 5.2 eq. (5.6) (σ bound), Thm 5.3/5.4, §6 (exact subproblem).
- **Cartis, Gould, Toint (2011b)**, Part II, *Math. Program.* 130:295–319: O(ε^{−3/2}) worst-case complexity.
- **Nesterov, Polyak (2006)**, "Cubic regularization of Newton method and its global performance", *Math. Program.* 108:177–205. §3 (global rates), §4.2 (gradient-dominated functions of degree 1–2, which is the Łojasiewicz case).
- **Sun, Nocedal (2023)**, "A trust region method for noisy unconstrained optimization", *Math. Program.* 202:445–472. Relaxed ratio (7), r = 2/(1−c₂) in (8), Algorithm 1, Thm 6 (critical region (37)), Thm 7.
- **Absil, Mahony, Andrews (2005)**, "Convergence of the iterates of descent methods for analytic cost functions", *SIAM J. Optim.* 16(2):531–547. Lemma 2.1, Def. 3.1 (strong descent conditions (10)–(11)), Thm 3.2 (single limit or ‖x‖ → ∞), Thm 4.1, Alg. 4.2, Lemma 4.3, Thm 4.4 conditions (B)–(D).
- **Attouch, Bolte (2009)**, *Math. Program.* 116:5–16. **Attouch, Bolte, Svaiter (2013)**, *Math. Program.* 137:91–129 (abstract descent framework, conditions H1–H3).
- **Łojasiewicz** (1963, 1965); **Kurdyka** (1998), *Ann. Inst. Fourier* 48:769–783.
- **Zhou, Wang, Liang (2018)**, "Convergence of cubic regularization for nonconvex optimization under KL property", NeurIPS.
- **Powell (1976)**, SIAM-AMS Proc. 9:53–72 (BFGS global convergence, convex). **Byrd, Nocedal, Yuan (1987)**, *SIAM J. Numer. Anal.* 24(5):1171–1190. **Dennis, Moré (1974)**, *Math. Comp.* 28:549–560; (1977) *SIAM Rev.* 19:46–89. **Dai (2002)**, *SIAM J. Optim.* 13(3):693–701. **Mascarenhas (2004)**, *Math. Program.* 99:49–61. **Moré, Thuente (1994)**, *ACM TOMS* 20:286–307.
- **Nocedal, Wright (2006)**, *Numerical Optimization*, 2nd ed.: Lemma 3.1, Thm 3.2 (Zoutendijk), Lemma 4.3, Thm 4.8, Thm 6.5/6.6.
- **Boyd, Vandenberghe (2004)**, §9.6.3 (self-concordant Newton analysis, whose premise fails here).
- **Nesterov (2004)**, *Introductory Lectures*, Thm 4.1.13/4.1.14. **Sun, Tran-Dinh (2019)**, "Generalized self-concordant functions", *Math. Program.* 178:145–213.
- **Shi, Xie, Byrd, Nocedal (2022)**, "A noise-tolerant quasi-Newton algorithm", *SIAM J. Optim.* 32(1):29–55.
- **Carter (1991)**, *SIAM J. Numer. Anal.* 28(1):251–265 (inexact gradients in trust regions). **Conn, Gould, Toint (2000)**, *Trust-Region Methods*, §8.4 and §10.6.
- **Ortega, Rheinboldt (1970)**, 12.6.2 (Kantorovich).
- **Bertsekas (1982)**, *SIAM J. Control Optim.* 20:221–246 (projected Newton). **Coleman, Li (1996)**, *SIAM J. Optim.* 6:418–445.
- **Verbyla (1990)**, *JRSS B* 52:493–508; **Lin (1997)**, *Biometrika* 84:309–326 (variance-component score tests; f_w(0) is this statistic).

---

## 6. Consequences for gamfit

### 6.1 Cluster → cause → fix

| Failing cluster | Evidence (`q1561/all-tests.log`) | Cause (label) | Fix |
|---|---|---|---|
| Binomial StepSizeTooSmall, railed | lines 324/871/886: \|g\| = 2.28e−5 vs 7.30e−6, ρ at 22.73 in box [−17.36, 22.73] | line-search floor above the certify bound (T4, proven) plus a flat tail (T5) | RA-ARC with exact H; face candidate at w = 0; delete the box |
| Prostate BFGS | lines 786/822: \|g\| = 9.6e−6 | same floor (T4, conjectured L) | same |
| x1+cc(x2) "stopped contracting" | line 1009 | false self-concordance premise (T6, proven); refused productive steps | delete the polish budget; Kantorovich certificate plus face candidate |
| Multinomial "stopped contracting" | — | same (conjectured) | same |
| Iso-kappa Matérn MaxAttempts | line 15518: \|g\| 3.7e−2 vs 2.5e−2 (and \|Pg\| = 0.331 after 129 iterations) | gradient-only plan (`drivers/spatial_optimization.rs:5468`) plus FD tail probes (`run.rs:7114`) | supply the exact Hessian; delete the probes |
| Weibull-AFT / statsmodels MaxAttempts | line 33697: 6.99e−2 vs 1.86e−3 | Hessian withheld (`survival/construction.rs:991, 1036`); cos-collapse or non-finite guard (T2) | exact H; the non-finite guard becomes a typed domain error |
| Survival reject floor | line 33712; dim-6 checkpoint | oracle inconsistent (T3, proven) | FD consistency test at the checkpoint; fix the oracle; audit becomes a typed error |
| Fake rail convergence | P3/P6 | box plus gradient test at the rail | face certificate |
| Timeouts | — | tail marching × seed cascade × arm retries (conjectured) | one seed, one algorithm, face candidates |

### 6.2 Heuristic inventory (SPEC class → action)

SPEC classes: **C** cap, **F** fallback/retry, **M** magic constant, **B** box, **D** derivative-free/FD, **W** wall-clock.

| Location | Item | Class | Action |
|---|---|---|---|
| `opt/src/lib.rs:3354-3355` | WOLFE_MAX_ATTEMPTS=20, BACKTRACKING_MAX_ATTEMPTS=50 | C | delete with line searches from the outer path |
| `opt/src/lib.rs:8975-9070` | c1/c2 relaxation, GLL cap 10 | F, M | delete |
| `opt/src/lib.rs:12212-12216`, `12500-12504` | `func_evals >= 3` non-finite | C | typed OracleDomain error |
| `opt/src/lib.rs:12565-12670`, `12672` | jiggle, expansion, `probe_alphas` grid | D, F | delete |
| `opt/src/lib.rs:3376-3378` | ARC_NUMERICAL_CONV_FACTOR=16 | M | replace with band_λ² = 2band_f |
| `opt/src/lib.rs:6780-6800` | ARC tol 1e−5, max_iter 100, σ_min 1e−10, σ_max 1e12, subproblem_max_iterations 80, AutoBfgs, history_cap 12 | M, C | tolerances from bands; no caps; exact-Hessian mode mandatory for the outer path |
| `opt/src/lib.rs:~903-960` | within_noise_floor → ρ = 1; RejectFloor radius clamp | M, F | Sun–Nocedal relaxed ratio plus audit |
| `opt/src/lib.rs:7500-7560` | σ_max saturation | C | delete (CGT Lemma 5.2 bound) |
| `newton_polish.rs:118-126`, `367-372` | "stopped contracting", SC budget | M (false premise) | "stopped contracting" **deleted**; the λ₊ ≤ 2λ² face-ordering test remains |
| `run.rs:1683` | ArcUnprogressingStallCheckpoint | F | delete; audit plus face candidate |
| `run.rs:14` | OPERATOR_TRUST_RESTART_RADIUS_FLOOR=1e−6 | M, F | delete |
| `run.rs:3080` | MAX_EXPANSIONS=64 | C | delete |
| `run.rs:4678` | GRADIENT_REPRODUCIBILITY_WIDENING=2 | M | band_g |
| `run.rs:4937` | LARGE_STEP_DELTA=1 | M | delete |
| `run.rs:5744`, `5757`, `5769` | ASYMPTOTE_* (rel tol 1e−4, 18 probes, δ 0.5) | M, D | delete; face certificate |
| `run.rs:6037`, `6050`, `6154` | FACE_LAW_* (slack 4, order band 0.5, margin 1e−6) | M | **deleted**: the analytic face proof mints rails with no value probe |
| `run.rs:6417` | TAIL_SNAP_DRIFT_REL=1e−2 | M | delete |
| `run.rs:6833`, `7114` | PROBE_DELTA=1.0 (FD tail probe) | D (SPEC violation) | delete |
| `run.rs:6969` | PROBE_DOMAIN_MARGIN=1e−6 | M | delete |
| `run.rs:7327` | CERTIFY_RESUME_PROGRESS_REL=32ε | M | delete |
| `run_plan.rs:4`, `621`, `683` | should_start_next_seed, outer_seed_cascade | F | single seed |
| `run_plan.rs:2152`, `2899`, `2955`, `2941-3000` | Bfgs, Efs, HybridEfs, FirstOrderFallbackRequested arms | F | delete; keep only the RA-ARC arm (`1443` path) |
| `run_plan.rs:250`, `873`, `885` | PARSIMONY_COMPARISON_SEED_COUNT=2, STRUCTURAL_EARLY_EXIT_MIN_COUNT=2, GENERIC_STRUCTURAL_BAIL_MIN_RUN=3 | M, F | delete |
| `seed_screening.rs:612`, `687`, `705` | OVERSMOOTH_BOUNDARY_MARGIN=0.5, PARSIMONY_* bands | M, B | delete; face multipliers decide parsimony |
| `asymptote_certificate.rs:81`, `86`, `221-230` | window 12, MIN_TAIL_SAMPLES 3, EXP4_* | M | delete |
| `bridges.rs:152`, `154`, `170`, `182` | probe cache 256, reject cost 1e11, refusal thresholds 150/25 | C, M | delete |
| `bridges.rs:300-302` | COST_STALL_WINDOW 6/3, REL_TOL_FLOOR 1e−7 | M | delete; band_f |
| `bridges.rs:2311`, `2315`, `2320` | inner-cap override 0.01, floor 3, ceiling 64 | C, M | adaptive inner tolerance (§6.4) |
| `bridges.rs:3525` | ACCEPTED_STEP_COST_MATCH_ULPS=8 | M | band_f |
| `bridges.rs:3787-3857` | box relaxed inward / coordinate_rail_margin | B | delete |
| `bridges.rs:4360-4396` | EFS backtrack 8, threshold 0.5, descent tol 1e−12, ψ-stagnation 1 | C, M, F | delete with the EFS arm |
| model_types (via `rho_optimizer.rs:73`) | CERTIFICATE_RAIL_MARGIN=0.5 | B, M | delete |
| `estimate/rho_domain.rs:144`, `190` | resolvability_interval, coordinate_domain boxes | B | delete; the w ≥ 0 domain constraint is the only constraint |
| `capability.rs:233`, `458-464`, `612-624`, `668-680` | prefer_gradient_only, plan → Bfgs/BfgsApprox, #2898 exact-curvature retry, saddle latch | F | plan() always returns exact-Hessian RA-ARC |
| `decrement_bands.rs` | band_f refuses when > rel_cost_floor(1+\|V\|) (1e−7) | M | keep the bands; delete the 1e−7 refusal; add the gradient-residual term |

### 6.3 Build list

1. **Exact Hessians everywhere they are withheld:**
   - `survival/construction.rs:991` (`.with_hessian(Unavailable)`), `1036` (`prefer_gradient_only`), `1167`;
   - `gamlss/builders.rs:4825-4829`;
   - `fit_orchestration/fit.rs:2144`;
   - `drivers/spatial_optimization.rs:1852`, `5468`;
   - `drivers/constant_curvature_profile.rs:873`;
   - `gam-sae manifold/support_outer.rs:1240-1241`.

   RA-ARC's floor advantage (§3.4) requires ζ_H = 0 or ζ_H ≤ band_H.
2. **In the opt crate:** an `ExactArc` driver with the Sun–Nocedal relaxed ratio, the rejection audit and the Kantorovich terminal phase, with typed outcomes (§3.9). It reuses the subproblem at `lib.rs:7158`/`7026`, without the iteration cap. General optimizer work belongs here, as the SPEC requires.
3. **In gamfit:** a face oracle per penalty. It returns V_face (the reduced fit plus ½log|S_F|₊) and the multipliers μ_j = ½[tr(S_j⁻¹K_j) − c_jᵀS_j⁻¹c_j] in the stable resolvent form (`demo_p3_reml.py:fw_w`). For LAML the analogue uses the working weights at the reduced optimum **[conjectured form; see glm-laml-landscape / compactification lanes]**.
4. **In the bands:** the gradient residual term |δg_k| ≤ ‖H^{−1/2}∂_βg_k‖·√(2E_r), or the first-order correction g_k ← g_k − (∂_βg_k)ᵀH⁻¹r.

### 6.4 Derived tolerances (no magic constants)

| Tolerance | Formula | Source |
|---|---|---|
| value band | band_f = B_channels + B_factor + \|E_r\| | FP error analysis (`decrement_bands.rs`) |
| interior stop | λ² ≤ 2 band_f | gap ≈ λ²/2 cannot be resolved below 2band_f (T4) |
| interior certificate | Kantorovich h = ‖H⁻¹‖L_H‖H⁻¹g‖ ≤ ½, and λ_min(H) > band_H | O–R 12.6.2 |
| face trigger | g_j < −band_g_j | sign resolvable |
| face acceptance | V_face < V − band_f | preserves monotone ARC |
| multiplier | μ_j > band_μ = γ_m(\|tr S⁻¹K\| + \|cᵀS⁻¹c\|) | FP of f_w(0) |
| ratio relaxation | r·band_f, r = 2/(1−η2) | Sun–Nocedal (8) |
| audit | C(s) > 0 ⇒ inconsistent | T3 |
| inner tolerance | ‖H^{−1/2}r‖ ≤ (1−η1)‖g‖/(4‖H^{−1/2}∂_βg‖) | makes η ≤ (1−η1)‖g‖/4 (Carter 1991; CGT 2000 §8.4); see the inexact-oracle lane |

The value error from this inner tolerance is ½‖H^{−1/2}r‖², second order and hence automatically within band once the gradient condition holds (P4 ratio O(10)).

---

## 7. Open problems

1. **Degenerate certificate.** At λ_min(H) ≤ band_H in the interior, with a non-penalty null direction, no certificate is available. A candidate is a higher-order test: a third/fourth-derivative sign on the null direction, as in the P2 quartic. This needs a band for D³/D⁴, which T1 supplies only loosely **[open]**.
2. **Sharp local L_H.** The T1 bound is ~800× loose on P3. Kantorovich with a local bound on the ball B(x, 2η) (e.g. from ‖A_k‖ at x) would certify earlier **[open]**.
3. **T1 for LAML and profiled φ.** This needs bounds on dβ̂/dρ through the inner Hessian, and on the weight derivatives W′ and W″ **[conjectured via generalized self-concordance]**.
4. **The λ → 0 face** (ρ_k → −∞, unpenalized limit). The analogous chart is v = e^{ρ_k}, which is analytic when XᵀWX is nonsingular on the penalty's range. The multiplier and its band are not derived here **[open]**.
5. **Face-set combinatorics.** With many terms, F from the sign rule might cycle (release/re-fix). Monotone acceptance prevents infinite cycling (finitely many faces, V strictly decreasing by > band_f), but no polynomial bound is known **[conjectured]**.
6. **Global optimality.** All certificates are local. P6 seed 1 shows that face and interior basins coexist (V_int < V_face by 2.1e−3). A global statement needs landscape results (glm-laml-landscape lane).
7. **x1+cc(x2) tail law.** The observed q ≈ 1 matches a log-divergent or pre-asymptotic ray. Whether the cyclic-cubic null space produces an unbounded-below REML ray (which would be a model-specification error, not an optimizer error) is not determined **[open]**.
