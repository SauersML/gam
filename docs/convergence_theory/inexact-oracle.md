# Inexact-oracle theory for the REML/LAML outer optimization

Slug: `inexact-oracle`. This report covers the errors in the outer criterion V(ρ), its gradient and its Hessian when they are computed from an inexact P-IRLS mode. It then covers when a line search on such an oracle must fail, what inner accuracy the outer search needs, a noise-aware line search that can be implemented in `opt`, and an audit of gamfit's tolerances.

Scripts and outputs are in `SP/theory/inexact-oracle/`, where `SP = /tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad`. Line numbers for gamfit refer to `/home/user/gam` at `486fd7441a`. Line numbers for `opt` refer to `/root/.cargo/git/checkouts/opt-4a38fa79856f3ac9/53ce029/opt/src/lib.rs`.

Each result is labelled with how well it is established:

- **[P]** proven here, with the proof given.
- **[P-lead]** proven to leading order in the inner decrement δ; a rigorous remainder needs segment constants (open problem O3).
- **[N]** checked numerically.
- **[C]** conjecture or hypothesis.

---

## 1. Summary

- **Value error.** Let δ = ‖r‖_{H⁻¹} be the inner Newton decrement.
  - The uncorrected LAML value has a *first-order* error, bounded by ½ν·edf·δ.
  - The corrected value that gamfit forms (objective.rs:488–548) is V − ½rᵀH⁻¹r − ∇_βGᵀH⁻¹r. Its error is *second order*: |V_c − V*| ≤ ¼(ν₂+2ν²)·edf·δ². [P-lead, N]
  - gamfit's band instead charges |cost_correction|, which is first order. In the check this charge is 10⁴–10⁶× the true error. [N]
- **Gradient error.** The IFT-corrected ρ-gradient (kkt.rs:78) removes only the ½βᵀA_kβ part of the first-order error. What remains is exactly linear, c_kᵀe, with a closed-form c_k.
  - Sharp computable bound: |g̃_k − g_k| ≤ ‖c_k‖_{H⁻¹}·δ. This is attained; the ratio in the adversarial direction is 1.000. [P-lead, N]
  - Cheap rigorous bound from quantities gamfit already has: B_k·δ, with B_k = ½ν(τ_k+edf) + (ν²+½ν₂)·edf·s_k. [P-lead, N]
  - gamfit's gradient band term |part.total − envelope| (decrement_bands.rs:84–86) is *not* a bound on this error. It is the size of the correction, not the size of what the correction leaves behind. [P, N]
- **Why value-tested line searches stop.** Take an Armijo test with value noise ε_f, cushion c and oracle slope error e₀. It can fail for every step (Theorem 4) iff
  (1−c₁)|φ̃'(0)| < e₀ + √(2L_d(2ε_f − c)).
  So there are two floors:
  - a noise floor g_min^val ≈ 2√(L·ε_f), which is 1e-5 to 6e-5 for V ≈ 306 in the prostate case and brackets the observed |g| = 9.6e-6 / 2.28e-5;
  - a gradient-error floor g_min^grad = e₀/(1−c₁).

  The certificate's bound 7.302e-6 = 490·√ε (run.rs:8913) lies below the first floor, so BFGS is asked for a gradient that a value-tested search cannot be guaranteed to deliver. The first floor is exactly the observed prostate/binomial-logit `StepSizeTooSmall` pattern. [P for the theorem, N for the gradient floor (κ = 1e-4 case failed at 9.38e-5 against a predicted 9.92e-5), C that this is the mechanism in the failing tests]
- **Gradient-certified line search.** Accept a step if the value test with a *negative* cushion −(ε_f(0)+ε_f(α)) passes, **or** if the trapezoid test
  ½α(φ̃'(0)+φ̃'(α)) + ½α(e₀+e_α) + R(α) ≤ c₁αφ̃'(0)
  passes. This search provably succeeds whenever (1−c₁)|φ̃'(0)| > 2ē (Theorem 6). The √(Lε_f) floor disappears.
  - `StepSizeTooSmall` then becomes a *certificate event*: descent is unresolvable, |g̃ᵀd| ≤ 2ē/(1−c₁).
  - The step floor 1e-12(1+‖x‖) (lib.rs:8018), the attempt caps 50/20 (lib.rs:3354–3355) and the c₁/c₂ adaptation become unnecessary.
  - [P given a remainder bound R; N with R omitted]
- **Inner accuracy.** Newton on the self-concordant inner problem converges quadratically, so solving to the *arithmetic floor* δ_floor costs about one extra Newton step per outer evaluation (measured: 36 against 21 inner steps for 14 against 15 evaluations).
  - At the floor every inner-residual band is at rounding level and the oracle is consistent.
  - gamfit's adaptive rule clamp(0.1‖g‖, reml_tol/100, 1e-6) on ‖r‖ ≤ tol·√(np) produced a *false stationary point*: oracle |g̃| = 7e-16 while the true |g| = 9e-8 (a warm start that stopped after 0 iterations). [N]
  - The derived rule has no constants. The Carter-type alternative, for inner solvers that are not quadratically convergent, is δ·Σ_k|d_k|‖c_k‖ ≤ θ(1−c₁)|g̃ᵀd| with θ < ½. [P]
- **Certificate.**
  - `newton_decrement_verdict` (lib.rs:2522) omits the δc²/μ term. At a coordinate with c = 0 it certifies regardless of how large that coordinate's band δc is. Fix: add Σ δc_j²/μ_j. [P]
  - The value band should be 2ε_f, the resolution of a *difference* of two evaluations. This closes a √2-wide dead zone between "the search cannot descend" and "the certificate cannot certify". [P]
- **Magic constants.** 23 constants in the inner/outer tolerance chain are removed or derived (§6.4).

---

## 2. Setup and notation

Fix ρ ∈ ℝ^K, λ_k = e^{ρ_k}, A_k = λ_kS_k ⪰ 0 and S_λ = Σ_kA_k.

- Inner objective: L(β) = −ℓ(β) + ½βᵀS_λβ, with mode β* = β*(ρ).
- Curvature: H(β) = XᵀW(β)X + S_λ, where W = diag(w_i) and w_i = w(η_i).
- Laplace term: G(β) = ½log|H(β)|.
- LAML: V(ρ) = L(β*) + G(β*) − ½log|S_λ|₊ (+ constants).

(The REML/Gaussian case is the special case w' ≡ 0, where every "moving-Hessian" term below vanishes.)

**Inexact mode.**
- β̃ is the returned iterate; e = β̃ − β*.
- r = ∇L(β̃) is the inner residual; q = H⁻¹r, with H evaluated at β̃.
- δ = ‖r‖_{H⁻¹} = √(rᵀq) is the inner Newton decrement.

**Per-row weight derivatives.** w_i' = dw/dη and w_i'' = d²w/dη².
- Logistic: w = μ(1−μ), w' = w(1−2μ), w'' = w(1−6w).
- κ_i = |w_i'/w_i| (logistic: |1−2μ_i| ≤ 1).
- κ₂ᵢ = |w_i''/w_i| (logistic: |1−6w_i| ≤ 1).

**Leverages and constants.**
- h_i = x_iᵀH⁻¹x_i; edf = Σ_i w_ih_i = tr(H⁻¹XᵀWX).
- ν = max_i κ_i√h_i (self-concordance constant of L in the H-metric).
- ν₂ = max_i κ₂ᵢh_i.
- τ_k = tr(H⁻¹A_k).
- v_k = H⁻¹A_kβ, so −v_k = ∂β/∂ρ_k.
- s_k = ‖A_kβ‖_{H⁻¹} = √((A_kβ)ᵀv_k). Note ½s_k² ≤ ½βᵀA_kβ = the `fixed_beta` channel.
- ∇G_j = ½tr(H⁻¹∂_jH), so ∇Gᵀu = ½tr(H⁻¹D_βH[u]) and ∇G = ½Xᵀ(w'⊙h).
- a = H⁻¹∇G.

**Directional derivatives.**
- D_βH[u] = Xᵀdiag(w'⊙Xu)X.
- D²_βH[u,u] = Xᵀdiag(w''⊙(Xu)²)X.
- T[u,v] = D_βH[u]v, the third-derivative tensor of L.

**The oracle.** gamfit returns Ṽ(ρ), g̃(ρ) (and H̃_V(ρ)) computed at β̃ with corrections. ε_f, ε_g,k and ε_H denote certified bounds on their errors.

**Outer line search.** Along d from ρ, φ(α) = V(ρ+αd) and φ̃(α) = Ṽ(ρ+αd).
- Slope bands: e_α = Σ_k|d_k|ε_g,k(ρ+αd); ē = max(e₀, e_α).
- L_d = sup φ'' on the segment; M₃ = sup|φ'''| on the segment.
- c₁ ∈ (0, ½) is the Armijo parameter. It is a free algorithmic parameter, not a tolerance: every threshold below depends on it only through 1/(1−c₁) ≈ 1.

**Arithmetic.** u = ε/2 and γ_m = mu/(1−mu) (Higham 2002, Lemma 3.1).

---

## 3. Results

### 3.1 Local geometry [P]

**Lemma 1.** For all u, v:
(i) |vᵀD_βH[u]v| ≤ ν‖u‖_H‖v‖_H², i.e. −ν‖u‖_H H ⪯ D_βH[u] ⪯ ν‖u‖_H H.
(ii) ±D_βH[u] ⪯ ν‖u‖_H·XᵀWX.
(iii) |∇Gᵀu| ≤ ½ν·edf·‖u‖_H, i.e. ‖∇G‖_{H⁻¹} ≤ ½ν·edf.
(iv) ‖H^{-1/2}D_βH[u]H^{-1/2}‖_F² ≤ ν²·edf·‖u‖_H².
(v) |tr(H⁻¹D²_βH[u,v])| ≤ ν₂·edf·‖u‖_H‖v‖_H.

*Proof.* By Cauchy–Schwarz in the H-metric, |x_iᵀu| ≤ √h_i‖u‖_H.
- Hence |w_i'x_iᵀu| ≤ κ_i√h_i·w_i‖u‖_H ≤ ν w_i‖u‖_H. Summing w_i'(x_iᵀu)(x_iᵀv)² over i gives (ii), and (i) follows because XᵀWX ⪯ H.
- (iii): ∇Gᵀu = ½tr(H⁻¹D_βH[u]) ≤ ½ν‖u‖_H·tr(H⁻¹XᵀWX).
- (iv): the operator norm of M = H^{-1/2}D_βH[u]H^{-1/2} is at most ν‖u‖_H by (i). Its trace norm is at most ν‖u‖_H·edf by (ii). Then ‖M‖_F² ≤ ‖M‖₂‖M‖_*.
- (v): |w_i''(x_iᵀu)(x_iᵀv)| ≤ κ₂ᵢh_i w_i‖u‖_H‖v‖_H ≤ ν₂w_i‖u‖_H‖v‖_H, and then take the trace against H⁻¹. ∎

For the canonical logistic, Poisson-log and binomial links, κ_i ≤ 1, so ν ≤ max_i√h_i ≤ 1 (Bach 2010 calls this generalized self-concordance). By Nesterov (2004, Thm 4.1.13), a standard self-concordant function with decrement δ < 1/ν has ‖e‖_H ≤ δ/(1−νδ). Hence ‖e‖_H = δ(1+O(νδ)), which is what "[P-lead]" rests on below.

### 3.2 Value error

**Theorem 1.**
(a) Uncorrected: V(β̃) − V* = ∇Gᵀe + ½rᵀH⁻¹r + O(δ²). Therefore |V(β̃) − V*| ≤ ½ν·edf·δ + O(δ²). [P-lead]
(b) Corrected, V_c = V(β̃) − ½rᵀq − ∇Gᵀq (this is objective.rs:488–548, where ∇Gᵀq = ½tr(H⁻¹D_βH[q]) is the gam#1395 moving-Hessian term). Then
  V_c − V* = −½eᵀ∇²_βG e + ½aᵀT[e,e] + O(δ³)   (1)
  |V_c − V*| ≤ ¼(ν₂ + 2ν²)·edf·δ² + O(δ³).   (2) [P-lead]

*Proof.* Expand about β̃, with every derivative at β̃.
- The stationarity condition 0 = ∇L(β*) gives r = He − ½T[e,e] + O(e³). Hence e = q + ½H⁻¹T[e,e] + O(e³).
- L* = L(β̃) − rᵀe + ½eᵀHe − ⅙T[e,e,e] + … = L(β̃) − ½rᵀq + O(νδ³).
- G* = G(β̃) − ∇Gᵀe + ½eᵀ∇²Ge + O(e³), and ∇Gᵀe = ∇Gᵀq + ½aᵀT[e,e] + O(e³).
- Adding the pieces gives (a) and (1).

For (2), write ∇²G[u,u] = ½tr(H⁻¹D²H[u,u]) − ½tr((H⁻¹DH[u])²).
- By Lemma 1(v) and (iv), |∇²G[e,e]| ≤ ½(ν₂+ν²)·edf·δ².
- By Lemma 1(i) applied as a trilinear bound, |aᵀT[e,e]| ≤ ν‖a‖_H δ² ≤ ½ν²·edf·δ².
- Summing gives ½·½(ν₂+ν²)edf δ² + ½·½ν²edf δ² = ¼(ν₂+2ν²)edf δ². ∎

**Consequence.** The corrected value is second-order accurate, but gamfit charges the *first-order* quantity |cost_correction| = |−½rᵀq − ∇Gᵀq| ≈ |∇Gᵀq| to `InnerResidualCharge` (objective.rs:539–549, then decrement_bands.rs:113–119). That charge is a bound on the error of the *uncorrected* value, not of the value gamfit actually returns.

### 3.3 Gradient error

The exact gradient at the mode is
  g_k = ½β*ᵀA_kβ* + ½tr(H⁻¹A_k) − ½tr(H⁻¹D_βH[v_k]) − ½rank_k,
with the third term being ½tr(H⁻¹D_βH[∂β/∂ρ_k]). Let g̃_k be the same formula at β̃, and let C_k = −(A_kβ̃)ᵀq + ½qᵀA_kq be the IFT correction (kkt.rs:78).

**Theorem 2.** Let E_k = g̃_k + C_k − g_k. Then E_k = c_kᵀe + O(δ²), where, as vectors in β-space,

  c_k = c₁ + c₂ + c₃ + c₄,
  c₁ = −½Xᵀ(w' ⊙ diag(X H⁻¹A_kH⁻¹ Xᵀ)),
  c₂ = −½Xᵀ(w'' ⊙ (Xv_k) ⊙ h − w' ⊙ diag(X H⁻¹D_βH[v_k]H⁻¹ Xᵀ)),
  c₃ = Xᵀ(w' ⊙ (Xa) ⊙ (Xv_k)),
  c₄ = −A_ka.   (3)

Consequently:
- the sharp bound is |E_k| ≤ ‖c_k‖_{H⁻¹}·δ(1+O(νδ)), and it is attained for e ∝ H⁻¹c_k;
- the cheap bound is |E_k| ≤ B_kδ + O(δ²), with B_k = ½ν(τ_k + edf) + (ν² + ½ν₂)·edf·s_k;
- without the IFT correction the bound gains the term s_kδ: B_k^unc = B_k + s_k.

[P-lead for all three; N]

*Proof.* Differentiate g_k(β) in the direction −e.
- The ½βᵀA_kβ part contributes −(A_kβ)ᵀe, and C_k cancels it up to (A_kβ)ᵀ(e−q) + O(δ²) = O(δ²).
- The ½tr(H⁻¹A_k) part contributes −½tr(H⁻¹DH[e]H⁻¹A_k), which is c₁ᵀe.
- The moving-Hessian part is −∇G(β)ᵀv_k(β), with Dv_k[e] = −H⁻¹DH[e]v_k + H⁻¹A_ke. It contributes −eᵀ∇²G v_k + aᵀDH[e]v_k − aᵀA_ke, which is (c₂+c₃+c₄)ᵀe.
- Each term is linear in e. Writing it as a vector gives (3), and then |c_kᵀe| ≤ ‖c_k‖_{H⁻¹}‖e‖_H.

For B_k, bound each term by Lemma 1:
- |tr(H⁻¹DH[e]H⁻¹A_k)| ≤ ν‖e‖τ_k (since A_k ⪰ 0), which gives ½ντ_kδ;
- |∇²G[e,v_k]| ≤ ½(ν₂+ν²)edf·δ s_k;
- |aᵀDH[e]v_k| ≤ ν‖a‖_H δ s_k ≤ ½ν²edf·δ s_k;
- |aᵀA_ke| ≤ ‖a‖_H δ ≤ ½ν·edf·δ, because A_k ⪯ H.

The sum is B_kδ. ∎

**Corollary 2a (second-order gradient) [P-lead, N].** g_k^{(2)} = g̃_k + C_k − c_kᵀq has error O(δ²), because e − q = O(νδ²).

**Corollary 2b (gamfit's gradient band is not a bound) [P, N].** decrement_bands.rs:84–86 charges |part.total − envelope| = |C_k| (the correction applied), plus rounding. By Theorem 2 the error left after the correction is c_kᵀe. This is unrelated to |C_k| ≈ |(A_kβ)ᵀq| ≤ s_kδ, which is the error *removed*.
- At a point where A_kβ ⟂ q the charge is 0, but the error is ‖c_k‖δ.
- In the numerical check the charge happened to exceed the true error. That was luck, not proof.

**Corollary 2c (value/gradient inconsistency) [P].** V_c is accurate to O(δ²), but g̃+C is accurate only to O(δ). Along a direction d, the slope implied by the values and the slope reported by the oracle therefore differ by c·δ·‖d‖. This is a *systematic* (smooth-in-ρ within a warm-start branch) error, not random noise, and §3.4 shows that systematic errors are what break line searches.

### 3.4 When value-tested line searches fail

Consider backtracking from α₀ with contraction ½ on the test
  φ̃(α) ≤ φ̃(0) + c₁αφ̃'(0) + c,   (A_c)
where |φ̃ − φ| ≤ ε_f, |φ̃'(0) − φ'(0)| ≤ e₀ and φ̃'(0) < 0.
- opt uses c = eps_f(f_k, τ_f) = τ_f·ε·(1+|f_k|), with τ_f = 1e3 (lib.rs:8481–8485, 1970).
- Berahas–Cao–Scheinberg use c = 2ε_f.
- A *certified* test uses c = −2ε_f.

Define a = (1−c₁)|φ̃'(0)| − e₀.

**Theorem 4 (value-tested Armijo) [P].**
(a) *Success.* If a > 0 and a² ≥ (9/4)·L_d‖d‖²·(2ε_f − c)⁺, then (A_c) holds on an interval [α₋, α₊] with α₊ ≥ 2α₋. So halving from any α₀ ≥ α₋ succeeds.
(b) *Failure is possible.* If (1−c₁)|φ̃'(0)| < e₀ + √(2L_d‖d‖²(2ε_f − c)), there exist a smooth φ with φ'' ≡ L_d‖d‖² and admissible errors (φ̃'(0) = φ'(0) − e₀; ε(0) = −ε_f, ε(α) = +ε_f for α > 0) for which (A_c) fails for every α > 0. The search then ends by a step floor, a trial cap, or never.

*Proof.* Write L = L_d‖d‖².
- (a): φ(α) ≤ φ(0) + α(φ̃'(0) + e₀) + ½α²L. So (A_c) holds whenever aα − ½Lα² ≥ 2ε_f − c. This set is an interval with roots α_± = (a ± s)/L, where s = √(a² − 2L(2ε_f−c)). The condition α₊ ≥ 2α₋ is equivalent to 3s ≥ a, i.e. a² ≥ (9/4)L(2ε_f − c).
- (b): with those errors, (A_c) reads (1−c₁)α|φ̃'(0)| − e₀α − ½Lα² ≥ 2ε_f − c. Its maximum over α is a²/(2L) < 2ε_f − c. ∎

**Two floors.** For a (quasi-)Newton direction with d̂ = d/‖d‖, L_d = d̂ᵀH_Vd̂:
- noise floor: g_min^val = √(2L_d(2ε_f−c))/(1−c₁) ≈ 2√(L_dε_f) for c ≪ ε_f;
- gradient-error floor: g_min^grad = e₀/((1−c₁)‖d‖).

In the Newton metric, the noise floor reads λ < √(2(2ε_f−c))/(1−c₁).

**Corollary 4a (dead zone of the value certificate) [P].** The certificate ½λ̂² ≤ band_f with band_f = ε_f certifies λ ≤ √(2ε_f). By 4(b), the search may stall for any λ < 2√ε_f/(1−c₁). The window √(2ε_f) < λ < 2√ε_f is a factor-√2 zone where neither succeeds, and hence a `line_search_failed` that is not certified.
- With band_f = 2ε_f, the resolution of the difference φ̃(α) − φ̃(0), the certificate covers λ ≤ 2√ε_f and the zone closes, up to the factor 1/(1−c₁) and opt's positive cushion, both of which shrink it further.
- In the prostate failure the ratio |g|/bound is 9.604e-6/7.302e-6 = 1.32, which is suggestively inside a √2 window. [C]

**Corollary 4b (relaxed Armijo never fails, but certifies nothing) [P].** With c = 2ε_f (Berahas et al. 2021, eq. 2.1), 4(a) holds for every a > 0. The search succeeds whenever (1−c₁)|φ̃'(0)| > e₀. Accepted steps may, however, *increase* V by up to 2ε_f each, so the iterates only reach a neighbourhood of size O(√(Lε_f)) (BCS Thm 3.13; Sun & Nocedal 2023, Thm 6). It is a search rule, not a certificate.

**Remark (iid noise versus systematic error) [N].**
- Theorem 4(b) needs the errors to be *aligned*: low at 0 and high at every trial. Independent errors at the ≤ 50 trials of a backtrack almost never align. In check_linesearch with iid or smooth multiscale noise up to ε_f = 1e-6, the strict search usually still reached λ ~ 1e-10, far below √ε_f.
- A *systematic* error does align. The gradient error of Corollary 2c is systematic, and so is a smooth value bias (check_inconsistency, κ-bias on V). With the cushion removed, the κ = 1e-4 case failed exactly at the predicted floor: |g̃ᵀd̂| = 9.38e-5 ≤ κ|aᵀd̂|/(1−c₁) = 9.92e-5.
- Superlinearly converging BFGS may jump over the zone. It did for κ = 1e-8 and 1e-6, which terminated at |g| = 4e-13. The floors are therefore necessary conditions for failure, not predictions of it.

### 3.5 Gradient-certified line search

The identity φ(α) − φ(0) = ½α(φ'(0) + φ'(α)) − (α³/12)φ'''(ξ) gives the test
  T(α) := ½α(φ̃'(0) + φ̃'(α)) + ½α(e₀ + e_α) + R(α) ≤ c₁αφ̃'(0),   R(α) = α³‖d‖³M₃/12.   (B)

**Theorem 5 (soundness) [P].** If (B) holds, then φ(α) − φ(0) ≤ c₁αφ̃'(0). This is a true sufficient decrease, stated against the oracle slope. If (A_{−2ε_f}) holds, i.e. φ̃(α) ≤ φ̃(0) + c₁αφ̃'(0) − ε_f(0) − ε_f(α), the same conclusion holds. *Proof:* immediate from the identity, respectively from |φ̃ − φ| ≤ ε_f. ∎

**Theorem 6 (the search cannot fail above the gradient-error floor) [P].** Suppose (1−c₁)|φ̃'(0)| > 2ē. Then (B) holds for all α ∈ (0, α*], where α* > 0 solves
  ½α L_d‖d‖² + α²M₃‖d‖³/12 = (1−c₁)|φ̃'(0)| − 2ē.
Halving from α₀ therefore terminates after at most ⌈log₂(α₀/α*)⌉ + 1 trials. No cap is needed, and ε_f does not enter.

*Proof.* φ̃'(α) ≤ φ'(α) + e_α ≤ φ'(0) + αL_d‖d‖² + e_α ≤ φ̃'(0) + e₀ + e_α + αL_d‖d‖². Hence T(α) ≤ αφ̃'(0) + α(e₀+e_α) + ½α²L_d‖d‖² + R(α). This is ≤ c₁αφ̃'(0) iff (1−c₁)|φ̃'(0)| ≥ e₀ + e_α + ½αL_d‖d‖² + α²M₃‖d‖³/12. ∎

**Corollary 6a (StepSizeTooSmall becomes a certificate event).** If (1−c₁)|φ̃'(0)| ≤ 2ē, the direction's descent is unresolvable at the oracle's accuracy. The search returns `DescentUnresolvable{φ̃'(0), ē}` to the certificate instead of failing.
- For d = −B⁻¹g̃ this says g̃ᵀB⁻¹g̃ ≤ 2ē/(1−c₁).
- The certificate (§6.3) then decides stationarity from g̃, its bands and the analytic outer Hessian.

**On R(α).** The remainder is O(α³‖d‖³). Near a minimum, a Newton step has α‖d‖ ≈ λ, so R/(α|φ'(0)|) = O(λM₃). It is negligible exactly in the regime that matters. A rigorous a-priori bound on M₃ for LAML is open (O1). With the analytic outer Hessian, the corrected trapezoid φ(α) − φ(0) = ½α(φ'(0)+φ'(α)) − (α²/12)(φ''(α) − φ''(0)) + O(α⁵) moves the unknown constant to fifth order. The numerical check used (B) with R omitted.

### 3.6 Inner-accuracy schedule

**Theorem 7 (Carter-type schedule) [P].** Let the inner decrement satisfy
  δ · Σ_k|d_k|·β_k ≤ θ(1−c₁)|g̃ᵀd|,   with θ < ½,
at both endpoints of each trial, where β_k is a certified gradient-band coefficient (‖c_k‖_{H⁻¹}, or B_k) and ε_g,k = β_kδ. Then 2ē ≤ 2θ(1−c₁)|g̃ᵀd| < (1−c₁)|g̃ᵀd|, and Theorem 6 guarantees the search succeeds.
- The rule is implicit (d depends on g̃, and g̃ depends on δ). It is resolved by *continuing* the inner Newton iteration until it holds. This is not a retry: each inner step is quadratic and the condition is monotone in δ.
- θ trades inner work against α*. It is an algorithmic parameter, not a tolerance.

This is the line-search analogue of Carter's (1991) relative-gradient-error condition for trust regions, and of the summable-tolerance condition of Pedregosa (2016).

**Theorem 8 (the arithmetic floor is cheap) [P, N].** In the quadratic region νδ < (3−√5)/2, Newton's decrement satisfies δ₊ ≤ νδ²/(1−νδ)² (Nesterov 2004, Thm 4.1.14, scaled by ν), which is strictly less than δ. Going from δ₀ to δ_floor therefore takes ⌈log₂(log(δ_floor ν)/log(δ₀ν))⌉ steps, typically 1–2 after a warm start.

The floor itself is derived from rounding. The residual is computed as fl(r), and by Higham (2002, §3.1) |fl(r) − r| ≤ ρ_r componentwise with
  ρ_r = γ_{n+c}·|X|ᵀ|s| + γ_{p+1}·|S_λ||β|,
where s is the per-row score vector and c its per-row evaluation depth (logistic: s = y − μ, c = 2). Then
  δ_floor = sup_{|z|≤ρ_r} ‖z‖_{H⁻¹} ≤ Σ_j ρ_{r,j}·√((H⁻¹)_jj),   (4)
using |(H⁻¹)_ij| ≤ √((H⁻¹)_ii(H⁻¹)_jj). The right-hand side is cheap, because diag(H⁻¹) is already formed for the traces.

Newton in floating point stagnates at a limiting residual of order ‖ρ_r‖ plus a solve-error term u·κ(H) (Tisseur 2001, Thm 2.2). The stop rule is therefore:

> Stop at the first k with δ̂_k ≤ δ_floor(β_k). Otherwise, if νδ̂_k < (3−√5)/2 and δ̂_{k+1} ≥ δ̂_k, the inner iteration is at its limiting accuracy: stop and pass δ̂_k to the bands.

All the numbers in this rule come from the analysis: γ_m from rounding, (3−√5)/2 from the self-concordant contraction and monotonicity from Theorem 8.

At the floor:
- ε_f's inner term is ¼(ν₂+2ν²)edf·δ_floor², at or below 1e-22·edf;
- ε_g,k's inner term is β_kδ_floor ~ 1e-12;
- the oracle is consistent to rounding, and the √(Lε_f) floor of Theorem 4 is the only one left for a value-tested search, which is why §3.5 is also needed.

---

## 4. Numerical checks

The model is penalized logistic regression (`model.py`):
- n = 500; intercept plus two cubic B-spline smooths with k = 10 each, sum-to-zero by QR, second-difference penalties; p = 19, K = 2.
- log|S|₊ is block-separable.
- Everything is analytic; finite differences are used only to check.

The analytic outer gradient agrees with the central-difference gradient of the exactly solved V to relative error 1.7e-9 and 3.8e-9. Newton from β = 0 reaches the floor in 5 iterations, with δ_floor = 4.3e-12 from (the |H⁻¹| form of) (4).

**Check 1–3 (`check_errors.py` → `check_errors.out`).** At ρ = (1, 3): ν = 0.351, ν₂ = 0.128, edf = 7.39, s = (1.52, 0.86), τ = (5.10, 6.51). Perturbation β* + e with ‖e‖_H = δ, taking the worst of 20 random directions:

| δ | \|ΔV_unc\|/δ | \|ΔV_cor\|/δ² | \|Δg_unc\|/δ (k=1,2) | \|Δg_cor\|/δ (k=1,2) |
|---|---|---|---|---|
| 1e-9 | 0.057 | (rounding) | 0.67, 0.39 | 0.014, 0.014 |
| 1e-6 | 0.061 | 0.11 (rounding) | 0.69, 0.66 | 0.013, 0.018 |
| 1e-5 | 0.077 | 8.0e-3 | 0.71, 0.46 | 0.012, 0.024 |
| 1e-3 | 0.073 | 8.9e-3 | 0.58, 0.49 | 0.015, 0.022 |
| 1e-1 | 0.069 | 7.7e-3 | 0.66, 0.40 | 0.015, 0.020 |

- The bound constants are ½ν·edf = 1.295 (V uncorrected), ¼(ν₂+2ν²)edf = 0.691 (V corrected), B^unc = (5.80, 4.48) and B = (4.29, 3.62).
- The error orders are as predicted: V_unc first order, V_c second order, and g first order both before and after the IFT correction. The correction reduces the gradient error about 40× but does not change its order.
- All bounds hold.

**Band audit** (gamfit's charges against the truth at a random e):

| δ | V charge (gamfit) | V derived (2) | V true | g charge (gamfit) | g derived B_kδ | g true |
|---|---|---|---|---|---|---|
| 1e-7 | 4.2e-9 | 6.9e-15 | 1.1e-13 | 9.4e-9, 1.1e-8 | 4.3e-7, 3.6e-7 | 2.7e-10, 9.6e-10 |
| 1e-5 | 2.3e-8 | 6.9e-11 | 5.7e-14 | 8.1e-7, 8.8e-7 | 4.3e-5, 3.6e-5 | 2.5e-8, 4.5e-8 |
| 1e-3 | 2.3e-5 | 6.9e-7 | 3.7e-9 | 3.0e-4, 2.6e-4 | 4.3e-3, 3.6e-3 | 7.2e-6, 1.3e-6 |

The gamfit V charge over-charges by 10⁴–10⁶×, because it is first order where the truth is second order. At δ = 1e-5 that is enough to trip `ObjectiveNotResolvable`.

**Check 4 (`check_sensitivity.py`).** The sharp bound is ‖c_k‖_{H⁻¹} = (3.25e-2, 3.74e-2), which is 130× and 97× tighter than B_k.
- The worst realized error divided by the sharp bound is 1.000 for k = 1, attained in the adversarial direction H⁻¹c_k; random directions give 0.4–0.6.
- The second-order gradient of Corollary 2a has error/δ² = (2.3e-2, 1.0e-2) for δ ≥ 1e-5. At δ = 1e-7 it is rounding.

**Check 5 (`check_linesearch.py`, iid and smooth noise).**
- Setup: ρ* = (−0.4986, 2.0030), V* = 299.655, eig(H_V) = (0.775, 1.489). BFGS from (4, −1) with opt's step floor, ε_g = 1e-9 and ε_f ∈ {1e-12, …, 1e-6}.
- Strict search (opt's cushion): terminal λ is usually 1e-10 to 1e-9 because noise lets some trial through. It stopped early in 2 of 24 runs: λ = 1.5e-6 (iid, ε_f = 1e-8) and 2.7e-5 at maxit (smooth, ε_f = 1e-6).
- Certified search: always ends in 10 iterations with the `DescentUnresolvable` event at λ = 7e-12 to 1.4e-9, the ε_g level, independent of ε_f.

**Check 6 (`check_schedule.py`).** A genuine inexact oracle: warm-started damped Newton, a strict opt-like search, and REML_TOL = 1e-5.

| inner rule | outer its | evals | inner Newton its | oracle \|g̃\| | true \|g\| | true λ | slope inconsistency |
|---|---|---|---|---|---|---|---|
| gamfit (clamp 0.1‖g‖ on ‖r‖ ≤ tol√(np)) | 10 | 15 | 21 | 7.2e-16 | **9.0e-8** | 7.7e-8 | 6.1e-8 |
| floor (Theorem 8) | 9 | 14 | 36 | 4.2e-13 | 4.2e-13 | 4.1e-13 | 2.8e-10 |
| Carter θ = ¼ (Theorem 7) | 16 | 27 | 21 | 5.3e-13 | 1.2e-11 | 9.8e-12 | 8.5e-10 |

Under the gamfit rule the outer run "converges" to a stationary point *of the stale oracle*. The warm start satisfied ‖r‖ ≤ tol√(np) with 0 inner steps, so V and g at the frozen β̃ form a consistent smooth function whose minimizer is off by λ ≈ 8e-8, and the certificate's |C_k| band does not see it. The floor rule costs about one Newton step per evaluation and removes the problem.

**Check 7 (`check_inconsistency2.py`).** A smooth value bias κaᵀ(ρ−ρ*) with the gradient unbiased; see the Remark in §3.4. With the cushion removed, κ = 1e-4 fails at |g̃ᵀd̂| = 9.38e-5 against a predicted floor of 9.92e-5 (Theorem 4(b)). With opt's cushion 6.7e-11 the failure was masked, because the bias difference κα‖d‖ fell below the cushion at tiny α. The certified search ends at |g| = 4.2e-13 in 10 iterations for every κ.

---

## 5. Literature

- **Berahas, A. S., Cao, L., Scheinberg, K. (2021).** Global convergence rate analysis of a generic line search algorithm with noise. *SIAM J. Optim.* 31(2):1489–1518. [read]
  - Modified Armijo condition with +2ε_f, eq. (2.1); Algorithm 2.2.
  - Theorem 3.13: convergence to a neighbourhood whose size is governed by ε_f and ε_g.
  - Used in Corollary 4b. Its relaxed cushion is exactly opt's cushion when ε_f is the oracle's error.
- **Sun, S., Nocedal, J. (2023).** A trust region method for noisy unconstrained optimization. *Math. Program.* 202:445–472. [read]
  - Relaxed ratio, eq. (7), with r = 2/(1−c₂), eq. (8); typical constants c₀ = 0.1, c₁ = ¼, c₂ = ½, ν = 2; M = ½(L_B + L), eq. (17).
  - Lemmas 1–3; β = √((rε_g)² + 8νr²(1/c₀−1)Mε_f), eq. (31).
  - Theorem 6: iterates enter C₁ = {‖g‖ ≤ (r+1)ε_g + β/2}. With the typical constants β/2 ≈ 24√(Mε_f), the same √(Lε_f) floor as Theorem 4. Theorem 7 gives the corresponding C₂ result.
  - This is the right model for `TrustRegionPolicy::noise_aware` (§6.2 O4).
- **Carter, R. G. (1991).** On the global convergence of trust region algorithms using inexact gradient information. *SIAM J. Numer. Anal.* 28(1):251–265. Global convergence under a relative gradient error ‖g̃ − g‖ ≤ ξ‖g̃‖ with ξ below a constant set by the acceptance parameters. Theorem 7 is the line-search counterpart, with ξ tied to (1−c₁)/2. [cited from memory; the exact constant was not re-verified]
- **Conn, A. R., Gould, N. I. M., Toint, Ph. L. (2000).** *Trust-Region Methods.* MPS-SIAM. §8.4 covers inexact gradients and §10.6 inexact function values. [section numbers from memory]
- **Devolder, O., Glineur, F., Nesterov, Yu. (2014).** First-order methods of smooth convex optimization with inexact oracle. *Math. Program.* 146:37–75. The (δ, L)-oracle model. Primal gradient methods do not accumulate oracle error, while fast methods accumulate O(kδ). This supports keeping a non-accelerated, noise-aware outer method.
- **Pedregosa, F. (2016).** Hyperparameter optimization with approximate gradient (HOAG). *ICML*, PMLR 48:737–746. Inner tolerance sequences ε_k with Σε_k < ∞ give convergence of the approximate hypergradient method. Theorem 7 is the per-step, certificate-driven version.
- **Grazzi, R., Franceschi, L., Pontil, M., Salzo, S. (2020).** On the iteration complexity of hypergradient computation. *ICML*, PMLR 119:3748–3758. Implicit-differentiation hypergradient error is linear in the inner error. Theorem 2 gives the exact linear map c_k for LAML, including the log-det terms that the generic analysis omits.
- **Hager, W. W., Zhang, H. (2005).** A new conjugate gradient method with guaranteed descent and an efficient line search. *SIAM J. Optim.* 16(1):170–192.
  - Approximate Wolfe conditions (2δ−1)φ'(0) ≥ φ'(α) ≥ σφ'(0), used once φ(α) ≤ φ(0) + ε_k, with ε_k = 10⁻⁶|f| a heuristic.
  - This is the precedent for a derivative-based acceptance. Test (B) is its certified form, with ε_k replaced by the oracle's ε_f and ε_g.
- **Xie, Y., Byrd, R., Nocedal, J. (2020).** Analysis of the BFGS method with errors. *SIAM J. Optim.* 30(1):182–209. Curvature pairs must be formed on steps long enough that sᵀy exceeds the gradient noise. This motivates the pair test in §6.2.
- **Nesterov, Yu. (2004).** *Introductory Lectures on Convex Optimization.* Kluwer. §4.1: Theorem 4.1.13 (distance to the minimizer in terms of the decrement) and Theorem 4.1.14 (quadratic convergence of Newton's decrement).
- **Bach, F. (2010).** Self-concordant analysis for logistic regression. *Electron. J. Statist.* 4:384–414. Logistic loss is generalized self-concordant, which gives κ_i ≤ 1 in Lemma 1.
- **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM. Lemma 3.1 (γ_m) and §3.1 (componentwise bounds for inner products), used for ρ_r.
- **Tisseur, F. (2001).** Newton's method in floating point arithmetic and iterative refinement of generalized eigenvalue problems. *SIAM J. Matrix Anal. Appl.* 22(4):1038–1057. Theorem 2.2 gives the limiting accuracy of Newton's method, which justifies the stagnation stop in Theorem 8.
- **Wood, S. N. (2011).** Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B* 73(1):3–36. Source of the LAML criterion and its derivative structure. It is used for the mathematics only, not as a target.

---

## 6. Consequences for gamfit and opt

### 6.1 Inner solve (gam-solve)

**Delete:**
- `PIRLS_INNER_TOLERANCE_FLOOR = 1e-6` and `pirls_tol = reml_tol.min(...)` (estimate/smoothing_correction.rs:14, 39);
- `ADAPTIVE_KKT_ETA = 0.1` and `ADAPTIVE_KKT_FLOOR_REML_DIVISOR = 100` (reml/state_caches.rs:6, 8);
- the adaptive schedule construction (reml/gradient_hessian.rs:6839–6862) and the full-fidelity override (gradient_hessian.rs:6660–6671);
- `effective_kkt_tolerance`'s clamp (pirls/convergence.rs:29–45);
- the √(np)/natural-scale test `certifies_kkt` (pirls/state.rs:159–161), which is not affine-invariant and has no derivation;
- `near_stationary_kkt`'s 10× band (pirls/state.rs:~175);
- the inner-cap schedule and constants `first_order_inner_cap_schedule`, `INNER_CAP_FLOOR = 3`, `INNER_CAP_CEILING = 64` and `INNER_CAP_CONVERGENCE_OVERRIDE_RATIO = 0.01` (rho_optimizer/bridges.rs:2263–2389), which are caps.

**Build: the arithmetic-floor stop** (Theorem 8). Per inner iterate:
1. Compute δ̂ = √(rᵀH⁻¹r) from the factor already in hand.
2. Compute ρ_r = γ_{n+c}|X|ᵀ|s| + γ_{p+1}|S_λ||β|. This is one extra pass of |X|ᵀ over |s|, the same cost as the score.
3. Compute δ_floor = Σ_jρ_{r,j}√((H⁻¹)_jj), from diag(H⁻¹), which the trace code already forms.
4. Compute ν = max_iκ_i√h_i, from the leverages h that the moving-Hessian correction ∇G = ½Xᵀ(w'⊙h) already needs.
5. Stop if δ̂ ≤ δ_floor. Otherwise, if νδ̂ < (3−√5)/2 and the last step did not decrease δ̂, stop at the limiting accuracy and hand δ̂ to the bands.

This replaces both the KKT stop and the adaptive η schedule. The measured cost is about one extra Newton step per outer evaluation.

**For inner solvers that are not quadratically convergent** (first-order or active-set paths): use Theorem 7's rule instead. Continue the inner iteration while δ̂·Σ_k|d_k|β_k > θ(1−c₁)|g̃ᵀd|, with θ = ¼ as the recommended algorithmic parameter.

**Seed screening** (`SEED_SCREENING_INNER_CONVERGENCE_TOLERANCE = 1e-3`, gradient_hessian.rs:7797): the derived condition for ranking two candidates whose V differ by ΔV is ¼(ν₂+2ν²)edf·δ² + (ε_f rounding) < ½|ΔV|, using the corrected value. Rank with that rule, or not at all.

### 6.2 opt (lib.rs)

**O1. Noise-aware oracle and line search.** Extend the objective's return type with certified bands:

```
struct Evaluation { f: f64, g: Array1<f64>, f_band: f64, g_band: Array1<f64> /* componentwise ε_g,k */ }
```

When a caller provides no bands, the honest default is f_band = u·|f| and g_band = u·|g|, i.e. representation error only. This replaces `eps_f`/`eps_g` with τ_f = 1e3/1e2 and τ_g = 1e2 (lib.rs:1970–1976, 8395–8396).

Line search along a descent direction d (φ̃'(0) = gᵀd < 0), with e(ev) = Σ_j|d_j|·ev.g_band_j:

```
e0 = e(ev0)
if (1-c1)*|φ0| <= 2*e0 { return DescentUnresolvable { slope: φ0, band: e0 } }
α = α0                                          // 1 for (quasi-)Newton
loop {
    if all_j |α d_j| < u |x_j| { return DescentUnresolvable (representation) }
    ev = oracle(x + α d); eα = e(ev); φα = ev.g·d
    value_ok = ev.f - f0 + (ev0.f_band + ev.f_band) <= c1*α*φ0                 // (A_{-2ε_f})
    grad_ok  = 0.5*α*(φ0 + φα) + 0.5*α*(e0 + eα) + R(α) <= c1*α*φ0           // (B)
    if value_ok || grad_ok { accept }
    if (1-c1)*|φ0| <= e0 + eα { return DescentUnresolvable { slope: φ0, band: max(e0,eα) } }
    α = safeguarded_cubic(φ0, φα, ev.f - f0, α)  clipped to [α/2·(…), α/2]   // any contraction in (0,1) works
}
```

- **Termination.** Theorem 6 bounds the number of trials by ⌈log₂(α₀/α*)⌉+1, so `BACKTRACKING_MAX_ATTEMPTS = 50`, `WOLFE_MAX_ATTEMPTS = 20` (lib.rs:3354–3355), `MAX_BACKTRACK_HALVINGS = 60` and `step_tolerance = 1e-12(1+‖x‖)+1e-16` (lib.rs:124, 8018–8031) are all deleted.
- **Remainder.** R(α) = α³‖d‖³M₃/12, or with an analytic Hessian the corrected trapezoid of §3.5 (open problem O1).
- **Curvature pair.** Update BFGS only if sᵀy > Σ_j|s_j|(g_band_j(x+αd) + g_band_j(x)), meaning the pair is resolvable (Xie–Byrd–Nocedal 2020). Otherwise keep B. This is a derived test, not a fallback.
- **Delete** the Wolfe c₁/c₂ adaptation (lib.rs:8960–8961, 9009–9013), the global-best salvage with `grad_drop_factor = 0.9/0.95` (lib.rs:8409, 9058–9066, 9712), `refresh_local_mode`'s 1e-2/5-success switches (lib.rs:8523–8534) and `ARMIJO_ROUNDOFF_EPS_MULTIPLE = 8` (lib.rs:126). With certified bands none of them has a role.
- **Fix the message.** `StepSizeTooSmall` is reported as "after 50 attempts" because the code substitutes the constant (lib.rs:9045), whatever the true count was. With O1 the event itself disappears.

**O2. Decrement verdict.** In `newton_decrement_verdict` (lib.rs:2521–2523), add the missing square term:

```
band_lambda_sq += 2|c|·δc/μ + δc²/μ + c²·curv_res/μ²
```

The formula is |c_true² − c²| ≤ 2|c|δc + δc², so the current code under-states the band whenever |c| < δc, and at c = 0 it certifies regardless of δc. [P]

**O3. Value resolution.** The certificate should compare against 2·ε_f, a difference of two evaluations; see Corollary 4a. Either let callers pass band_f = 2ε_f, or double it inside the verdict and document it.

**O4. Trust region.** In `TrustRegionPolicy::noise_aware(1e-12, 1e6, 1e-14)` (lib.rs:813), replace `noise_floor_rel` with the oracle's f_band, and use Sun–Nocedal's relaxed ratio
  ρ_k = (f̃(x) − f̃(x+s) + r·ε_f)/(m(0) − m(s) + r·ε_f),   r = 2/(1−c₂).
The radius bounds 1e-12 and 1e6 are hand bounds and should be deleted: the radius is controlled by the ratio test alone.

### 6.3 Bands and certificate (gam-solve)

**Value band ε_f** (`ObjectiveBand`, decrement_bands.rs:98–126). Keep `channels` (growth·Σ|parts|) and `factor` (½|log-det forward error|); both are derived rounding bounds. Replace `inner_residual = |cost_correction|` (decrement_bands.rs:113–119; objective.rs:539–549) with Theorem 1(b):
  E_inner = ¼(ν₂ + 2ν²)·edf·δ̂²,
with ν, ν₂ and edf from the leverages already formed. At the floor this is at rounding level. For β-independent curvature (Gaussian), ν = ν₂ = 0 and E_inner = |½rᵀH⁻¹r − ½rᵀH⁻¹r_true|, which is rounding; the value error ½rᵀH⁻¹r is then removed exactly by the correction up to O(δ³) terms that vanish.

**Gradient band ε_g,k** (decrement_bands.rs:84–86). Replace |part.total − envelope| with the cheap rigorous bound
  ε_g,k = growth·channels_k + B_k·δ̂,   B_k = ½ν(τ_k + edf) + (ν² + ½ν₂)·edf·s_k.
- τ_k = tr(H⁻¹A_k) is already a gradient term, and s_k = √((A_kβ)ᵀv_k) comes from the IFT solve v_k, so this costs nothing extra.
- If B_kδ̂ is the binding term, use the sharp ‖c_k‖_{H⁻¹}δ̂ from (3). It costs O(np²) per k for diag(X H⁻¹A_kH⁻¹ Xᵀ) and diag(X H⁻¹DH[v_k]H⁻¹ Xᵀ), about 100× tighter in the check.
- Optionally apply Corollary 2a (g − c_kᵀq), which makes the gradient O(δ²). With the floor stop this is unnecessary.

**Gradient consistency.** kkt.rs:78 differentiates only −½rᵀKr, but objective.rs:537 adds the moving-Hessian value term. Two options:
- accept the first-order inconsistency and charge it through ε_g,k (it *is* Theorem 2's c_kᵀe); or
- use the floor stop, which makes it rounding.

Do not leave it both uncharged and unremoved.

**Certificate.** Let λ̂² and band_λ² be computed as in opt's verdict with O2, from g̃, ε_g and the analytic outer Hessian with its band. Then:
- **Certified** iff ½λ̂² + band_λ² ≤ 2ε_f.
- **DescentUnresolvable** from O1 is the normal way the search ends, and it hands over to this test.
- If band_λ² > 2ε_f, the oracle's *gradient* is the limit. The inner accuracy required is then determined directly, not by retrying: solve Σ_j(2|c_j|δc_j + δc_j²)/μ_j ≤ 2ε_f for δ in ε_g = βδ + channels.
- The outer stationarity bound `outer_arithmetic_gradient_floor = max(tol, scale·√ε)` (rho_optimizer/run.rs:8913–8918) and its use as the solver band (run.rs:8954–8995) are deleted. √ε is the resolution of a *finite difference*; the gradient is analytic and its resolution is ε_g. The solver's gradient tolerance becomes the O1 event.
- The `ObjectiveNotResolvable` test against `outer_rel_cost_floor·(1+|V|)` (decrement_bands.rs:128–137; run.rs:8839–8844 with `COST_STALL_REL_TOL_FLOOR = 1e-7`, bridges.rs:302) compares a derived band with a user tolerance scaled by 1e-2. It is replaced by the derived precondition "inner at floor (δ̂ ≤ δ_floor) or at limiting accuracy". With that precondition band_f is rounding-level by construction.

### 6.4 Magic-constant audit

| site | constant | verdict | derived replacement |
|---|---|---|---|
| smoothing_correction.rs:14, 39 | inner tol 1e-6 | delete | δ̂ ≤ δ_floor, (4) |
| state_caches.rs:6 | η = 0.1 | delete | floor stop, or Theorem 7 with θ < ½ |
| state_caches.rs:8 | reml_tol/100 | delete | same |
| convergence.rs:29–45 | clamp(η‖g‖, floor, ceil) | delete | same |
| state.rs:159 | tol·√(np), tol·(1+scale) | delete | δ̂ in the H⁻¹ metric (affine-invariant) |
| state.rs:~175 | 10·tol | delete | none needed |
| gradient_hessian.rs:7797 | screening 1e-3 | delete | ranking rule, §6.1 |
| bridges.rs:2263–2389 | caps 3, 64; ratio 0.01; ×2/×3 | delete | quadratic convergence (Theorem 8) |
| bridges.rs:302 | 1e-7 | delete | derived band_f |
| run.rs:8839 | tol·1e-2 | delete | derived band_f |
| run.rs:8857 | 2·rel_cost_floor | derive | curvature resolvable iff ½\|μ\| > 2ε_f at unit step, i.e. μ_res = 4ε_f |
| run.rs:8913 | scale·√ε | delete | ε_g-based certificate |
| run.rs:7550 | 32ε | derive | progress iff drop > ε_f(prior) + ε_f(retried) |
| decrement_bands.rs:113 | \|cost_correction\| | replace | ¼(ν₂+2ν²)edf δ̂² |
| decrement_bands.rs:86 | \|C_k\| | replace | B_kδ̂ or ‖c_k‖δ̂ |
| lib.rs:1970, 8395 | τ_f = 1e3 / 1e2 | delete | oracle f_band |
| lib.rs:1974, 8396 | τ_g = 1e2 | delete | oracle g_band |
| lib.rs:8018 | 1e-12(1+‖x‖)+1e-16 | delete | representability u\|x_j\| plus Theorem 6 |
| lib.rs:3354–3355, 124 | 50, 20, 60 | delete | Theorem 6 trial bound |
| lib.rs:9009–9013, 8960 | c₂ → 0.5/0.1, c₁ → 1e-3, ×0.9/×1.1 | delete | none needed |
| lib.rs:8409, 9712 | grad_drop 0.9/0.95 | delete | none (salvage is a fallback) |
| lib.rs:126 | 8·ε | delete | oracle bands |
| lib.rs:813 | 1e-12, 1e6, 1e-14 | delete | Sun–Nocedal ratio with oracle ε_f |

The following remain and are *not* tolerances: c₁ ∈ (0, ½), any contraction factor in (0, 1), and θ in (0, ½) if the Carter rule is used. Each is a free parameter of a convergent algorithm, and no threshold depends on it beyond 1/(1−c₁).

### 6.5 Failing test clusters addressed

- **Binomial logit on prostate** (all-tests.log:786, 822): |Pg| = 9.604e-6 against bound 7.302e-6 = 490√ε, BFGS `StepSizeTooSmall`, V = 306.26.
  - opt's cushion there is 6.8e-11, and for ε_f ∈ [1e-11, 1e-10] and L ∈ [1, 10] the noise floor g_min^val is 6e-6 to 6e-5 (§3.4). So the stop is where Theorem 4 says a value-tested search may stop, and the requested bound is below it.
  - Fixed by O1 (gradient-certified search), the §6.1 floor stop (ε_g at rounding), the §6.3 certificate (O2 and O3, deleting the √ε floor at run.rs:8913) and the corrected message.
- **Binomial logit, railed** (all-tests.log:324, 871, 886): |Pg| = 2.28e-5, railed at ρ = 22.73.
  - O1 and the floor stop fix the line-search half, for the same reason as the prostate case.
  - The rail and box half belongs to the boundary-asymptotics and compactified-coordinates lanes.
- **x1 + cc(x2)** (all-tests.log:1009): "Newton decrement stopped contracting", ½λ̂² = 4.7e-5 against band_f = 3.95e-11.
  - This is not an oracle-noise failure: each step decreased V by 4e-5 to 6e-5, six orders above band_f.
  - My only contributions are the honest bands (O2, §6.3), so that its verdict is right. The flat direction is a rail/asymptote problem, and "2 of 2 Newton steps" is a cap.
- **Multinomial (ARC)**, **survival dim = 6**, **transformation/Box-Cox/Yeo-Johnson timeouts**:
  - The noise-aware verdict and bands apply as they are.
  - The inner-accuracy results need ν and ν₂ for those likelihoods. Theorem 2's c_k is family-generic (w', w'' become derivatives of the joint Hessian), but Lemma 1's constants are not available for non-self-concordant families (O2).
  - The ARC trust-region acceptance should use O4.
- **Iso-κ Matérn joint REML** (|Pg| = 0.331): outside the noise regime by five orders. Not addressed here.

---

## 7. Open problems

- **O1. A rigorous M₃ for (B).** This needs a certified bound on |∂³V/∂α³| along a segment for LAML. Leverage-based bounds similar to Lemma 1, applied one level up in ρ, look feasible: ∂_ρ of v_k and H⁻¹ are all expressible through H⁻¹A_k and D_βH. Until then, (B) is rigorous only conditionally on M₃, or to O(α⁵) with the analytic outer Hessian.
- **O2. Non-self-concordant families.** For Weibull/AFT, Cox, transformation, Box-Cox and Yeo-Johnson models, κ_i = |w_i'/w_i| is unbounded in the tails, and for location-scale models the joint Hessian is indefinite off the mode.
  - Theorem 2 stays valid as a first-order statement, with c_k computed from the family's third and fourth derivative tensors.
  - The segment constant needed to make it rigorous, a local ν on {‖β − β̃‖_H ≤ δ/(1−νδ)}, has to come from the family.
- **O3. Rigorous remainders.** "[P-lead]" results carry O(δ³) (value) and O(δ²) (gradient) remainders. Making them rigorous means replacing ν by ν/(1−νδ) and bounding w''' in the same way, which is routine for logistic but not yet written.
- **O4. Joint ρ+κ.** For Matérn range and anisotropy parameters, the basis X itself depends on the hyperparameter. c_k gains the terms from ∂X/∂κ, and the oracle-noise analysis has to be redone.
- **O5. Multinomial.** The inner Hessian is block-coupled (w is a matrix-valued per-row weight). Lemma 1 generalizes with κ_i replaced by an operator-norm bound of the per-row third-derivative tensor relative to the per-row Fisher block. This is not yet done.
- **O6. Validation in the Rust engine.** Every number in §4 comes from the Python model. After the lanes implement §6, the prostate and binomial-railed fits should be re-run with the band components logged (δ̂, δ_floor, ν, B_kδ̂, E_inner and the O1 event).
