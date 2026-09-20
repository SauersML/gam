# Inner-problem convergence theory and certificates for gamfit

Slug: `inner-convergence`. Topic: the inner problem β̂(ρ) = argmin_β f(β; ρ), the penalized negative log-likelihood, for all gamfit families. This covers global convergence, exact stopping certificates, non-convex families, separation and ρ → −∞, an audit of the current inner solver, and how the inner error feeds the outer (REML/LAML) certificate.

Status tags: **[proven]** means a complete proof is given here or cited exactly. **[numerical]** means checked in `SP/theory/inner-convergence/check_gsc.py`. **[conjecture]** or **[open]** are labelled as such.

---

## 1. Summary

1. **[proven] Leverage-GSC certificate (Thm A).** Take any row-wise generalized self-concordant family: logistic, Poisson-log, softmax/multinomial, or Cox partial likelihood. At any computed β̃, let λ be the Newton decrement and κ = M·max_i √(x_iᵀH̃⁻¹x_i), which is M times the square root of the largest leverage-to-weight ratio. If κλ < 1, then a minimizer exists and is unique, ‖β̃ − β̂‖_{H̃} ≤ r̄ = −ln(1−κλ)/κ, and there is a closed-form bound on the objective gap f(β̃) − f*. This needs no strong convexity, so it works when S_λ is rank-deficient, and it is strictly sharper than Sun & Tran-Dinh (2019, Thm 4). Numerically, true error / r̄ = 0.97–1.00 once κλ ≲ 0.06.
2. **[proven] Closed-form damped Newton (Thm B).** The step is α = ln(1+c)/c with c = M‖X d‖_∞, computed exactly from the Newton direction d. It guarantees the decrease f(β+αd) ≤ f(β) − λ²·((1+c)ln(1+c) − c)/c² and contraction λ₊ ≤ 2√(1+c)(1 − ln(1+c)/c)·λ ≲ κλ². There is no line search, no step halving, no Levenberg–Marquardt ridge, and no phase-switch constant. When S_λ ≻ 0, an a priori iteration bound exists. That bound is a termination proof, not a cap.
3. **[proven] Mixed barrier + GSC step (Thm B′).** Transformation and survival models add −log h′ barrier rows, which are standard self-concordant but not GSC. For these, the step α = min(ln(1+c)/c, 1/(1+ρ_b)) is monotone, stays inside the domain {h′ > 0} with no box, and tends to 1. Here ρ_b is the largest barrier-row local norm. The current custom-family step α = 1/(1+λ) with the (3−√5)/2 switch (`inner_blockwise_fit.rs:220-235`) is Nesterov's standard-SC step. It has **no guarantee** for logistic or softmax rows, because those are not standard self-concordant.
4. **[proven] Floating-point stopping rule with no tolerance parameter.** A rigorous bound λ_fp on the rounding noise in the computed decrement is obtained from γ_n error analysis of g = Xᵀφ′ + S_λβ. Stop at the first iterate where another exact Newton step could not certifiably reduce the bound. That is λ̂ ≤ λ_fp·(1+ψ)/(1−ψ), where ψ = ψ(ĉ) is the proven contraction factor. Because convergence is quadratic, this costs 1–2 extra iterations from a warm start. Numerically, logit reaches λ̂ = 5e-15 in 8 evaluations cold, and Poisson in 9.
5. **[proven, numerical] Inexact inner solves inject FIRST-order error into the outer.** The LAML value error is |V(β̃) − V(β̂)| ≤ gap + ½(e^{κr}−1)·edf_data (Prop 6). The LAML gradient error has leading coefficient √(λ_k β̃ᵀS_kβ̃) + ½(e^{κr}−1)tr(H̃⁻¹λ_kS_k) (Prop 7). Both are O(r), not O(r²). This is the mechanism behind the binomial and prostate BFGS `StepSizeTooSmall` failures, the "Newton decrement stopped contracting" failures, and the multinomial stalls. It interacts with the outer-driven inner caps of 3–64 iterations (`rho_optimizer/bridges.rs:2311-2449`), the adaptive-KKT η = 0.1 (`reml/state_caches.rs:6`) and the 1e-3 screening tolerance (`reml/gradient_hessian.rs:7914`).
6. **[proven] Existence and separation.** If a separating direction exists inside null(S_λ), no minimizer exists for any ρ (Albert–Anderson restricted to the null space). Otherwise one exists. The a-posteriori certificate κλ < 1 *proves* existence. Divergence is diagnosed exactly by an LP (Konis 2007), not by an iteration cap. With a double or null-space penalty, S_λ ≻ 0, so a minimizer always exists. As ρ → −∞, κ grows like e^{max|η|/2}. The iteration count grows mildly (7 → 33 evaluations from ρ = 4 to −16) and the certificate still reaches about 1e-13.
7. **[proven] Convexity catalogue.**
   - Convex: canonical GLMs, multinomial softmax, Cox partial likelihood (Breslow and Efron forms), cause-specific competing risks, Fine–Gray weighted partial likelihood, and transformation models with a log-concave reference (normal, logistic, minimum-extreme-value, i.e. Weibull-AFT). The transformation models are convex on the domain {a′(y)ᵀθ > 0}, including interval and right censoring.
   - Not convex: Box–Cox and Yeo–Johnson in the transformation parameter, the frailty marginal likelihood, and Gaussian location–scale in (μ, log σ). For the last one the row Hessian determinant is −2r²e^{−4s}.
   - For the non-convex families: use a trust region (Moré–Sorensen) to reach a second-order point, with Fisher scoring as the PD metric. Then certify with a Brouwer/Banach ball on the *observed* Hessian (Thm C).
8. **[audit] The inner solver has many underived constants and several SPEC violations.** Examples: a gradient-descent fallback, a force-Fisher-after-2 rule, iteration caps of 300/400/1200, step-halving counts of 30/60, a polish cap of 8, and outer-scheduled inner caps with floor 3 and ceiling 64. §6 maps each one to its principled replacement and to the failing test clusters.

---

## 2. Setup and notation

- **Objective.** For fixed ρ,
  f(β) = Σ_{i=1}^n φ_i(x_iᵀβ) + ½ βᵀS_λβ, where S_λ = Σ_k λ_kS_k, λ_k = e^{ρ_k}, and S_k ≽ 0.
  - For multi-predictor families (multinomial, location–scale, transformation), the row term is φ_i(η_i) with η_i = X_iβ ∈ ℝ^{m}, and X_i is an m × p row block.
- **Derivatives.** Gradient g = Xᵀφ′(η) + S_λβ. Hessian H = XᵀWX + S_λ, with W = diag(φ_i″) (block-diagonal for m > 1).
  - H̃ = H(β̃) and ‖v‖_{H̃} = √(vᵀH̃v).
- **Newton quantities.**
  - Newton direction d = −H⁻¹g.
  - Newton decrement λ(β) = √(gᵀH⁻¹g) = ‖d‖_H.
  - This is affine-invariant (Deuflhard 2004), which is why every tolerance below is stated in it rather than in ‖g‖₂.
- **Leverage quantities.** q_i = x_iᵀH⁻¹x_i. Note w_iq_i = h_ii is the leverage, so q_i = h_ii/w_i.
  - For every δ: |x_iᵀδ| ≤ √q_i·‖δ‖_H (Cauchy–Schwarz in the H inner product).
- **Row generalized self-concordance (GSC, ν = 2 in Sun & Tran-Dinh's sense).** A scalar row satisfies |φ‴(η)| ≤ M·φ″(η).
  - Logistic (Bernoulli-logit): φ″ = μ(1−μ), φ‴ = φ″(1−2μ), so M = 1.
  - Poisson-log: φ‴ = φ″ = μ, so M = 1.
  - Min-extreme-value reference (Weibull / Cox–Snell) φ(h) = e^h − h: M = 1.
  - Normal reference φ = h²/2: M = 0.
- **Vector-valued rows.** For softmax (log-sum-exp of K−1 free logits plus a reference) and for each Cox risk-set term, |D³φ[u,u,v]| ≤ (max_k v_k − min_k v_k)·D²φ[u,u] ≤ 2‖v‖_∞·D²φ[u,u].
  - So M = 2 with respect to the ∞-norm of the row-block change.
  - The same leverage argument applies with q_i replaced by the largest diagonal entry of X_iH⁻¹X_iᵀ.
- **Leverage-GSC constant.** κ(β) = M·max_i √q_i(β).
- **Useful functions.**
  - ω̃(t) = (1+t)ln(1+t) − t.
  - h(c) = ω̃(c)/c², which decreases from ½ at c = 0.
  - ψ(c) = 2√(1+c)(1 − ln(1+c)/c).
  - c† = 1.17633 is the root of ψ(c†) = 1.
- **Floating point.** u = 2^{−53} and γ_n = nu/(1−nu) (Higham 2002, Lemma 3.1).

---

## 3. Results with proofs

### Lemma 1 (row-wise Hessian sandwich) [proven]

For GSC rows with constant M and any β, δ:

  e^{−M‖Xδ‖_∞} H(β) ≼ H(β+δ) ≼ e^{M‖Xδ‖_∞} H(β),

and ‖Xδ‖_∞ ≤ √(max_i q_i(β))·‖δ‖_{H(β)}. So for ‖δ‖_{H(β)} ≤ r,

  e^{−κr}H(β) ≼ H(β+δ) ≼ e^{κr}H(β).

*Proof.*
1. |(ln φ_i″)′| ≤ M, so φ_i″(η+s) ∈ [e^{−M|s|}, e^{M|s|}]·φ_i″(η).
2. Multiply by x_ix_iᵀ and sum.
3. The quadratic penalty is unchanged, and S ≽ e^{−a}S and S ≼ e^{a}S hold for a ≥ 0, so the bound extends to all of H.
4. The ∞-norm bound is Cauchy–Schwarz in the H(β) inner product.

This is the row form of Sun & Tran-Dinh (2019), Prop 8, eq. (16). They use M_f = M·max‖x_i‖₂ with the Euclidean norm, which is weaker by the factor ‖x_i‖₂√(λ_max(H⁻¹))/√q_i ≥ 1. ∎

### Theorem A (a-posteriori existence and accuracy certificate) [proven]

Let S_λ ≽ 0 (possibly singular) and H̃ ≻ 0. Let λ = λ(β̃) and κ = κ(β̃). If κλ < 1, then:

1. f has a unique minimizer β̂.
2. r := ‖β̃ − β̂‖_{H̃} ≤ r̄ := −ln(1−κλ)/κ = λ(1 + κλ/2 + O(κ²λ²)).
3. f(β̃) − f* ≤ min{ λ·r̄ , r̄²·(e^{a}(a−1)+1)/a² }, where a = κr̄. The second term ≈ λ²/2.

*Proof.*
1. Fix δ with ‖δ‖_{H̃} = r. By Lemma 1, applied at the fixed base point β̃ with its fixed q_i, we have δᵀH(β̃+tδ)δ ≥ e^{−κtr}r².
2. Taylor with integral remainder then gives
   f(β̃+δ) − f(β̃) ≥ g̃ᵀδ + r²∫₀¹(1−t)e^{−κrt}dt ≥ −λr + (κr − 1 + e^{−κr})/κ².
3. The right side is a function of r only. Its r-derivative is −λ + (1−e^{−κr})/κ, which is increasing. It is positive for r > r̄ exactly when κλ < 1, because (1−e^{−κr})/κ → 1/κ > λ.
4. So the bound → +∞ uniformly over directions, the sublevel set {f ≤ f(β̃)} is bounded, and a minimizer exists (f is continuous).
5. Strict convexity holds because H(β) ≽ e^{−κ‖β−β̃‖}H̃ ≻ 0 everywhere, so the minimizer is unique.
6. For the radius: at β̂ = β̃ + δ we have g(β̂) = 0. Then −g̃ᵀδ = ∫₀¹δᵀH(β̃+tδ)δ dt ≥ r(1−e^{−κr})/κ. Also −g̃ᵀδ ≤ λr. So 1 − e^{−κr} ≤ κλ, which gives r ≤ r̄.
7. For the gap: expand f(β̃) around β̂ along −δ. The Hessian at β̂ + t(β̃−β̂) = β̃ − (1−t)δ is ≼ e^{κ(1−t)r}H̃. So f(β̃) − f* ≤ r²∫₀¹ s e^{κsr}ds = r²(e^{a}(a−1)+1)/a². Convexity also gives f(β̃) − f* ≤ g̃ᵀ(β̃−β̂) ≤ λr. ∎

*Comparison.*
- Sun & Tran-Dinh (2019) Thm 4, at ν = 2, needs λ < √σ_min(∇²f)/M_f. That is unusable when S_λ is rank-deficient or X is poorly scaled.
- Boyd & Vandenberghe (2004) §9.6.3 gives f − p* ≤ λ² for λ ≤ 0.68, but only for *standard* self-concordant f. Logistic loss is not standard self-concordant: φ‴/φ″^{3/2} = (1−2μ)/√(μ(1−μ)) is unbounded.
- Thm A needs only κλ < 1, with κ computed at the iterate.

**Strong-convexity alternative [proven, weak].** When λ_min(S_λ) = μ > 0: ‖β̃ − β̂‖₂ ≤ ‖g̃‖₂/μ and gap ≤ ‖g̃‖₂²/(2μ). Numerically it is useless (∞ in every test here, since S is rank-deficient) and not affine-invariant. Do not use it.

### Theorem B (closed-form damped Newton for GSC families) [proven]

At β, let d = −H⁻¹g, λ = ‖d‖_H, c = M‖Xd‖_∞ (exact), and α = ln(1+c)/c (α = 1 if c = 0). Then α ∈ (0,1] and:

- (i) f(β + αd) ≤ f(β) − λ²·h(c), with h(c) = ((1+c)ln(1+c) − c)/c² ∈ (0, ½].
- (ii) λ(β+αd) ≤ ψ(c)·λ, with ψ(c) = 2√(1+c)(1 − ln(1+c)/c) = c(1 + O(c)).
- (iii) c ≤ κ(β)·λ, so λ₊ ≤ κλ²(1+O(κλ)). The rate is quadratic, and the full-step region is reached automatically as α → 1.
- (iv) The full step (α = 1) satisfies λ₊ ≤ e^{c/2}·((e^c − 1 − c)/c)·λ.

*Proof.*
1. Along β + td, ‖X(td)‖_∞ = tc/M, so H(β+td) ≼ e^{ct}H(β) by Lemma 1. Hence f(β+αd) ≤ f − αλ² + λ²(e^{cα} − 1 − cα)/c².
2. The minimizer over α satisfies e^{cα} = 1 + c, which gives α = ln(1+c)/c. Substituting gives (i).
3. For (ii), g₊ = ∫₀^α(H(β+td) − H)d dt − (1−α)Hd.
4. By Lemma 1, ‖H^{-1/2}(H_t − H)H^{-1/2}‖ ≤ e^{ct} − 1. So ‖g₊‖_{H⁻¹} ≤ λ[(e^{cα}−1)/c − α + 1 − α] = 2(1−α)λ.
5. Since H₊ ≽ e^{−cα}H = H/(1+c), we get ‖g₊‖_{H₊⁻¹} ≤ √(1+c)‖g₊‖_{H⁻¹}.
6. (iii) follows from Lemma 1's ∞-norm bound. (iv) follows from the same computation with α = 1. ∎

Remarks.
- (a) c < c† = 1.17633 guarantees λ₊ < λ even far from the optimum. For c ≥ c†, (i) still guarantees decrease.
- (b) Sun & Tran-Dinh (2019) Thm 2, eq. (27) and (29), use the same functional form with β = M_f‖d‖₂. Using c = M‖Xd‖_∞ is exact and is never larger.
- (c) Each iteration costs one evaluation of (f, g, H) and one Cholesky factorization. The accepted step is proven to decrease f, so there are no trial evaluations.

**Corollary B1 (a priori termination; not a cap) [proven].**
1. If S_λ ≻ 0, then q_i ≤ ‖x_i‖²/λ_min(S_λ) for all β. So κ ≤ κ̄ := M·max_i‖x_i‖₂/√λ_min(S_λ) uniformly.
2. By (i), (iii) and the monotonicity of h, each step decreases f by at least λ²h(κ̄λ) = ω̃(κ̄λ)/κ̄².
3. For any t₀ > 0 (a proof device only), the number of steps with κ̄λ ≥ t₀ is ≤ κ̄²(f₀ − f_low)/ω̃(t₀). Here f_low is any lower bound of f, for example 0 for Bernoulli/binomial deviance halves.
4. After that, λ₊ ≤ ψ(κ̄λ)λ, which gives O(log log(1/λ_fp)) further steps.
5. This proves the loop terminates, so no iteration cap is needed. The bound is an *assertion*: exceeding it indicates a bug, not a hard problem.
6. With singular S_λ, termination still follows from compactness of the level set (Thm A), but without an explicit constant [open: explicit bound through the null-space LP margin].

### Theorem B′ (mixed GSC + self-concordant barrier rows) [proven]

For transformation and survival models:
- f = Σ_i φ_i(x_iᵀβ) + Σ_j b_j(a_jᵀβ) + ½βᵀSβ, where b_j(s) = −ln s are the −log h′(y_j) terms.
- b_j is standard self-concordant: |b‴| = 2b″^{3/2}. It is not GSC with a constant M.
- Split λ² = λ_G² + λ_B² + λ_S² into data, barrier and penalty parts: dᵀXᵀWXd, dᵀAᵀW_bAd and dᵀSd.
- Let ρ_b = max_j |a_jᵀd|/(a_jᵀβ), so ρ_b ≤ λ_B ≤ λ.

Then α = min(ln(1+c)/c, 1/(1+ρ_b)) satisfies:
- f(β+αd) < f(β) whenever λ > 0;
- a_jᵀ(β+αd) > 0 for all j, i.e. the step stays in the domain with no box;
- α → 1 as λ → 0, so convergence is locally quadratic.

*Proof.*
1. On [0, α], the model derivative is −λ² + λ_G²(e^{ct}−1)/c + λ_B²·t/(1−ρ_bt) + λ_S²t. This uses Lemma 1 for the data rows and Nesterov (2004) Thm 4.1.6 row-wise for the barrier rows: b″(s+τ) ≤ b″(s)/(1−|τ|/s)².
2. For t ≤ ln(1+c)/c, (e^{ct}−1)/c ≤ 1. For t ≤ 1/(1+ρ_b), t/(1−ρ_bt) ≤ 1. Also t ≤ 1.
3. So the model derivative is ≤ 0, and < 0 on [0, α).
4. Domain: ρ_bα < 1 keeps every a_jᵀβ positive. ∎

This replaces α = 1/(1+λ) with the switch at (3−√5)/2 in `gam-custom-family/src/inner_blockwise_fit.rs:220-235`. That step is Nesterov (2004) Thm 4.1.12. It is valid only if the whole objective is standard self-concordant, and for logistic, softmax or extreme-value rows it is not.

Rows whose |φ‴|/φ″ is unbounded but locally finite are handled the same way, with M replaced by the row-local constant M_i(s) = sup_{|t|≤s}|φ‴(η_i+t)|/φ″(η_i+t) evaluated at s = |x_iᵀd|. Examples: censored-normal −ln Φ̄(h) and probit. The condition becomes a monotone scalar inequality in α [proven for monotone M_i; implementation detail open].

### Theorem C (certificate for non-convex or non-GSC families: Newton–Kantorovich in affine-covariant form) [proven]

Let H̃ ≻ 0 be the **observed** Hessian at β̃. Let θ(r) ≥ sup_{‖δ‖_{H̃}≤r} ‖H̃^{-1/2}(H(β̃+δ) − H̃)H̃^{-1/2}‖₂ and Θ(r) = ∫₀¹θ(tr)dt. If some r satisfies

  λ̃ + r·Θ(r) ≤ r and θ(r) < 1,

then the ball {‖β − β̃‖_{H̃} ≤ r} contains exactly one stationary point, and it is a strict local minimizer.

*Proof.*
1. Define T(β) = β − H̃⁻¹g(β). Then ‖T(β̃+δ) − β̃‖_{H̃} ≤ λ̃ + ‖δ‖Θ(‖δ‖), so T maps the ball into itself.
2. T has Lipschitz constant ≤ θ(r) < 1 in ‖·‖_{H̃}. Banach's theorem gives a unique fixed point, i.e. g = 0.
3. H ≽ (1−θ(r))H̃ ≻ 0 on the ball, so the point is a strict local minimum. ∎

Instances:
- **GSC rows:** θ(r) = e^{κr} − 1 and Θ(r) = (e^{κr} − 1 − κr)/(κr). The condition is feasible iff κλ̃ ≤ 2ln2 − 1 = 0.3863, attained at κr = ln 2. This is weaker than Thm A, because Thm A also uses global convexity.
- **General analytic rows** (location–scale, Box–Cox, frailty, and so on):
  - θ(r) ≤ max_i(|Δw_i|/w̃_i) if W̃ ≻ 0 (because Σ w̃_ix_ix_iᵀ ≼ H̃). Otherwise θ(r) ≤ ‖Σ_i|Δw_i| H̃^{-1/2}x_ix_iᵀH̃^{-1/2}‖ ≤ Σ_i |Δw_i|q_i.
  - |Δw_i| ≤ L_{3,i}·√q_i·r, where L_{3,i} = sup over |s| ≤ √q_i r of |φ_i‴(η̃_i + s)|. This is an interval bound on the analytic third derivative, which gamfit already computes for the LAML Hessian.
  - For m × m row blocks, use the block version with the spectral norm of the block third-derivative tensor.
  - Θ(r) ≤ θ(r) works as a simple upper bound.

### Proposition 4 (floating-point floor and the stopping rule) [proven]

1. **Gradient noise.** The computed ĝ satisfies |ĝ − g| ≤ δg componentwise, with
   δg = γ_{n+2}·|X|ᵀ(|φ′| + |y| + |μ|) + |X|ᵀ(|W|·γ_p|X||β|) + γ_{p+2}·|S_λ||β|.
   This is Higham (2002) Lemma 3.1 and eq. (3.4)–(3.5) for inner products. The term |X|ᵀ(|W|γ_p|X||β|) propagates the rounding error of η through φ′. The checks use the first and third terms; the middle one is of the same order and should be included.
2. **Decrement noise.** Using sup_{|e|≤δg} eᵀH⁻¹e ≤ δgᵀ|H⁻¹|δg, we get |λ̂ − λ| ≤ λ_fp := √(δgᵀ|H⁻¹|δg) + (Cholesky backward-error term).
3. **Cholesky term.** Higham Thm 10.3: the computed d solves (H + ΔH)d = −ĝ with |ΔH| ≤ γ_{3p+1}|R̂ᵀ||R̂|. With η_H = ‖H^{-1/2}ΔH H^{-1/2}‖ ≤ γ_{3p+1}‖|R̂ᵀ||R̂|‖₂‖H⁻¹‖₂, we have λ ≤ λ̂/√(1−η_H).
4. **Certified input to Thm A.** λ_cert = (λ̂ + λ_fp)/√(1−η_H).
5. **Stopping rule.** The true decrement satisfies λ ≤ λ̂ + λ_fp. After one more exact step (Thm B ii), λ₊ ≤ ψ(ĉ)(λ̂ + λ_fp), so λ̂₊ + λ_fp ≤ ψ(λ̂ + λ_fp) + 2λ_fp. Another step can therefore reduce the certified bound only if λ̂ + λ_fp > 2λ_fp/(1−ψ). The rule is: **stop at the first iterate with λ̂ ≤ λ_fp·(1+ψ(ĉ))/(1−ψ(ĉ))**, which is ≈ λ_fp for small ĉ.
   - This rule has no free parameter.
   - Termination follows from Thm B: λ → O(λ_fp) quadratically.
   - A *requested* accuracy λ_req > λ_fp from the outer (Prop 8) can stop earlier. It is never needed for correctness.

### Proposition 5 (existence, separation, and ρ → −∞) [proven]

Let N = null(S_λ). This is the same for all finite ρ when every λ_k > 0.

1. For binomial-logit: if there is v ∈ N with Xv ≠ 0, (2y_i−1)x_iᵀv ≥ 0 for all i (quasi-complete separation in N), then t ↦ f(β + tv) is non-increasing and the penalty is constant. If additionally some row has y_i ∈ (0,1), or the inequality is strict, the infimum is not attained. This holds for **every** ρ, so no smoothing parameter can rescue the fit.
2. Conversely, suppose no such v exists and N ∩ null(X) = {0}. Then f is coercive: the recession function of f is positive on every nonzero direction (Rockafellar 1970, §8 and Thm 27.1), so a minimizer exists. If N ∩ null(X) ≠ {0}, the minimizer set is an affine family. gamfit's identifiability constraints remove it.
3. The LP of Konis (2007) decides the condition exactly: find v ∈ N maximizing Σ(2y_i−1)x_iᵀv subject to (2y_i−1)x_iᵀv ≥ 0 and a normalization. It is an exact finite algorithm, not a derivative-free search.
4. With a double or null-space penalty, N = {0}, so a minimizer exists for all ρ.
5. As ρ → −∞, the penalty in the separating range directions vanishes, max|η̂| grows, and κ ≈ M/√(min_i w_i) grows like e^{max|η|/2}. By Cor B1 and Thm B, the number of damped steps grows like κ̄²; the quadratic phase is unaffected.
   - Numerically (§4 C3) the iteration count goes 7 → 33 from ρ = 4 to ρ = −16, and κ goes 0.41 → 2470. The certificate still reaches r̄ ≈ 1e-13.
   - Without the null-space penalty, λ does not go to zero (it stays at 1.3–2.1), κλ grows (5 → 66), and ‖β‖ grows linearly. The observable non-existence signature is that κλ < 1 is never reached while f keeps decreasing by the Thm B amount. The LP then confirms it.

### Proposition 6 (outer LAML *value* error from an inexact inner solve) [proven]

Let V(β; ρ) = f(β) + ½ log det H(β) − ½ log|S_λ|₊ (+ const), which is LAML evaluated at β. Let r = ‖β̃ − β̂‖_{H̃} ≤ r̄, and let edf_data = tr(H̃⁻¹XᵀW̃X) ≤ rank(X). Then

  |V(β̃) − V(β̂)| ≤ [f(β̃) − f*] + ½(e^{κr̄} − 1)·edf_data.

*Proof.*
1. Let B = H̃^{-1/2}XᵀW̃XH̃^{-1/2}. Then 0 ≼ B ≼ I and tr B = edf_data.
2. Row-wise, Ŵ_i/W̃_i ∈ [e^{−a}, e^{a}] with a = κr̄.
3. Upper side: log det Ĥ − log det H̃ ≤ log det(I + (e^a−1)B) ≤ (e^a−1)tr B.
4. Lower side: log det Ĥ − log det H̃ ≥ Σ_j log(1 − (1−e^{−a})b_j). Here log(1 − (1−e^{−a})b) is concave in b ∈ [0,1], equals 0 at b = 0 and −a at b = 1, so it is ≥ −ab ≥ −(e^a−1)b. ∎

**Consequence.** The error is **first order** in r. The reason is that ∇_β ½ log det H ≠ 0 at β̂: the envelope theorem cancels only ∇f. So an inner solve that stops at r injects value noise of order ½κ·r·edf into the outer line search.

### Proposition 7 (outer LAML *gradient* error) [proven for the f and S-trace parts; partial for the third-derivative part]

For G_k = ∂V/∂ρ_k evaluated with β̃ in place of β̂, split e_k = G_k(β̃) − G_k(β̂) into three parts.

1. **f-part:** ½λ_k(β̃ᵀS_kβ̃ − β̂ᵀS_kβ̂) = λ_kδᵀS_kβ̃ − ½λ_kδᵀS_kδ. Because λ_kS_k ≼ H̃, we have ‖λ_kS_kβ̃‖_{H̃⁻¹} ≤ √(λ_kβ̃ᵀS_kβ̃). Hence |e_k^f| ≤ r√(λ_kβ̃ᵀS_kβ̃) + ½r².
2. **Trace part, S-piece:** ½tr(H⁻¹λ_kS_k). Since e^{−κr}H̃⁻¹ ≼ Ĥ⁻¹ ≼ e^{κr}H̃⁻¹, this gives |e_k^{S}| ≤ ½(e^{κr}−1)·tr(H̃⁻¹λ_kS_k).
3. **Trace part, W′-piece:** ½tr(H⁻¹XᵀW′[X v_k]X), with v_k = −H⁻¹λ_kS_kβ.
   - It changes through H⁻¹ (factor e^{κr}−1), through v_k (factor e^{κr}−1 plus r), and through φ‴(η).
   - For logistic, |φ⁗| ≤ φ″, since φ⁗ = φ″(1 − 6φ″) and φ″ ≤ ¼. For Poisson-log, φ⁗ = φ″. So |Δφ‴_i| ≤ φ″_i(e^{M√q_i r} − 1)/M.
   - All three pieces are therefore O(r) with explicit constants [explicit assembled constant: open, §7].
4. **A-posteriori estimate.** ê_k = |G_k(β̃) − G_k(β₊)|, where β₊ is one Newton step from β̃. Since r₊ = O(κr²), this estimate is exact to leading order.

### Proposition 8 (the inner accuracy the outer actually needs) [proven given Props 6–7]

If the outer certifies |P G| ≤ τ_outer and runs an Armijo test with value resolution ε_V (derived by the floating-point lane from the conditioning of V), then an inner r is sufficient when

  gap(r) + ½(e^{κr}−1)·edf_data ≤ ε_V and max_k ê_k(r) ≤ τ_outer − |P G̃|.

The left side is monotone in r, and r ≈ 2ε_V/(κ·edf_data) to leading order. By Thm A, r̄(λ) ≤ r holds when λ ≤ λ_req := (1 − e^{−κr})/κ.

**Recommendation.** Always solve to the fp floor (Prop 4). With quadratic convergence and warm starts, this costs 1–2 extra Cholesky factorizations per outer evaluation. It makes ê_k ≈ 1e-12 or smaller, far below every certification bound in the failing log (smallest 7.3e-6). λ_req is only an optional early exit, and it must be derived as above, never a constant.

### Proposition 9 (convexity catalogue) [proven unless marked]

| Family (parametrization) | Inner convexity | Tool |
|---|---|---|
| Canonical GLM (logit, Poisson-log, gamma with log link only if canonical…) | convex, GSC M=1 | Thm A/B |
| Non-canonical GLM (probit, cloglog, log-binomial…) | observed H may be indefinite | Fisher scoring metric + Thm C on observed H |
| Multinomial softmax | convex, GSC M=2 (∞-norm on the row block) | Thm A/B (block q_i) |
| Cox PL (Breslow, Efron) | convex (log-sum-exp over risk sets), GSC M=2 | Thm A/B |
| Cause-specific competing risks | separable sum of Cox/parametric terms, convex | Thm A/B per cause |
| Fine–Gray weighted PL | convex (weights fixed, non-negative) | Thm A/B |
| Transformation model h(y)=a(y)ᵀθ + xᵀβ, log-concave reference (normal, logistic, min-extreme-value ⇒ Weibull-AFT, Cox–Snell) | **convex on {a′(y)ᵀθ>0}**, including right, left and interval censoring | Thm B′ + Thm A (barrier rows need Thm C's θ, or the SC sandwich) |
| Weibull-AFT in (β, log σ) | not convex; convex after the transformation reparametrization (1/σ, β/σ) | reparametrize |
| Box–Cox / Yeo–Johnson (transformation λ) | not jointly convex | trust region + Thm C |
| Gamma / log-normal frailty (marginal) | not convex in general | trust region + Thm C |
| Gaussian location–scale (μ, log σ) | not convex: row det = −2r²e^{−4s}; blockwise convex; convex in (μ/σ, 1/σ) | Fisher metric diag(e^{−2s}, 2) ≻ 0, trust region, Thm C |

Proofs.
- **Transformation model, uncensored row:** −ln f_Z(h) is convex because f_Z is log-concave and h is affine in the parameters. −ln h′ is a self-concordant barrier with M = 2.
- **Right censoring:** −ln(1 − F_Z(h)). The survivor function of a log-concave density is log-concave (Bagnoli & Bergstrom 2005, Thm 3 and its corollaries).
- **Interval censoring:** −ln(F_Z(h_b) − F_Z(h_a)). The map (u, v) ↦ ∫1{v ≤ z ≤ u}f_Z(z)dz is log-concave by Prékopa (1973), because it is the marginal of a log-concave function on the convex set {v ≤ z ≤ u}.
- **Location–scale:** the row Hessian is [[e^{−2s}, 2re^{−2s}], [2re^{−2s}, 2r²e^{−2s}]] with det −2r²e^{−4s}. The Fisher information is diag(e^{−2s}, 2). So Fisher scoring d = −F⁻¹g is a descent direction with a linear rate.
- **Non-convex families generally:** the final certificate must use the observed Hessian in Thm C. A Fisher-metric iterate is only a starting point.

---

## 4. Numerical checks

Script: `SP/theory/inner-convergence/check_gsc.py`, with outputs in `check_gsc.out` and `check_gsc_c8.out`, where SP = `/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad`. The design is a cubic B-spline (20 columns) with a second-order difference penalty, which is rank-deficient. Reference solutions were computed with mpmath at 40 digits. The stopping rule is the plain λ̂ ≤ λ_fp form. There is no iteration cap; the script raises if 10 000 iterations are exceeded, as an assertion.

- **C1 – closed-form damped Newton (Thm B).**
  - Logit, n = 300, p = 20, λ = 1, cold start β = 0: 8 evaluations to the fp floor, with f strictly monotone.
    - λ: 10.7, 5.12, 2.13, 0.579, 6.0e-2, 7.8e-4, 1.4e-7, 5.2e-15.
    - α: 0.591, 0.717, 0.826, 0.933, 0.992, 0.9999, 1, 1.
    - λ_fp = 2.8e-12.
    - Quadratic contraction is visible: 6.0e-2 → 7.8e-4 → 1.4e-7 ≈ κλ² with κ ≈ 1.
  - Poisson: 9 evaluations. λ goes 50 → … → 1.7e-3 → 7.9e-7 → 1.8e-13, with λ_fp = 4.4e-12.
- **C2 – Thm A tightness (true r / r̄).**
  - Logit: 0.70 (κλ = 0.58), 0.974, 1.000, 1.000, 1.009. The last is at 5e-15, which is rounding level.
  - Poisson at κλ = 0.98: r̄ = 7.5 versus true 2.35.
  - The gap bound held everywhere except one entry: 2.84e-14 against a bound of 1.02e-14 at λ = 1.4e-7. That entry is the rounding of f itself (|f| ≈ 145, so u|f| ≈ 3e-14), not a bound failure. The gap bound must also be read against the fp floor of f.
  - The strong-convexity bound was ∞ throughout, because S is rank-deficient.
- **C3 – near separation (Prop 5).** y = 1{x > 0.5}, separable by the linear trend, which lies in null(S). Double penalty, warm starts:

  | ρ | evals | κ | cond(H) | max\|η\| | final r̄ |
  |---|---|---|---|---|---|
  | 4 | 7 | 0.41 | 5.7e2 | 1.5 | 1.1e-13 |
  | 0 | 9 | 0.95 | 6.9e1 | 3.9 | 5.6e-15 |
  | −4 | 12 | 5.0 | 7.0e1 | 9.1 | 5.9e-12 |
  | −8 | 21 | 45 | 1.3e3 | 35 | 8.8e-15 |
  | −12 | 29 | 335 | 4.7e3 | 94 | 1.1e-13 |
  | −16 | 33 | 2470 | 8.5e3 | 168 | 3.2e-13 |

  Without the null-space penalty, f decreases without bound, λ stays at 1.3–2.1, κλ grows 5 → 66, and ‖β‖ grows linearly. This is the non-existence signature from Prop 5.
- **C4 – outer gradient error (Prop 7).**
  - Exact dV/dρ = −1.903, and the f-part coefficient bound is 1.567.
  - For r = 1e-1 … 1e-6, |err|/r = 0.50, 0.41, 0.39, 0.36, 0.59, 0.11: first order, and inside the bound.
  - After **one** Newton step: r₊ = 2.9e-4, 1.1e-6, 2.9e-8, 4.2e-10, 1.5e-12, 2.7e-14, and |err| = 1.0e-4 … 8.9e-15.
  - The a-posteriori estimate |G(β̃) − G(β₊)| matched |err| to 3 digits in every case.
- **C6 – location–scale row Hessian determinant** at residuals 0, 0.5, 2: 0, −0.5, −8, matching −2r²e^{−4s} with s = 0. Not convex.
- **C7 – transformation model** (normal reference, right censoring, h = a(y)ᵀθ with a monotone basis): over 200 random feasible θ, min λ_min/|λ|_max = 1.46e-3 > 0. Convex. The finite-difference Hessian was used only here, as a test.
- **C8 – LAML value error (Prop 6).** Logit, edf_data = 8.39, κ ≈ 1:

  | r | \|ΔV\| | bound |
  |---|---|---|
  | 1e-2 | 1.3e-3 | 4.2e-2 |
  | 1e-4 | 7.7e-7 | 4.2e-4 |
  | 1e-6 | 3.4e-8 | 4.2e-6 |
  | 1e-8 | 6.4e-10 | 4.2e-8 |

  The bound holds. |ΔV|/r is 0.03–0.06 for r ≤ 1e-6, so the error is first order. The coefficient is ∇_β(½ log det H)·δ/r, which depends on the direction of δ.

---

## 5. Literature (precise citations)

- Nesterov, Yu. & Nemirovskii, A. (1994). *Interior-Point Polynomial Algorithms in Convex Programming.* SIAM. Self-concordance (Ch. 2).
- Nesterov, Yu. (2004). *Introductory Lectures on Convex Optimization.* Kluwer. §4.1: Thm 4.1.6 (Hessian sandwich (1−r)²H ≼ H(x+δ) ≼ H/(1−r)²), Thm 4.1.7 (function bounds), Thm 4.1.11 (existence), Thm 4.1.12 (damped Newton step 1/(1+λ)), and the quadratic region λ < (3−√5)/2. 2nd ed. (2018), *Lectures on Convex Optimization*, Springer, §5.1–5.2.
- Bach, F. (2010). Self-concordant analysis for logistic regression. *Electronic Journal of Statistics* 4:384–414. First use of |φ‴| ≤ Mφ″ for logistic loss.
- Sun, T. & Tran-Dinh, Q. (2019). Generalized self-concordant functions: a recipe for Newton-type methods. *Mathematical Programming* 178:145–213.
  - Def. 2 (GSC (M_f, ν)).
  - Prop 8, eq. (16): Hessian sandwich for ν = 2. Lemma 1 here is its row form.
  - Props 9–10: function bounds.
  - Thm 2, eq. (27), (29): damped step τ = ln(1+β)/β with β = M_f‖d‖₂.
  - Thm 3: local quadratic region, with constant d*₂ ≈ 0.12964 for ν = 2.
  - Thm 4: existence when λ < 2σ^{(3−ν)/2}/((4−ν)M_f).
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization.* CUP. §9.5 (Newton decrement, affine invariance) and §9.6.3 (f − p* ≤ λ² for λ ≤ 0.68, standard self-concordant only).
- Deuflhard, P. (2004). *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms.* Springer. Affine-covariant Newton–Kantorovich (Ch. 2–3), the basis of Thm C.
- Moré, J.J. & Sorensen, D.C. (1983). Computing a trust region step. *SIAM J. Sci. Stat. Comput.* 4(3):553–572.
- Conn, A.R., Gould, N.I.M. & Toint, Ph.L. (2000). *Trust-Region Methods.* MPS-SIAM. Ch. 6 (global convergence to second-order points).
- Albert, A. & Anderson, J.A. (1984). On the existence of maximum likelihood estimates in logistic regression models. *Biometrika* 71(1):1–10.
- Konis, K. (2007). *Linear programming algorithms for detecting separated data in binary logistic regression models.* DPhil thesis, University of Oxford.
- Kosmidis, I. & Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in binomial-response generalized linear models. *Biometrika* 108(1):71–82. Relevant to the Firth path, `loop_driver.rs:1521`.
- Rockafellar, R.T. (1970). *Convex Analysis.* Princeton. §8 (recession cones) and Thm 27.1 (attainment of the minimum).
- Hothorn, T., Möst, L. & Bühlmann, P. (2018). Most likely transformations. *Scandinavian Journal of Statistics* 45(1):110–134.
- Bagnoli, M. & Bergstrom, T. (2005). Log-concave probability and its applications. *Economic Theory* 26:445–469.
- Prékopa, A. (1973). On logarithmic concave measures and functions. *Acta Sci. Math. (Szeged)* 34:335–343.
- Cox, D.R. (1972). Regression models and life-tables. *JRSS B* 34:187–220. Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis of Failure Time Data*, 2nd ed., Wiley. Partial likelihood, convexity of Breslow/Efron.
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. Lemma 3.1 (γ_n), §3.1 (inner products), Thm 10.3 (Cholesky backward error).
- Wood, S.N. (2011). Fast stable REML and ML estimation of semiparametric GLMs. *JRSS B* 73(1):3–36. LAML definition; context only, not a design target.

---

## 6. Consequences for gamfit

Paths are relative to `SP/main_src/crates/`, a read-only checkout of main.

### 6.1 What to build: one certified inner Newton loop

```
inner_solve(β0, ρ) -> (β̃, InnerCertificate) | NonExistence(LP witness)
  β ← β0
  loop:
    evaluate f, g, W (analytic), H = XᵀWX + S_λ;  R ← chol(H)
        # Cholesky failure ⇒ H not PD ⇒ non-convex family path (6.1b); never add jitter
    d ← −H⁻¹g;  λ̂ ← √(−gᵀd)
    λ_fp, η_H ← Prop 4 (δg from |X|ᵀ…, |H⁻¹| via R; the p×p inverse is needed only at the
               candidate stop, or bound it by ‖δg‖₂²‖H⁻¹‖₂ earlier)
    ĉ ← M‖Xd‖_∞   (row-local M_i for non-GSC rows; ρ_b for barrier rows)
    if λ̂ ≤ λ_fp (1+ψ(ĉ))/(1−ψ(ĉ))  [or λ̂+λ_fp ≤ λ_req from Prop 8]: break
    α ← ln(1+ĉ)/ĉ                  (GSC)       |  min(ln(1+ĉ)/ĉ, 1/(1+ρ_b))  (with barrier)
    β ← β + αd
  κ ← M max_i √(x_iᵀH⁻¹x_i)  (one triangular solve R⁻ᵀXᵀ, O(np²), reuse for leverages)
  λ_cert ← (λ̂+λ_fp)/√(1−η_H)
  certificate: exists ⇔ κλ_cert < 1;  r̄ = −ln(1−κλ_cert)/κ;  gap ≤ min(λ_cert r̄, r̄²(e^a(a−1)+1)/a²)
               ΔV ≤ gap + ½(e^{κr̄}−1)edf_data;  ê_k from Prop 7 (or the one-step a-posteriori)
```

- There is exactly one step rule. There are no trial evaluations, so step halving and LM are gone.
- The theorems prove the objective decreases, so a check that "f increased" is an *assertion of a bug*, not a branch.
- Non-existence is established by κλ_cert failing to fall below 1 while Thm B decreases hold, and is confirmed by the Konis LP on null(S_λ) (Prop 5). It is reported as a model error ("separated in the unpenalized space"), not as a fit.

**6.1b Non-convex families** (location–scale, Box–Cox/Yeo–Johnson λ, frailty, non-canonical links):
- Use a Moré–Sorensen trust region in the Fisher metric F ≻ 0 until the observed H is PD and λ is small.
- Certify with Thm C on the observed H̃. θ(r) comes from interval bounds on the analytic φ‴, which gamfit already has for the LAML third derivatives.
- The trust-region radius updates use the ratio-of-reduction test. The theory needs only that the thresholds are fixed numbers in (0,1). Those are part of the algorithm's definition and have no statistical meaning to tune, but the lanes should state them as such [open: whether SPEC allows them; §7].
- Trust-region radius bounds (`joint_newton.rs:2857-2858`, 1e-12 and 1e6) are not needed. No ceiling is needed, because a radius beyond the unconstrained model minimizer has no effect. No floor is needed, because the iteration terminates through the Thm C certificate plus the Prop 4 noise floor, not through the radius.

### 6.2 What to delete or replace (file:line → replacement → derivation)

| Location | Current | Replace with | Derivation |
|---|---|---|---|
| `gam-solve/src/pirls/reweight.rs:793` | `for iter in 1..=options.max_iterations` | `loop` + Prop 4 stop; Cor B1 bound as debug assertion | Thm B, Cor B1 |
| `pirls/reweight.rs:761-765` | LM λ clamp [u,1/u], `lm_max_attempts = max_step_halving` | delete; α = ln(1+c)/c | Thm B |
| `pirls/reweight.rs:66-70` | Madsen factor clamp [1/3, 2] | delete (no LM) | Thm B |
| `pirls/reweight.rs:1197-1200` | "Fallback to gradient descent" | delete (SPEC: no fallbacks); non-PD H ⇒ 6.1b | Thm C |
| `pirls/reweight.rs:862, 1824, 1964` | `consecutive_fisher_fallbacks > 2` ⇒ force Fisher | delete; Fisher only as the metric in 6.1b, certification on observed H | Prop 9, Thm C |
| `pirls/reweight.rs:438, 523, 665` | AA1 reject threshold 3, AA α clamp [−1,1], `FlatStreak::new(2)` | delete Anderson acceleration (the Newton tail is already quadratic) | Thm B(iii) |
| `pirls/reweight.rs:944` | non-finite rescue | delete: Thm B/B′ never leave the domain | Thm B′ |
| `pirls/reweight.rs:204` | `kkt_tolerance * 10.0` | Prop 4 | Prop 4 |
| `pirls/reweight.rs:1333-1342` | noise-floor gain ratio | λ_fp from Prop 4 | Prop 4 |
| `pirls/reweight.rs:1589-1685` | exact decrement checked only on a plateau | decrement is the primary quantity every iteration | Thm A |
| `pirls/reweight.rs:1716-1727, 1793-1811, 2431-2527`; `pirls/newton_solve.rs:1483-1540` | soft acceptance ⇒ `StalledAtValidMinimum`, soft rescue | delete; a fit exists only with a Thm A/C certificate | SPEC |
| `pirls/reweight.rs:2150-2215, 2213, 2292` | undamped polish, `MAX_UNDAMPED_POLISH_STEPS = 8`, `0.25*‖β‖²` guard | delete: the main loop *is* the undamped quadratic tail (α → 1); no LM-ridge bias ever enters g (the #1122 envelope break disappears) | Thm B |
| `pirls/state.rs:131-180` (and L175 `10×tol`) | ‖g‖ < τ√n√p or relative test | λ̂ ≤ λ_fp(1+ψ)/(1−ψ) (affine-invariant) | Prop 4 |
| `pirls/convergence.rs:16-45` | rounding band + adaptive KKT clamp | Prop 4 | Prop 4 |
| `pirls/newton_solve.rs:1388-1390` | `ACTIVE_BOUND_REL_TOL=1e-6`, `ABS_TOL=1e-10` | domain constraints handled by Thm B′ (step never reaches the boundary); no active set for barrier-type domains | Thm B′ |
| `pirls/loop_driver.rs:1521` | step halving 60 (Firth) / 30 | delete; Firth path = non-convex class (6.1b) | Thm C |
| `pirls/loop_guard.rs:89, 97, 100-110, 222, 333-358` | damping floor/cap, retry predicates, reject factor 2.0, projection cap | delete | Thm B, Cor B1 |
| `pirls/loop_driver.rs:73, 76, 1627, 1631, 1878, 2017` | φ/shape/θ refresh ≤5, rel tol 1e-4 | treat (β, φ, shape) as one joint inner problem with Newton on all coordinates and Thm C certificate; if kept separate, tolerance from Prop 8 | Prop 8 [open §7] |
| `pirls/pls_solver.rs:476-520` | minimum-norm solve with p·ε·‖H‖ band | **keep** (derived from rounding) | Higham |
| `gam-solve/src/estimate/smoothing_correction.rs:14, 39, 49` | `PIRLS_INNER_TOLERANCE_FLOOR=1e-6`, `min(1e-6)`, `max_iterations(300)` | Prop 4 stop; no cap | Prop 4, Cor B1 |
| `gam-solve/src/reml/state_caches.rs:6, 8` | `ADAPTIVE_KKT_ETA=0.1`, divisor 100 | delete; inner always to fp floor, or λ_req from Prop 8 | Props 6–8 |
| `gam-solve/src/reml/gradient_hessian.rs:6566-6580, 6632, 6670, 6710-6726` | inner iteration budget, screening cap, `outer_inner_cap`, `min(tol, reml_tol/100)` | delete | Props 6–8 |
| `reml/gradient_hessian.rs:6747-6750, 7914` | `SEED_SCREENING_INNER_CONVERGENCE_TOLERANCE=1e-3` | delete; screening values need the same error bars (Prop 6), else the seed ranking is noise ∝ r·edf | Prop 6 |
| `reml/gradient_hessian.rs:6905-6932` | adaptive KKT | delete | Prop 8 |
| `reml/gradient_hessian.rs:7543-7665, 7950-7972` | capped solve returns `Err(PirlsDidNotConverge)` logged as "scheduled" | delete (no caps ⇒ no scheduled failures) | Cor B1 |
| `gam-solve/src/rho_optimizer/bridges.rs:2311, 2315, 2320, 2366-2449` (call sites 1899, 2481, 3683) | `INNER_CAP_*` 0.01 / 3 / 64, `first_order_inner_cap_schedule` | **delete entirely**; the log shows `inner_max_iterations=3` inside the outer line search | Props 6–7 |
| `gam-models/src/survival/base.rs:4872` | `max_iterations: 400` | Prop 4 stop; Thm B′ step | Thm B′ |
| `gam-models/src/multinomial.rs` (was `:116, 3505, 3508`) | ~~`MULTINOMIAL_FORMULA_INNER_TOL=1e-5`~~ and ~~`.max(tol)`~~ **removed (#4053)**: the inner target is the caller's `tol` floored only by the joint solve's own rounding band `max(tol·(1+max(‖∇L‖∞,‖Sβ‖∞)), band)` (#2812, #2976, #2977); 1200 cycles remains | λ_fp from Prop 4 (this *is* the saturated-row floor, computed instead of guessed); block-q_i κ; no cycle cap | Prop 4, Thm A |
| `gam-model-api/src/families/custom_family/options.rs:649, 665+` | `DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES=1200`, inner_tol 1e-6 | delete; Prop 4 | Prop 4 |
| `gam-custom-family/src/inner_blockwise_fit.rs:220-235` | α = 1/(1+λ), `SC_QUADRATIC_PHASE_THRESHOLD = (3−√5)/2` | α = ln(1+c)/c (GSC) or Thm B′ (with barrier) | Thm B, B′ |
| `inner_blockwise_fit.rs:2781, 2793, 4180, 4202, 4893` | `MAX_SADDLE_ESCAPES=2`, `MAX_ESCAPE_FACE_EXCHANGES=3`, `BLOCK_NEWTON_STEP_INITIAL=20`, `DIVERGENCE_FROZEN_LOGLIK_CYCLES=8`, `POLISH_MAX_ITER=16` | delete: convex families have no saddles; non-convex families use the trust region with second-order termination; divergence detected by Prop 5 LP | Thm B, C, Prop 5 |
| `gam-custom-family/src/joint_newton.rs:2021, 2302, 2307, 2584, 2722, 2737, 2857-2858, 3272, 3434, 3475, 3496, 3505, 5652-5660, 6181` | trust noise floor 1e-14, factors 0.5/2/4, radius [1e-12,1e6], 300 iterations, PCG constants | convex blocks: Thm B; non-convex: Moré–Sorensen with Prop 4 noise floor; PCG stopping by the decrement-noise criterion (residual in H⁻¹-norm ≤ λ_fp) | Prop 4, Thm C |
| `inner_blockwise_fit/exact_joint_fit.rs:821-823, 848, 914, 938, 958, 1053, 1172, 2715-2716, 3284, 5974` | stall constants 30/40/0.9, `LINEAR_RATE_PROJECTION_CAP=100`, `JOINT_TRUST_MAX_ATTEMPTS=24`, … | delete; Thm B/C | Thm B, C |
| `gam-custom-family/src/blockwise_solve.rs:713, 2211` | `MAX_BISECT=12`, 1e-14 | delete: step is closed form | Thm B |
| `gam-solve/src/gpu/reml_outer.rs:272, 304, 333`; `gpu_kernels/sae_resident.rs:1404-1405` | 10/100/10; 16, 1e-9 | same inner loop and Prop 4 stop on GPU (γ_n with the GPU accumulation length) | Prop 4 |
| `gam-solve/src/reml/reparameterized_inner.rs:216` | `orth_tol = 128·p·ε` | derive the factor from Higham Thm 19.4 (Householder QR, γ̃ ∝ mnu); 128 is not derived | Higham |

Blockwise (Gauss–Seidel) custom-family fits: for a jointly convex objective, replace block cycles with the joint Newton step of Thm B. The joint H is already assembled for LAML. Block-coordinate descent has no computable a-posteriori certificate comparable to Thm A.

### 6.3 How every remaining tolerance is derived

- **Inner stop:** λ_fp (Prop 4). It comes from γ_n bounds on the actual summation lengths n and p and from the Cholesky backward error. No user tolerance is involved.
- **Optional early stop:** λ_req = (1 − e^{−κr_req})/κ, where r_req solves gap(r) + ½(e^{κr}−1)edf_data = ε_V (Prop 8). ε_V is the outer value resolution from the fp lane.
- **Existence:** κλ_cert < 1, an exact inequality.
- **Non-convex certificate:** the Thm C inequality with interval θ(r).
- **Nothing else.** No iteration count, halving count or KKT ratio appears.

### 6.4 How the inner certificate feeds the outer

Every inner solve returns `InnerCertificate { λ_cert, κ, r̄, gap, value_err = Prop 6, grad_err[k] = Prop 7 (or the a-posteriori one-step), exists }`. The outer must:

1. Certify stationarity as |P G̃|_k + grad_err_k ≤ τ_outer. It must never compare G̃ alone when the inner is inexact.
2. Run Armijo and Wolfe comparisons with value_err on both sides. A decrease smaller than value_err(ρ) + value_err(ρ+αp) is "not resolved", which is different from "failed". With the fp-floor inner solve this margin is at rounding level, so it never triggers spuriously.
3. Never evaluate V at a capped or uncertified β̃. There is then no `PirlsDidNotConverge` inside line searches.

### 6.5 Mapping to the failing clusters (SP/q1561/all-tests.log)

- **Binomial / prostate BFGS `line_search_failed`** (|Pg| = 2.28e-5 vs 7.30e-6; 9.604e-6 vs 7.302e-6; StepSizeTooSmall after 50 attempts; "direction descended but no step improved").
  - Cause: the inner caps of 3–64 iterations (`bridges.rs:2311-2449`, log line `inner_max_iterations=3`) and adaptive-KKT η = 0.1 leave r > 0. Props 6 and 7 then put O(r·edf) noise into V and O(r) noise into G. When |Pg| is within 3× of the bound, that noise decides the line search.
  - Fix: delete the caps, stop at the fp floor, and apply 6.4(1)–(2).
  - This is the rail and box issue only in part. The box itself belongs to other lanes.
- **x1+cyclic(x2) "Newton decrement stopped contracting"** (`rho_optimizer/newton_polish.rs:123`) and **multinomial "declined certified optimum"** (hessian_psd = NO, 7 railed coordinates).
  - Cause: the outer Newton contraction test is impossible to satisfy when V and G carry O(r) inner noise. Multinomial also used the hard-coded 1e-5 (`multinomial.rs:116`, removed in #4053).
  - Fix: fp-floor inner with the block-q_i κ, error bars passed to the outer, and delete the 1e-5 and 1200-cycle constants.
- **Iso-kappa Matérn** (|Pg| 0.331 vs 0.0181; 3.622 vs 0.065).
  - The inner contribution is the envelope-theorem break from LM-ridge bias in g (#1122, `reweight.rs:2150-2215`). The joint (ρ, κ) gradient assumes ∇f(β̂) = 0 exactly.
  - Fix: Thm B has no ridge, so ∇f(β̃) = g exactly and the fp-floor stop makes it ≈ 0.
  - The size of these gradients (0.3–3.6) points mainly to an outer and identifiability problem (other lanes). The inner fix removes one confounder.
- **Survival transformation dim = 6** (Weibull-AFT; |Pg| 6.99e-2 vs 1.86e-3; one case hessian_psd = NO).
  - The inner problem is **convex** in the transformation parametrization (Prop 9), so inner non-convexity is *not* the cause.
  - Fix: Thm B′ step (domain-preserving, no active-set tolerances `newton_solve.rs:1388-1390`), no 400-iteration cap (`survival/base.rs:4872`), and fp-floor stop.
  - The outer hessian_psd = NO is an outer landscape question.
- **Timeouts** (transformation-normal, Box–Cox, Yeo–Johnson, frailty, competing-risks, gamlss-LS).
  - Likely cause: trial-evaluation loops, i.e. 30/60 halvings × LM attempts × up to 5 refresh cycles × 300–1200 iterations per outer evaluation. This is a plausible reading of the code, but it was **not measured** here.
  - Fix, convex ones (transformation-normal, competing-risks): Thm B/B′, which costs one evaluation per iteration and about 10 iterations cold.
  - Fix, non-convex ones (Box–Cox/Yeo–Johnson λ, frailty, gamlss-LS): trust region in the Fisher metric plus the Thm C certificate, and a joint (β, shape) Newton step instead of refresh loops (`loop_driver.rs:1627-2017`).

---

## 7. Open problems

1. **Explicit third-derivative constant in Prop 7.** A closed-form bound for the W′-trace piece of the gradient error (item 3) for every family, using |φ⁗| ≤ M₂φ″. It is proven in form for logistic and Poisson; the assembled constant has not been written out. Until then, use the one-step a-posteriori estimate, which C4 shows is accurate to leading order.
2. **Frailty with a non-log-concave mixing distribution, and Box–Cox/Yeo–Johnson λ.** The convexity structure is unknown. Is the profile over β convex in the transformation parameter at least locally, with a computable radius?
3. **Location–scale joint convexity.** The (μ/σ, 1/σ) reparametrization makes the Gaussian row convex. Does it survive penalized smooths on both predictors, where penalties are quadratic in the *original* coefficients? If a penalized convex reparametrization exists, gamlss-LS moves to Thm A/B.
4. **Certified dispersion and shape refresh.** Formulate (β, φ, shape) as one inner problem with an affine-covariant certificate (Thm C), or prove that the alternating refresh contracts with a computable rate.
5. **Pessimism of λ_fp.** The componentwise bound can be about 1000× the observed noise (C1: λ_fp ≈ 3e-12 while λ̂ reached 5e-15). A probabilistic rounding model (Higham & Mary 2019, √n instead of n) would tighten it, but would not be rigorous. Decide which the SPEC accepts.
6. **Explicit termination bound when S_λ is singular.** Cor B1 needs λ_min(S_λ) > 0. Bound κ along the iteration through the separation-LP margin of Prop 5.
7. **Large n or large p.** κ needs max_i x_iᵀH⁻¹x_i. The exact cost is O(np²) through R⁻ᵀXᵀ, the same order as forming H. For sparse or banded designs, the block-banded structure should give O(n·bandwidth²). An upper bound through the block-diagonal part of H is cheaper but looser.
8. **Trust-region acceptance thresholds.** The ratio thresholds in the non-convex path (η₁, η₂ ∈ (0,1)) are algorithm-defining, not statistical. Decide whether SPEC treats them as constants. An alternative is ARC with the exact third-derivative bound, which removes the ratio test (Cartis, Gould & Toint 2011, Math. Program. 127:245–295).
