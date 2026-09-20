# Literature survey: what published results say about gamfit's outer-optimization failures

Lane: `literature-survey` (convergence theory team). Scope: smoothing-parameter selection, mixed-model boundaries, noisy-oracle optimization, Matérn/GP identifiability, and bilevel/continuation theory. Each result is mapped onto the failing test clusters F1-F6 in `SP/q1561/all-tests.log`. `SP` is the session scratchpad.

Status labels:
- **[P]**: proven here, or a direct consequence of a cited theorem whose hypotheses I check.
- **[N]**: checked numerically (Section 4).
- **[C]**: conjecture.
- **[K]**: a source cited from my own knowledge of the literature, not re-read in this session. Its equation numbers are given only where I am confident of them.
- **[R]**: a source I read in full in this session. PDFs and text are in `SP/theory/literature-survey/papers/`.

Scripts are in `SP/theory/literature-survey/` and run with the shared venv `SP/theory/venv`:
- `check_tau_chart.py` / `.out`
- `check_d2.py`
- `check_matern_ridge.py` / `.out`
- `check_tail_newton.py` / `.out`

Sibling reports with the full proofs of some statements used here:
- `boundary-asymptotics.md`: τ-analyticity and the closed forms of a₁ and a₂.
- `compactification.md`: charts, corners and the lower face.
- `boundary-probability.md`
- `inexact-oracle.md`
- `glm-laml-landscape.md`

Where they overlap with this report, this report supplies the published provenance and the connection to the failing tests.

---

## 1. Summary

- **The published smoothing-parameter optimizers all handle the λ→∞ end by an ad hoc rule [R].** Wood 2008 §3.4 and Wood-Pya-Säfken 2016 §3 step 4 drop a coordinate when "∂V/∂ρ_i ≃ ∂²V/∂ρ_i² ≃ 0" at "working infinity", with thresholds they never define. Wood-Fasiolo 2017 caps λ at a "pre-defined upper limit". lme4 (Bates et al. 2015 §4) uses derivative-free BOBYQA/Nelder-Mead with a θ ≥ 0 box. None of these is admissible under SPEC, and all of them run into the same geometric fact.
- **That fact [P, N]:** in τ = e^{-ρ} = 1/λ the REML/LAML criterion is real-analytic at τ = 0, with V = V_∞ + cτ + O(τ²).
  - The face slope c is exactly the variance-component score statistic (Lin 1997; Crainiceanu-Ruppert 2004).
  - In the ρ chart this becomes an exponential tail, dV/dρ = −c e^{-ρ}. That tail causes F1, F3, F4 and F5.
  - In lme4's θ = τ^{1/2} chart it becomes a degenerate face with dV/dθ = 2θc → 0, which destroys strict complementarity.
  - Only the τ chart, or an analytic reparameterization of it such as the edf fraction in `compactification.md`, gives a face with a nonzero KKT multiplier.
- **Rails are the typical case, not a pathology [P, N].**
  - Under the null, P(face optimum) = P(Σ μ_s w_s² ≤ Σ μ_s), with w_s iid N(0,1) and μ_s the eigenvalues of ZᵀP₀Z. This is the Crainiceanu-Ruppert spectral representation.
  - It equals 0.68 on the test design (MC 0.6794 against spectral 0.6806).
  - A certifier that treats "railed" as suspicious is wrong about 2/3 of the time on null smooths.
- **Boxes can be replaced by a bound-constrained method on the true domain τ ≥ 0 [P].**
  - The theory is projected Newton or trust-region Newton: Bertsekas 1982, Conn-Gould-Toint 1988, Lin-Moré 1999 (TRON).
  - Under strict complementarity (c > 0), Calamai-Moré 1987 and Burke-Moré 1988 guarantee identification of the active face in finitely many steps.
  - In the ρ chart the face is never reached. Newton steps have constant unit length and the contraction is Θ = 1 [P, N: Θ = 0.99932 → 0.99999, and the decrement shrinks by e^{-1} per step].
- **The line-search and trust-region acceptance tests in the `opt` crate are noise-blind [R, P].**
  - The Armijo cushion is 8ε(1+|f|), about 5.5e-13 at f = 306 (opt `lib.rs:134`).
  - The ARC ratio is raw (opt `lib.rs:7781`).
  - The trust-region noise floor is 1e-14 (opt `lib.rs:782`).
  - The published fixes are Shi-Xie-Byrd-Nocedal 2022 (relaxed Armijo with +2ε_f) and Sun-Nocedal 2023 (the ratio (Δf + rε_f)/(Δm + rε_f) with r = 2/(1−c₂)). Both need only ε_f, and gamfit already derives it: `band_f` in `rho_optimizer/decrement_bands.rs:67`. They address F1 (StepSizeTooSmall after 50 halvings), F4 (ARC reject floor after 44 rejections) and part of F2.
- **The achievable gradient accuracy is a derived quantity [P].**
  - Function-value descent can be certified only while ½gᵀH⁻¹g > 2ε_f, i.e. only while ‖g‖ > 2√(M ε_f) along steepest descent with curvature M. This is the N₁ region of Shi et al. Thm 3.5.
  - Below that, StepSizeTooSmall is the predicted outcome and should become a certificate.
  - gamfit's decrement rung compares ½λ̂² with band_f, not 2·band_f. That leaves a window, ½λ̂² ∈ (band_f, 2band_f], where the line search must fail but the decrement cannot certify.
- **The Matérn failures F3 are a microergodic ridge, not a search failure [R, P, N].**
  - Zhang 2004 Thm 2 identifies only σ²κ^{2ν}, so REML is flat along ρ − 2νψ = const. The numerical fit gives slope 1.0000 for ν = ½, and ρ* − ψ converges to −1.59962.
  - The raw-chart Hessian has condition number 4.6e4 at ψ = −4, and its soft eigenvector is (0.707, 0.707), the ridge.
  - The κ → 0 end is a genuine face: the intrinsic (Brownian) limit. The profile criterion converges to it at rate κ¹ (ΔV ratio e² per two units of ψ). The intrinsic-limit minimiser c′ = 4.951125 matches κ/λ = 4.951150 at ψ = −10.
  - So the ψ coordinates do have a tail law after the change of chart. The decline "psi coordinate, no exponential tail law" at `rho_optimizer/run.rs:6528-6534` is wrong in the chart (μ = ρ − 2νψ, t = κ^{2min(ν,1)}).
- **Implicit-differentiation error bounds give a derived ε_g [R].**
  - Ablin-Peyré-Moreau 2020 Prop. 3: the implicit-function estimator of a gradient through an inner minimum has error O(‖z − z*‖²).
  - The LAML log-determinant part is not a minimum, so its error is O(‖z − z*‖).
  - Pedregosa 2016 Thm 2: summable inner tolerances give convergence.
  - Grazzi et al. 2020 Thm 2.2 (13): explicit constants.
- **Not usable under SPEC:**
  - α-theory and interval Newton (Smale 1986; Rump 2010) need rigorous bounds on higher derivatives.
  - PC priors (Simpson et al. 2017) need a user-chosen rate.
  - Chung et al. 2013 boundary-avoiding priors make the null unrecoverable.
  - INLA's grid/CCD exploration is a grid search.
  - lme4 and glmmTMB optimizers are derivative-free or boxed.

---

## 2. Setup and notation

**Parameters.**
- ρ_j = log λ_j are the outer coordinates of the smoothing parameters. τ_j = e^{-ρ_j} = 1/λ_j.
- θ_j = τ_j^{1/2} is the lme4 relative-covariance-factor coordinate. For a single penalty, λ = φ/σ_b², so θ = σ_b/σ_ε.
- ψ = log κ is the log inverse length scale of a Matérn term with smoothness ν.

**Design split for one penalty.**
- β = (β₀, β_R). β₀ spans the null space of S, and X = [X₀, X_R].
- The penalty is S = blockdiag(0, D) with D ≻ 0. Set Z = X_R D^{-1/2}.
- P₀ = I − X₀(X₀ᵀX₀)⁻¹X₀ᵀ. The eigenvalues of ZᵀP₀Z are μ_s ≥ 0, s = 1..r.
- In the Demmler-Reinsch basis, u_s is the coordinate of P₀y along the s-th eigenvector of P₀ZZᵀP₀, scaled so that Σ μ_s u_s² = ‖ZᵀP₀y‖².

**Gaussian REML, known φ.** Up to constants:

V(τ) = ½ Σ_s [ log(1 + τμ_s) + u_s² / (φ(1 + τμ_s)) ] + ½ yᵀ(residual part outside range) / φ + const. (2.1)

This is the restricted likelihood of y ~ N(X₀β₀, φ(I + τZZᵀ)). The random-effect form b ~ N(0, φτI) is the Ruppert-Wand-Carroll 2003 mixed-model representation. Check A in Section 4 verifies it directly from the penalized-regression form.

**Binomial LAML.**
- Limit fit on X₀: μ₀, w₀ = μ₀(1−μ₀), r₀ = y − μ₀, A₀ = X₀ᵀW₀X₀.
- H₀ = X₀A₀⁻¹X₀ᵀW₀. P_W = W₀ − W₀X₀A₀⁻¹X₀ᵀW₀.
- w₀′ = w₀(1 − 2μ₀).

**Noise model for the outer oracle.**
- f̃ = f + e_f with |e_f| ≤ ε_f.
- g̃ = g + e_g with ‖e_g‖ ≤ ε_g.
- gamfit's derived ε_f is `band_f = B_channels + B_factor + |E_r|`, computed in `outer_decrement_bands` (`crates/gam-solve/src/rho_optimizer/decrement_bands.rs:67`). The last term is E_r = ½rᵀH⁻¹r for the inner residual r.

**Failure clusters** (quotes from `SP/q1561/all-tests.log`, "did not certify"):

| Id | Test family | Signature |
|---|---|---|
| F1 | binomial logit (prostate), BFGS | \|Pg\|=2.280e-5 against bound 7.302e-6 (rung=solver-band, derived_standard=false); curvature_source=unavailable; #2 railed at the upper box 22.73 (margin 0.5); line_search_failed with StepSizeTooSmall after 50 attempts, 7 iterations, f=306.27 |
| F2 | x1+cyclic(x2), ARC | "Newton decrement stopped contracting": ½λ̂²=4.7e-5 against band_f=3.95e-11 |
| F3 | iso-kappa Matérn joint REML, including the statsmodels parity test | ρ railed at the lower edge ≈ −21; \|Pg\| 0.331 vs 0.0181 (129 iterations), 3.62 vs 0.065, 2.79e-2 vs 1.51e-2, 3.72e-2 vs 2.51e-2; "tail-snap declined: … psi coordinate, no exponential tail law" |
| F4 | survival Weibull AFT, dim 6 | BFGS MaxAttempts \|Pg\|=6.99e-2 vs 1.86e-3; ARC trust_region_reject_floor (radius 1e-12 after 44 rejections), \|Pg\|=0.272; probes show a clean tail g ∝ e^{-ρ} |
| F5 | multinomial penguins, 15 coordinates | rails at −15 (λ_min = −23), \|Pg\|=24.7; "inner mode's softest curvature 3.096e-8 at/below rounding band" (separation) |
| F6 | transformation-normal, Box-Cox, Yeo-Johnson, frailty, competing risks, gamlss-LS, interval-censored, tram | timeouts |

---

## 3. Results with proofs or derivations

### 3.1 The τ chart is analytic at the λ = ∞ face; the face slope is the score statistic [P, N]

**Proposition 1 (Gaussian).**
- (2.1) is real-analytic in τ on (−1/μ_max, ∞). The radius of convergence at τ = 0 is 1/μ_max.
- Its derivatives are

  V_τ(τ) = ½ Σ_s [ μ_s/(1+τμ_s) − u_s²μ_s/(φ(1+τμ_s)²) ],
  V_ττ(τ) = ½ Σ_s [ −μ_s²/(1+τμ_s)² + 2u_s²μ_s²/(φ(1+τμ_s)³) ].

- Therefore c := V_τ(0) = ½[ tr(ZᵀP₀Z) − ‖ZᵀP₀y‖²/φ ].

*Proof.* Each summand is a composition of log and rational functions of τ. Their only singularities are at τ = −1/μ_s. Differentiate term by term, then use Σμ_s = tr(ZᵀP₀Z) and Σμ_s u_s² = ‖ZᵀP₀y‖². ∎

`boundary-asymptotics.md` finds the same radius, stated as the smallest DR eigenvalue s_min = 1/μ_max.

**Identification with the score statistic.**
- c is −1 times the REML score for the variance component σ_b² = φτ at σ_b² = 0, scaled by φ. That is Lin 1997's score test statistic; Stram-Lee 1994 and Self-Liang 1987 give the mixture-χ² null laws for the corresponding LRT.
- The KKT condition of min_{τ≥0} V at τ = 0 is c ≥ 0. So "the optimum is on the face" means exactly "the one-sided variance-component score test does not reject at the level where its statistic is 0".

**Proposition 2 (binomial LAML).**
- With the notation of Section 2, c = c_score + c_drift, where
  - c_score = ½[ tr(ZᵀP_W Z) − ‖Zᵀr₀‖² ],
  - c_drift = ½ tr( A₀⁻¹ X₀ᵀ diag(w₀′ ∘ (I−H₀)ZZᵀr₀) X₀ ).
- Status: [P] as a first-order expansion, sketched below; [N] to 6e-7 relative.

*Sketch.*
1. Expand the inner mode: b̂ = τD⁻¹X_Rᵀr₀ + O(τ²), and η̂ = X₀β̂₀ + τ(I−H₀)ZZᵀr₀ + O(τ²).
2. The penalized log-likelihood contributes −½τ‖Zᵀr₀‖². This is the envelope term: the first-order change of β̂₀ does not enter, by stationarity.
3. In log|XᵀWX + λS| − r log λ, the λ-block gives τ tr(Zᵀ P_W Z) at first order.
4. W changes with η̂: W = W₀ + τ diag(w₀′ ∘ Δη) + O(τ²). This moves the log|A₀|-type term by ½τ tr(A₀⁻¹X₀ᵀ diag(w₀′∘Δη)X₀). That is the drift.
∎

This is the rank-2 drift already present in `crates/gam-solve/src/reml/rail_face_limit.rs:56`. Check B shows that without the drift term c is wrong in the third digit.

**Proposition 3 (θ-chart degeneracy).**
- In θ = τ^{1/2}, V_θ = 2θV_τ, so V_θ(0) = 0 for every data set.
- The face θ = 0 is therefore never strictly complementary.
- Any first-order test at the face (lme4's boundary check, or the KKT test of a bound-constrained method run in θ) sees a zero multiplier and must decide from second-order information, V_θθ(0) = 2c.
- [P]; [N] Check C: dV/dθ = 1.790e-1, 1.788e-2, 1.787e-3 against 2θc at θ = 1e-1, 1e-2, 1e-3.

This explains the Bolker et al. warning quoted in lme4 §4 about "asymptotically flat" surfaces. The flatness is an artifact of the chart.

**Corollary (the ρ chart).**
- dV/dρ = −τV_τ = −c e^{-ρ} + O(e^{-2ρ}), and d²V/dρ² = τV_τ + τ²V_ττ = c e^{-ρ} + O(e^{-2ρ}).
- Exact Newton in ρ therefore steps Δρ = −g/H = 1 + O(e^{-ρ}) forever. The half-decrement ½g²/H = ½c e^{-ρ} shrinks by e^{-1} per step.
- So the face at infinity is approached linearly in the decrement and never reached in the iterate.
- Deuflhard's contraction factor Θ_k = ‖Δρ_{k+1}‖/‖Δρ_k‖ → 1.
- [P]; [N] in `check_tail_newton.out`: Θ = 0.999317, 0.999749, …, 0.999988, and the decrement ratio is 0.3676 ≈ e^{-1}.

**Corollary (cancellation).**
- The assembled ρ-gradient ½[λb̂ᵀSb̂/φ + λ tr(H⁻¹S) − r] is a difference of O(r) terms whose true value is O(cτ). Its relative error is therefore about ε_mach·r·λ/c.
- The DR form of V_τ in Proposition 1 has no subtraction of near-equal terms at small τ.
- [N] Relative error of −λ g_ρ against the DR form, from `check_tail_newton.out`:

  | ρ | relative error |
  |---|---|
  | 10 | 4.4e-12 |
  | 20 | 2.6e-7 |
  | 25 | 6.9e-6 |
  | 30 | 8.0e-3 |
  | 33 | 3.5e-2 |
  | 36 | 1.31 |

- This is the "#2298 rail-cancellation class" that `rho_optimizer/run.rs:6562-6571` works around with a magnitude-only tie band. In the τ chart the phenomenon does not exist.

### 3.2 Boundary probability [P, N]

**Proposition 4.**
- Under the null model y = X₀β₀ + ε with ε ~ N(0, φI) and known φ, ZᵀP₀y ~ N(0, φZᵀP₀Z).
- Hence ‖ZᵀP₀y‖²/φ has the same law as Σ_s μ_s w_s² with w_s iid N(0,1).
- P(c ≥ 0) = P(Σ μ_s w_s² ≤ Σ μ_s).

*Proof.* Immediate from the Gaussian law, diagonalized in the eigenbasis of ZᵀP₀Z. ∎

**Profiled φ.** The analogue replaces φ by the REML scale estimate and gives a ratio of quadratic forms. This is the form in which Crainiceanu-Ruppert 2004 derive the finite-sample null law of the RLRT, including the point mass at λ̂ = ∞. I attribute the form to them but do not give an equation number; the paper was not available in full.

**Limit case.** When one μ_s dominates (here the top share is 0.992), the probability tends to P(χ²₁ ≤ 1) = 0.6827.
- [N] MC 0.6794 (20000 replicates) against the spectral value 0.6806.
- Crainiceanu, Ruppert, Claeskens and Wand 2005 report point masses of the same size for penalized splines [K].

**Consequences.**
- Face optima are the majority outcome whenever a smooth is not needed.
- A test suite whose expected answer is "interior optimum" on null data is testing a 1/3 event.

**The first-order face test is only locally sufficient [N].**
- In the scanned test, the face test agreed with the global minimum in 398 of 400 replicates.
- Over 2000 replicates (`check_d2.py`), 4 had c > 0 (local face minimum) and also a lower interior minimum near ρ ≈ −5:
  - c = 1.535, V∞ − min = 1.8e-3;
  - c = 1.43, V∞ − min = 1.05;
  - c = 0.98, V∞ − min = 1.57;
  - c = 1.71, V∞ − min = 0.49.
- So REML can be bimodal with a local optimum at the face.
- With a single dominant μ_s, (2.1) has at most one interior stationary point: 1 + τμ = u²/φ. Bimodality needs at least two comparable μ_s.
- This is an open problem for global certification (Section 7).

### 3.3 The λ → 0 end

- **Single penalty with X of full column rank.** −½ log|λS|₊ = −(r/2) log λ → +∞ while ½ log|XᵀX + λS| stays finite. So V → +∞ and the λ → 0 end is repelling. No face exists there. [P]
- **Overlapping penalties.** If range(S_j) ⊆ Σ_{k≠j} range(S_k), log|Σ_k λ_k S_k|₊ stays finite as λ_j → 0 and V is analytic in λ_j = e^{ρ_j} at 0. The face λ_j = 0 has the KKT condition ∂V/∂λ_j ≥ 0. [P]; `compactification.md` gives the full three-case analysis.
- **Binomial with separation (F5).** The inner mode diverges as λ → 0 and W → 0. The softest inner curvature 3.1e-8 in F5 is the signature of this. Albert-Anderson 1984 characterize existence of the MLE [K]. Kosmidis-Firth 2021 show that the Jeffreys-prior-penalized estimate is always finite in binomial GLMs [K]. Whether LAML has a finite limit here is unresolved in the literature (Section 7; see `glm-laml-landscape.md`).

### 3.4 Noise-limited acceptance tests [P]

**Proposition 5 (resolvability of descent).**
- Let f̃ have error at most ε_f, and let the step p have model decrease Δm = −gᵀp − ½pᵀHp.
- A measured decrease f̃(x) − f̃(x+p) certifies a true decrease only if it exceeds 2ε_f.
- The largest decrease along the Newton direction is ½gᵀH⁻¹g = ½λ_N², and along steepest descent with curvature M it is ‖g‖²/(2M).
- So no function-value line search can certify progress once ½λ_N² ≤ 2ε_f, or once ‖g‖ ≤ 2√(Mε_f) on steepest descent.
- Below that threshold, StepSizeTooSmall and MaxAttempts are the predicted outcome, not a failure of the method.

*Proof.* The measured difference differs from the true one by at most 2ε_f, and the true decrease is at most the model decrease up to third-order terms. ∎

This matches Shi et al. 2022 Thm 3.5: linear convergence to N₁ = {‖∇φ‖ ≤ max(A√(Mε_f)/γ*, Bε_g/γ*)}.

**Consequences for gamfit.**
1. **Armijo cushion.** opt's `armijo_roundoff_cushion` = 8ε(1+|f|) (opt `lib.rs:134`) models only the final floating-point rounding of f. It is 5.5e-13 at f = 306. It ignores the inner-solve and log-determinant error that `band_f` accounts for.
   - The Shi et al. relaxed Armijo (their (4.6)) is f̃(x+αp) ≤ f̃(x) + c₁αg̃ᵀp + 2ε_f.
   - With ε_f = band_f this is derived and constant-free apart from the classical c₁.
2. **Gap between decrement and line search.** gamfit's decrement verdict certifies when ½λ̂² ≤ band_f (see `decrement_stationarity_bound`, `decrement_bands.rs:288`).
   - In the window band_f < ½λ̂² ≤ 2·band_f the line search cannot certify a step, and the decrement verdict cannot certify stationarity.
   - The two thresholds should be made consistent: either the certificate uses 2·band_f, or the line search's noise allowance uses band_f/2 per evaluation. The error of a difference of two independently rounded evaluations is 2ε_f, so the former is the derived choice.
3. **F1 specifically.** F1 reports curvature_source=unavailable. The decrement verdict never ran, and the certificate fell back to the solver-band rung 7.302e-6 (derived_standard=false).
   - The true noise floor 2√(Mε_f) is unknown without M. The observed |g| = 2.28e-5 is consistent with a noise-limited stop whenever Mε_f ≥ 1.3e-10.
   - That is plausible at f = 306 with LAML log-determinants. It is [C] until M is measured.
   - The fix is to evaluate the analytic outer Hessian once at line-search failure and run the decrement verdict, rather than certifying against a solver-internal band.

**Proposition 6 (noise-tolerant trust region) [R].**
- Sun-Nocedal 2023 (7) replaces ρ_k = Δf̃/Δm by ρ_k = (Δf̃ + rε_f)/(Δm + rε_f), with r = 2/(1−c₂) (their (8)).
- Lemma 2: once Δ_k is small enough, the radius cannot keep shrinking.
- Thm 6: the iterates visit C₁ = {‖g‖ ≤ (r+1)ε_g + β/2} infinitely often.
- Thm 7: they then stay in a level set of width [2 + r(1−c₀)]ε_f.
- gamfit's ARC ratio at opt `lib.rs:7781` is the raw ratio. As the radius shrinks, Δm shrinks linearly while Δf̃ stays at noise size, so the ratio oscillates and the step is rejected. The 44 consecutive rejections to radius 1e-12 in F4 are exactly the failure Sun-Nocedal §1 describe.
- The TrustRegionPolicy default `noise_aware(1e-12, 1e6, 1e-14)` (opt `lib.rs:782`) uses a neutral band only when both reductions are below a relative floor of 1e-14. That floor is a magic constant, not ε_f.
- Adapting the ratio to ARC rather than a TR radius is [C]; the analysis carries over formally. Bellavia et al. 2019 (2.12)-(2.13) give the ARC analogue with relative accuracy on the model decrease.

### 3.5 The Matérn microergodic ridge and the κ → 0 face [R, P, N]

**Identifiability.**
- Zhang 2004 Thm 2: for d ≤ 3 and fixed ν, the Gaussian measures of Matérn(σ₁², κ₁) and Matérn(σ₂², κ₂) on a bounded domain are equivalent iff σ₁²κ₁^{2ν} = σ₂²κ₂^{2ν}.
- So only c = σ²κ^{2ν} is consistently estimable under infill asymptotics.
- Kaufman-Shaby 2013 Thm 1: ĉ at a fixed range is consistent and asymptotically N(c, 2c²/n).
- In gamfit's parameterization λ = σ_ε²/σ_f², so σ_f² ∝ e^{-ρ} and c ∝ e^{-ρ}e^{2νψ}.
- The likelihood is therefore asymptotically flat along ρ − 2νψ = const. [P, given Zhang]

**Numerical check (Check E).** ν = ½ with nugget, n = 150, true κ = 0.2.
- ρ*(ψ) − ψ converges: −1.7529, −1.6481, −1.6161, −1.6055, −1.6017, −1.6004, −1.5997, −1.5996, −1.5996 for ψ = 1, 0, −1, …, −10.
- The profile V*(ψ) decreases monotonically to a limit. Successive differences fall by e² over two units of ψ (−3.69e-5, then −5.00e-6), i.e. V* − V*_∞ ∝ κ¹ = κ^{2ν}.
- The finite-difference Hessian at ψ = −4 (test only) has eigenvalues 2.13e-4 and 9.81, and the soft eigenvector is (−0.7068, −0.7074), parallel to the ridge (1, 1)/√2.

**The κ → 0 face.**
- exp(−κh) = 1 − κh + O(κ²h²).
- With a fixed intercept, REML is invariant to adding a multiple of 11ᵀ to the covariance. The limit is the intrinsic model with generalized covariance −c′h, i.e. Brownian motion, which has a finite REML.
- Check E: the intrinsic-limit minimizer c′ = 4.951125 against κ/λ = 4.951150 at ψ = −10.
- So in F3 the lower rails at ρ ≈ −21 are what happens when ψ drifts toward the intrinsic face along the ridge: ψ → −∞ forces ρ → −∞.
- For this data set the optimum is at the face. The "unfinished search" in the `run.rs:6522-6527` comment is in fact a face optimum seen in the wrong chart.

**Expansion variable [R, C].**
- Gu-Wang-Berger 2018 Assumption 3.2 writes R = 11ᵀ + ν(γ)D + ν(γ)ω(γ)(D* + B) as the range γ → ∞.
- Their Table 1 gives the Matérn rates:
  - ν(γ) = γ^{-2α}, ω = γ^{-2+2α} for 0 < α < 1;
  - log terms at α = 1;
  - γ^{-2} and γ^{2−2α} for 1 < α < 2;
  - log terms at α = 2;
  - γ^{-2} and γ^{-2} for α > 2.
- Here α is the Matérn roughness (α = ν). So the first-order face variable is t = κ^{2min(ν,1)}, analytic for ν = ½ and fractional or logarithmic otherwise.
- Lemma 3.3 of Gu-Wang-Berger: the marginal (REML-type) likelihood is O(1) as the range → ∞, i.e. flat. The profile likelihood tends to 0. This matches Check E.
- Conjecture [C]: a first-order face test in (μ = ρ − 2νψ, t) still decides optimality at the face for general ν, but the Hessian in t is unbounded when ν < 1. A certificate at that face must be one-sided and first-order.

**Existing work on reparameterization.**
- Gu-Wang-Berger Thm 3.1 and 4.1: the posterior mode under the reference prior in ξ = log(1/γ) is "robust", i.e. it avoids both the κ → 0 and κ → ∞ degenerate ends. But it changes the estimand, so it is not usable as a default under SPEC.
- Fuglstad et al. 2019 Thm 2.1-2.6: PC priors on (range, σ) are a Weibull-type prior with d/2 shape on κ, i.e. distance ∝ κ^{d/2}. They note the range-variance ridge. The rate is user-chosen, so they are not usable either.
- Kaufman-Shaby 2013 Thm 2 proves consistency of the joint MLE of c only with the range restricted to a known interval [ρ_L, ρ_U], which is a box assumption.

### 3.6 Convergence of iterates on analytic criteria [R, K]

- **Absil-Mahony-Andrews 2005, main theorem (§3; theorem number not re-verified) [K].** For a real-analytic cost, descent iterates satisfying a strong-descent condition and a "primary descent" angle condition either diverge to infinity or converge to a single point.
- **Attouch-Bolte-Redont-Soubeyran 2010 Thm 9 [R] and Attouch-Bolte-Svaiter 2013 Thm 2.9 [K].** For KL functions, bounded sequences with sufficient decrease (H1), relative error (H2) and continuity (H3) have finite length and converge. ABRS Thm 11 gives the rates.
- **Semialgebraic and o-minimal functions** are KL (Kurdyka 1998; Bolte-Daniilidis-Lewis 2007 [K]). REML in τ is analytic on a neighbourhood of the compact domain [0, τ_max].
- **Consequence [P, given the cited theorems].** In the compactified τ chart, with an analytic extension across τ = 0 from Proposition 1, the "diverge" branch of Absil et al. is excluded. Iterates of a projected descent method with sufficient decrease converge to one KKT point.
- In the ρ chart the "diverge" branch is exactly what happens at every face optimum.
- Kurdyka-Orro-Simon 2000 [K]: asymptotic critical values, i.e. values approached along sequences with ‖x‖‖∇f‖ → 0, are finite in number for semialgebraic f. V_∞ at a face is such a value in the ρ chart.

### 3.7 Implicit-differentiation error: a derived ε_g [R]

- **Ablin-Peyré-Moreau 2020.** With inner iterate z and exact inner solution z*, they define estimators g₁ (analytic, ignoring the inner dependence), g₂ (autodiff through the inner iterations) and g₃ (implicit function).
  - Props 1-3: |g₁ − g*| = O(‖z−z*‖), |g₂ − g*| = o(‖z−z*‖), |g₃ − g*| = O(‖z−z*‖²).
- **gamfit's outer gradient** is g₃-type for the penalized-deviance part. That gives an error term quadratic in the inner residual, which is consistent with E_r = ½rᵀH⁻¹r in band_f.
- The LAML ½ log|H(β̂)| term is not a value at a minimum. Its gradient error is linear, about ½‖tr(H⁻¹∂H/∂β)‖·‖β̂ − β*‖.
- **Derived ε_g.** ε_g = ‖∇_β(½ log|H|)‖·‖H⁻¹r‖ + O(‖r‖²). The first factor is the third-derivative contraction already computed for the LAML gradient. The second is the Newton step of the inner problem. No new constants are needed. [P as a first-order bound]
- Grazzi et al. 2020 Thm 2.2 (13): the AID bound Δ̂ ≤ (…)D_λρ_λ(t) + (L_Φ L_E/μ_λ)σ_λ(k), with explicit constants. Useful for proving that the bound holds, not for computing it.
- Pedregosa 2016 Thm 2 (HOAG): if the inner tolerances ε_k are summable, approximate-gradient descent converges to a stationary point. This justifies tightening the inner tolerance with the outer decrement rather than fixing it.

### 3.8 Continuation in λ (F6) [K, C]

- LARS (Efron et al. 2004) and the GLM path algorithm (Park-Hastie 2007) are predictor-corrector continuation in the regularization parameter. The predictor is the tangent from the implicit function theorem dβ̂/dρ = −H⁻¹λSβ̂, and the corrector is Newton.
- Allgower-Georg 2003 give step-length control from the corrector's contraction rate. Beltrán-Leykin 2012 give a certified (α-theory) version for polynomial systems.
- **Use for gamfit [C].** Warm-start each outer trial's inner PIRLS from the tangent predictor β̂ + (dβ̂/dρ)Δρ.
  - Under a Kantorovich-type condition the inner iteration count drops from O(log(1/ε)) Newton steps from a cold start to O(1) steps.
  - This addresses the F6 timeouts if they are dominated by inner solves, which the logs do not show directly.
  - The predictor-corrector is not a fallback or a retry: it is the exact inner solve with a better starting point.

---

## 4. Numerical checks

All checks run with `SP/theory/venv/bin/python`. Finite differences appear only inside these test scripts.

| Check | Claim | Result | Script / output |
|---|---|---|---|
| A | Gaussian (V−V∞)/τ → c; remainder O(τ²) | (V−V∞)/τ = 0.8944769, 0.8938467, 0.8937446, 0.8937340, 0.8937329 for τ = 1e-2…1e-6, against c = 0.8937328; remainder/τ² ≈ 0.118 | `check_tau_chart.out` |
| A′ | dV/dρ = −c e^{-ρ} | −4.058e-5 / −4.058e-5 (ρ=10); −7.432e-7 / −7.432e-7 (ρ=14); −1.343e-8 / −1.361e-8 (ρ=18, FD floor) | same |
| B | binomial LAML c = c_score + c_drift | (V−V∞)/τ = 0.23736914 at τ=1e-5 against c_score+drift = 0.23736929; c_score alone = 0.24006239 (wrong in the 3rd digit) | same |
| C | θ chart: V_θ = 2θc → 0 | 1.790e-1/1.787e-1, 1.788e-2/1.787e-2, 1.787e-3/1.787e-3 | same |
| D | P(c ≥ 0) spectral formula | MC 0.6794 against spectral 0.6806; top μ share 0.992 | same |
| D2 | face test against global minimum | 398/400 agree; 4/2000 bimodal cases, c>0 with an interior minimum near ρ ≈ −5 | `check_d2.py` |
| E | Matérn ν=½ ridge ρ* − 2νψ → const | −1.5996 at ψ = −8, −10; ΔV* ∝ κ; intrinsic c′ = 4.951125 against 4.951150; Hessian eigenvalues 2.1e-4 / 9.81, soft eigenvector along the ridge | `check_matern_ridge.out` |
| F | ρ-Newton on the tail: Δρ → 1, Θ → 1, decrement × e^{-1} | Θ = 0.99932…0.99999; decrement ratio 0.3676 | `check_tail_newton.out` |
| F′ | cancellation of assembled g_ρ against the DR V_τ | relative error 4.4e-12 (ρ=10), 2.6e-7 (20), 8.0e-3 (30), 1.31 (36) | same |

---

## 5. Literature with precise citations

Columns: citation; key result, with equation, theorem or section; the gamfit failure it bears on; whether it is directly usable within SPEC. "Read" gives R (read in full this session) or K (from knowledge). DOIs for [K] entries were not re-verified in this session.

### (a) Smoothing-parameter selection

| # | Citation | Key result | Failure | SPEC-usable | Read |
|---|---|---|---|---|---|
| 1 | Wood, S.N. (2004). Stable and efficient multiple smoothing parameter estimation for generalized additive models. *JASA* 99(467):673-686. | Stable QR/SVD reparameterization for multiple penalties; Newton on log λ for GCV/UBRE | background | partial (the stable algebra yes; GCV no) | K |
| 2 | Wood, S.N. (2008). Fast stable direct fitting and smoothness selection for GAMs. *JRSS-B* 70(3):495-518. doi:10.1111/j.1467-9868.2007.00646.x | §3.4: indefinite Hessian replaced by the Gill-Murray-Wright absolute-eigenvalue modification; coordinates "converged at working infinity" dropped | F1, F4 (the rails) | no: the drop rule is an undefined threshold | R |
| 3 | Wood, S.N. (2011). Fast stable REML and ML estimation of semiparametric GLMs. *JRSS-B* 73(1):3-36. doi:10.1111/j.1467-9868.2010.00749.x | §3, §3.1: Newton on ρ with PD perturbation and step halving; App. B: stable log-determinant for overlapping penalties | background, cancellation | partial (the stable log-determinant yes) | R |
| 4 | Wood, S.N., Pya, N., Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *JASA* 111(516):1548-1563. doi:10.1080/01621459.2016.1180986 | §3 steps 4(b)-(f): drop ρ_i where ∂V/∂ρ_i ≃ ∂²V/∂ρ_i² ≃ 0; §3.1.2: diagonal preconditioning and pivoted Cholesky | F1, F4, F5 | no (the drop rule); yes (preconditioning) | R |
| 5 | Wood, S.N., Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization. *Biometrics* 73(4):1071-1081. doi:10.1111/biom.12666 | Update (3), capped at a "pre-defined upper limit"; Thm 1: the numerator is positive; Thm 3: the update lies between λ and λ̂; §3 (5) neglects ∂H/∂λ | F6 (cheap updates) | no (cap; neglected term) | R |
| 6 | Wood, S.N., Li, Z., Shaddick, G., Augustin, N.H. (2017). Generalized additive models for gigadata. *JASA* 112(519):1199-1210. | Performance-oriented iteration: one smoothing-parameter update per PIRLS step; no certified convergence | F6 | no (uncertified) | K (the PDF obtained was the wrong paper) |
| 7 | Reiss, P.T., Ogden, R.T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *JRSS-B* 71(2):505-523. | GCV is prone to multiple minima and undersmoothing; REML has lower variability and better-behaved derivatives | justifies REML | yes (as justification) | K |
| 8 | Krivobokova, T., Kauermann, G. (2007). A note on penalized spline smoothing with correlated errors. *JASA* 102(480):1328-1337. | REML-based smoothing is robust to moderately misspecified error correlation, unlike GCV/AIC | justifies REML | yes (as justification) | K |
| 9 | Ruppert, D., Wand, M.P., Carroll, R.J. (2003). *Semiparametric Regression*. Cambridge UP. | Ch. 4-5: the mixed-model representation of penalized splines, b ~ N(0, σ_b²I), λ = σ_ε²/σ_b² | Setup (2.1) | yes | K |
| 10 | Kauermann, G., Krivobokova, T., Fahrmeir, L. (2009). Some asymptotic results on generalized penalized spline smoothing. *JRSS-B* 71(2):487-503. | Laplace/LAML accuracy for generalized penalized splines | LAML validity | yes | K |
| 11 | Marra, G., Wood, S.N. (2011). Practical variable selection for GAMs. *CSDA* 55(7):2372-2387. | Extra null-space penalty ("double penalty") so a whole term can be shrunk to zero | "null recoverable" | yes (it is penalize-by-default) | K |

### (b) Mixed models and the boundary

| # | Citation | Key result | Failure | SPEC-usable | Read |
|---|---|---|---|---|---|
| 12 | Bates, D., Mächler, M., Bolker, B., Walker, S. (2015). Fitting linear mixed-effects models using lme4. *J. Stat. Softw.* 67(1):1-48. doi:10.18637/jss.v067.i01 | §3.4 (34), (39)-(41): profiled deviance and REML; §3.5: singular Λ_θ allowed; §4 pp. 23-24: BOBYQA/Nelder-Mead with θ_ii ≥ 0 box; transformed scales give "asymptotically flat" surfaces | F1, F4, F5 (boundary) | no (derivative-free; θ chart degenerate, Prop. 3) | R |
| 13 | Kristensen, K., Nielsen, A., Berg, C.W., Skaug, H., Bell, B.M. (2016). TMB: automatic differentiation and Laplace approximation. *J. Stat. Softw.* 70(5):1-21. doi:10.18637/jss.v070.i05 | (4) Laplace objective; (7) implicit-function gradient assuming an exact inner solution; (8) log-determinant derivative; outer nlminb | ε_g (Section 3.7) | partial (the formulas yes; AD and nlminb no) | R |
| 14 | Brooks, M.E. et al. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated GLMMs. *R Journal* 9(2):378-400. | TMB Laplace + nlminb/optim; convergence warnings at the variance boundary | same as 13 | no | K |
| 15 | Rue, H., Martino, S., Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models (INLA). *JRSS-B* 71(2):319-392. | Mode of π(θ\|y) by quasi-Newton, then grid/CCD exploration of the hyperparameters | none directly | no (grid) | K |
| 16 | Simpson, D., Rue, H., Riebler, A., Martins, T.G., Sørbye, S.H. (2017). Penalising model component complexity: PC priors. *Stat. Sci.* 32(1):1-28. doi:10.1214/16-STS576 | §2.4: base model; Principles 1-4; (3.1) π(ξ) = λe^{-λd(ξ)}; d ∝ σ gives a nonzero slope at σ = 0 | "null recoverable" | partial: its point that the base model must be reachable with a nonzero slope is exactly the τ-chart KKT; the prior itself needs a user rate, so no | R |
| 17 | Fuglstad, G.-A., Simpson, D., Lindgren, F., Rue, H. (2019). Constructing priors that penalize the complexity of Gaussian random fields. *JASA* 114(525):445-452. doi:10.1080/01621459.2017.1415907 | Thm 2.1 (τ\|κ); Thm 2.3: κ has a Weibull prior with shape d/2; Thm 2.5-2.6 joint; range-variance ridge | F3 | no (user rate) | R ([arXiv](https://arxiv.org/abs/1503.00256)) |
| 18 | Chung, Y., Rabe-Hesketh, S., Dorie, V., Gelman, A., Liu, J. (2013). A nondegenerate penalized likelihood estimator for variance parameters in multilevel models. *Psychometrika* 78(4):685-709. | Gamma(2, ·) boundary-avoiding prior on σ keeps the mode off 0 | F1, F4, F5 | no (the null is not recoverable) | K |
| 19 | Self, S.G., Liang, K.-Y. (1987). Asymptotic properties of MLEs and LRTs under nonstandard conditions. *JASA* 82(398):605-610. | Boundary MLE law; ½χ²₀ + ½χ²₁ mixtures | Section 3.2 | yes (test expectations) | K |
| 20 | Stram, D.O., Lee, J.W. (1994). Variance components testing in the longitudinal mixed effects model. *Biometrics* 50(4):1171-1177. | Mixture-χ² LRT at the variance boundary | Section 3.2 | yes | K |
| 21 | Lin, X. (1997). Variance component testing in generalised linear models with random effects. *Biometrika* 84(2):309-326. | Score test for variance components in GLMMs; the statistic is the τ-chart face slope c (Section 3.1) | F1, F4, F5 face KKT | yes (it is c) | K |
| 22 | Crainiceanu, C.M., Ruppert, D. (2004). Likelihood ratio tests in linear mixed models with one variance component. *JRSS-B* 66(1):165-185. doi:10.1111/j.1467-9868.2004.00438.x ([OUP](https://academic.oup.com/jrsssb/article-abstract/66/1/165/7098985), [Wiley](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2004.00438.x)) | Spectral representation of the (R)LRT null law; point mass at λ̂ = ∞ well above ½ | Section 3.2 | yes (test expectations) | K (full text not available) |
| 23 | Crainiceanu, C.M., Ruppert, D., Claeskens, G., Wand, M.P. (2005). Exact likelihood ratio tests for penalised splines. *Biometrika* 92(1):91-103. | P(λ̂ = ∞) for penalized splines under the null is large | Section 3.2 | yes | K |

### (c) Noisy or inexact-oracle optimization, bound constraints and convergence

| # | Citation | Key result | Failure | SPEC-usable | Read |
|---|---|---|---|---|---|
| 24 | Shi, H.-J.M., Xie, Y., Byrd, R., Nocedal, J. (2022). A noise-tolerant quasi-Newton algorithm for unconstrained optimization. *SIAM J. Optim.* 32(1):29-55. doi:10.1137/20M1373190 ([arXiv](https://arxiv.org/abs/2010.04352)) | (2.7)-(2.8) Armijo-Wolfe; (2.9) noise-control lengthening (g(x+βp)−g(x))ᵀp ≥ 2(1+c₃)ε_g‖p‖; Thm 3.5: linear convergence to N₁; N₂ = {φ ≤ 2ε_f + max_{N₁}φ}; (4.6) relaxed Armijo +2ε_f | F1, F4 (BFGS) | yes, with ε_f = band_f | R |
| 25 | Sun, S., Nocedal, J. (2023). A trust region method for noisy unconstrained optimization. *Math. Program.* 202:445-472. doi:10.1007/s10107-023-01941-9 ([arXiv](https://arxiv.org/abs/2201.00973)) | (7) ρ_k = (Δf̃ + rε_f)/(Δm + rε_f); (8) r = 2/(1−c₂); Alg. 1; Lemma 2: the radius stays bounded below; Thm 6: visits C₁ = {‖g‖ ≤ (r+1)ε_g + β/2}; Thm 7: level-set confinement | F4 (reject floor), F2 | yes | R |
| 26 | Berahas, A.S., Cao, L., Scheinberg, K. (2021). Global convergence rate analysis of a generic line search algorithm with noise. *SIAM J. Optim.* 31(2):1489-1518. doi:10.1137/19M1291832 ([arXiv](https://arxiv.org/abs/1910.04055)) | (2.1) f(x+αd) ≤ f(x) − c₁α‖g‖² + 2ε_f; Thm 3.13 complexity to the noise-limited neighbourhood | F1 | yes | R |
| 27 | Xie, Y., Byrd, R., Nocedal, J. (2020). Analysis of the BFGS method with errors. *SIAM J. Optim.* 30(1):182-209. doi:10.1137/19M1240794 ([arXiv](https://arxiv.org/abs/1901.09063)) | BFGS with bounded gradient/function errors converges to a neighbourhood of size O(ε_g); lengthening procedure | F1, F4 | yes | R |
| 28 | Bellavia, S., Gurioli, G., Morini, B., Toint, Ph.L. (2019). Adaptive regularization algorithms with inexact evaluations for nonconvex optimization. *SIAM J. Optim.* 29(4):2881-2915. doi:10.1137/18M1226282 ([arXiv](https://arxiv.org/abs/1811.03831)) | (2.10) inexact optimality measure; (2.12)-(2.13) relative accuracy \|ΔT̄ − ΔT\| ≤ ωΔT̄ | F2, F4 (ARC) | partial: the accuracy requirement becomes a derived stopping rule ΔT < ε_f/ω | R |
| 29 | Cartis, C., Gould, N.I.M., Toint, Ph.L. (2011). Adaptive cubic regularisation methods, Part I. *Math. Program.* 127(2):245-295. | ARC global convergence and complexity with exact evaluations | F2, F4 | yes (base theory) | K |
| 30 | Hager, W.W., Zhang, H. (2005). A new conjugate gradient method with guaranteed descent and an efficient line search. *SIAM J. Optim.* 16(1):170-192. | §4: "approximate Wolfe" conditions σφ′(0) ≤ φ′(α) ≤ (2δ−1)φ′(0), which use only directional derivatives once function differences drop below roundoff | F1 | yes, if the allowance ε\|f\| is replaced by band_f; the derivative test uses ε_g, which can be far below √ε_f | K |
| 31 | Moré, J.J., Wild, S.M. (2011). Estimating computational noise. *SIAM J. Sci. Comput.* 33(3):1292-1314. | ECnoise: a difference-table estimate of ε_f | validating band_f | tests only (it uses FD tables) | K |
| 32 | Bertsekas, D.P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM J. Control Optim.* 20(2):221-246. | Projected Newton with an ε-active set; superlinear convergence; finite active-set identification | all rails | yes (on τ ≥ 0) | K |
| 33 | Calamai, P.H., Moré, J.J. (1987). Projected gradient methods for linearly constrained problems. *Math. Program.* 39(1):93-116. | Finite identification of the active face under nondegeneracy (strict complementarity) | all rails | yes | K |
| 34 | Burke, J.V., Moré, J.J. (1988). On the identification of active constraints. *SIAM J. Numer. Anal.* 25(5):1197-1211. | Identification iff the projected gradient → 0 (under nondegeneracy) | all rails | yes | K |
| 35 | Conn, A.R., Gould, N.I.M., Toint, Ph.L. (1988). Global convergence of a class of trust region algorithms for optimization with simple bounds. *SIAM J. Numer. Anal.* 25(2):433-460. | TR with a generalized Cauchy point on boxes; finite active-set identification | all rails | yes | K |
| 36 | Lin, C.-J., Moré, J.J. (1999). Newton's method for large bound-constrained optimization problems (TRON). *SIAM J. Optim.* 9(4):1100-1127. | Projected-search Cauchy step + TR Newton on free variables; identification + quadratic rate | all rails | yes (recommended template) | K |
| 37 | Absil, P.-A., Mahony, R., Andrews, B. (2005). Convergence of the iterates of descent methods for analytic cost functions. *SIAM J. Optim.* 16(2):531-547. | Main theorem (§3): iterates diverge or converge to a single point | justifies compactification | yes (theory) | K |
| 38 | Attouch, H., Bolte, J., Redont, P., Soubeyran, A. (2010). Proximal alternating minimization and projection methods for nonconvex problems: an approach based on the KL inequality. *Math. Oper. Res.* 35(2):438-457. doi:10.1287/moor.1100.0449 ([arXiv](https://arxiv.org/abs/0801.1780)) | Thm 9 (convergence), Thm 11 (rates) under KL | convergence theory | yes (theory) | R |
| 39 | Attouch, H., Bolte, J., Svaiter, B.F. (2013). Convergence of descent methods for semi-algebraic and tame problems. *Math. Program.* 137:91-129. | Thm 2.9: (H1) sufficient decrease + (H2) relative error + (H3) continuity ⇒ convergence of bounded sequences | the certificate as a KL statement | yes | K |
| 40 | Kurdyka, K. (1998). On gradients of functions definable in o-minimal structures. *Ann. Inst. Fourier* 48(3):769-783. | KL inequality for definable functions | theory | yes | K |
| 41 | Kurdyka, K., Orro, P., Simon, S. (2000). Semialgebraic Sard theorem for generalized critical values. *J. Differential Geom.* 56(1):67-92. | Asymptotic critical values are finite in number | the "rail at infinity" is an asymptotic critical value | theory | K |
| 42 | Deuflhard, P. (2004; 2011 reprint). *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*. Springer SCM 35. | Affine-invariant contraction Θ_k = ‖Δx̄_{k+1}‖/‖Δx_k‖; local convergence needs Θ < 1, with the Θ < ½ monitor for quadratic rate | F2 ("decrement stopped contracting") | yes: Θ is scale-free; Θ → 1 is the tail signature (Section 3.1) | K |
| 43 | Smale, S. (1986). Newton's method estimates from data at one point. In *The Merging of Disciplines*, Springer. Rump, S.M. (2010). Verification methods. *Acta Numerica* 19:287-449. | α-theory: α < 0.157 certifies a zero; Krawczyk/interval Newton | certificate | no: needs rigorous bounds on all higher derivatives of V, unavailable for LAML | K |

### (d) GP/Matérn identifiability

| # | Citation | Key result | Failure | SPEC-usable | Read |
|---|---|---|---|---|---|
| 44 | Zhang, H. (2004). Inconsistent estimation and asymptotically equal interpolations in model-based geostatistics. *JASA* 99(465):250-261. | Thm 2: equivalence iff σ²κ^{2ν} agree (d ≤ 3); only the microergodic parameter is consistent | F3 | yes: gives the chart μ = ρ − 2νψ | K (theorem quoted via 45) |
| 45 | Kaufman, C.G., Shaby, B.A. (2013). The role of the range parameter for estimation and prediction in geostatistics. *Biometrika* 100(2):473-484. doi:10.1093/biomet/ass079 | Thm 1: ĉ at a fixed range is consistent and N(c, 2c²/n); Lemma 1: ĉ(ρ) is monotone in the range; Thm 2: the joint MLE is consistent only with a bounded range interval | F3 | partial (Thm 2 assumes a box) | R |
| 46 | Gu, M., Wang, X., Berger, J.O. (2018). Robust Gaussian stochastic process emulation. *Ann. Statist.* 46(6A):3038-3066. doi:10.1214/17-AOS1648 ([arXiv](https://arxiv.org/abs/1708.04738)) | Assumption 3.2 expansion; Table 1 Matérn rates; Lemma 3.3 marginal likelihood O(1) as range → ∞; Thm 3.1, 4.1 robust posterior mode | F3 (face variable t) | partial: the expansion yes; the reference prior no | R |
| 47 | Stein, M.L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer. | Ch. 4: equivalence of Gaussian measures; Ch. 6: infill asymptotics | F3 | yes (theory) | K |

### (e) Bilevel/hypergradients, continuation, separation

| # | Citation | Key result | Failure | SPEC-usable | Read |
|---|---|---|---|---|---|
| 48 | Ablin, P., Peyré, G., Moreau, T. (2020). Super-efficiency of automatic differentiation for functions defined as a minimum. *ICML*, PMLR 119:32-41 ([arXiv](https://arxiv.org/abs/2002.03722)) | Estimators (3)-(5); Props 1-3: O(e), o(e), O(e²) gradient errors | derived ε_g | yes | R |
| 49 | Grazzi, R., Franceschi, L., Pontil, M., Salzo, S. (2020). On the iteration complexity of hypergradient computation. *ICML*, PMLR 119:3748-3758 ([arXiv](https://arxiv.org/abs/2006.16218)) | Thm 2.1 (ITD), Thm 2.2 (AID bound (13)-(14)), Thm 2.4 (AID-FP) | derived ε_g | partial (constants) | R |
| 50 | Pedregosa, F. (2016). Hyperparameter optimization with approximate gradient. *ICML*, PMLR 48:737-746 ([arXiv](https://arxiv.org/abs/1602.02355)) | Thm 1: gradient error O(ε); Thm 2: summable tolerances ⇒ convergence | inner tolerance schedule | yes | R |
| 51 | Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). Least angle regression. *Ann. Statist.* 32(2):407-499. Park, M.Y., Hastie, T. (2007). L1-regularization path algorithm for GLMs. *JRSS-B* 69(4):659-677. | Exact and predictor-corrector paths in the regularization parameter | F6 | yes (the warm-start predictor) | K |
| 52 | Allgower, E.L., Georg, K. (2003). *Introduction to Numerical Continuation Methods*. SIAM Classics 45. Beltrán, C., Leykin, A. (2012). Certified numerical homotopy tracking. *Exp. Math.* 21(1):69-83. | Step control from corrector contraction; certified tracking | F6 | partial (Beltrán-Leykin needs polynomial structure) | K |
| 53 | Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80(1):27-38. Kosmidis, I., Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in binomial-response GLMs. *Biometrika* 108(1):71-82. Albert, A., Anderson, J.A. (1984). On the existence of maximum likelihood estimates in logistic regression models. *Biometrika* 71(1):1-10. | Jeffreys penalty ½log\|I(β)\| gives finite estimates always (binomial); separation characterization | F5 | partial: a Jeffreys term is a derived, data-free penalty, but it changes the criterion, so it must be a model choice, not a fallback | K |
| 54 | Gill, P.E., Murray, W., Wright, M.H. (1981). *Practical Optimization*. Academic Press. Nocedal, J., Wright, S.J. (2006). *Numerical Optimization*, 2nd ed. Springer. | Modified-Newton, TR, and bound-constrained background | all | yes | K |

That is 54 entries covering more than 60 works. 21 were read in full this session: rows 2-5, 12, 13, 16, 17, 24-28, 38, 45, 46, 48-50, plus Zhang 2004 Thm 2 as restated in Kaufman-Shaby.

**What the literature does not provide.**
- Every production smoothing-parameter or mixed-model optimizer found (mgcv's §3.4/§3 step 4 rules, lme4, glmmTMB/TMB, INLA) either uses a box, a threshold "working infinity" rule, a derivative-free search, or a grid.
- None certifies a face optimum by a first-order KKT test in an analytic chart.
- The closest statistical object is the variance-component score test (Lin 1997), which is c. It has not, as far as I found, been used as an optimizer's face certificate.

---

## 6. Consequences for gamfit

File:line references are against the current tree. `run.rs` means `crates/gam-solve/src/rho_optimizer/run.rs`, and opt means `/root/.cargo/git/checkouts/opt-4a38fa79856f3ac9/53ce029/opt/src/lib.rs`.

### 6.1 Delete (hand boxes and magic constants replaced by derived objects)

| Site | What | Replaced by |
|---|---|---|
| `crates/gam-solve/src/estimate/rho_domain.rs:144` `resolvability_interval`, `:190` `coordinate_domain` | finite ρ box [ln γ_min − 18, ln γ_max + 18]-type | the τ_j ≥ 0 domain (Section 6.2); the resolvability facts become the face-slope evaluation, not a box |
| `crates/gam-solve/src/model_types/result_types.rs:660` `CERTIFICATE_RAIL_MARGIN = 0.5`; `rho_optimizer/bridges.rs:3809` `coordinate_rail_margin` | rail margin | exact face membership τ_j = 0 |
| `run.rs:5823` `ASYMPTOTE_ESTIMAND_REL_TOL = 1e-4`, `:5836` `ASYMPTOTE_PROBE_COUNT = 18`, `:5848-5849` local probe delta/count | tail-probe ladder | c_j from `rail_face_limit` (analytic) |
| `run.rs:6435` `TAIL_SNAP_CURVATURE_BAND = (0.25, 4.0)` and `:6548-6580` tie test | magnitude-only tie band, needed only because of the cancellation in Section 3.1 | τ-chart V_τ, V_ττ in DR form (Prop. 1): no cancellation, sign reliable |
| `run.rs:6522-6534` "psi coordinate, no exponential tail law" decline | treats ψ faces as unfinished searches | Matérn chart (Section 6.3) |
| `crates/gam-terms/src/smooth/term_specs.rs:3770-3772` half-mantissa ψ box; `:3810` `spatial_term_psi_search_box`; `:2789`/`:2806` data-derived bounds | hand box by SPEC | (μ, t) chart with the domain t ∈ [0, ∞) |
| opt `lib.rs:134` `armijo_roundoff_cushion = 8ε(1+\|f\|)` as the only noise allowance | noise-blind Armijo | Shi et al. (4.6) with ε_f = band_f |
| opt `lib.rs:782` `noise_aware(1e-12, 1e6, 1e-14)` | magic relative floor | Sun-Nocedal ratio with ε_f = band_f |

### 6.2 Build: native face coordinates with an analytic KKT test (F1, F3, F4, F5)

**Chart.**
- τ_j = e^{-ρ_j} for coordinates whose optimum is at or near the upper face.
- Alternatively use the edf-fraction chart of `compactification.md`, which is analytic at both ends.
- λ_j itself serves at the covered lower face (Section 3.3).
- The domain τ_j ≥ 0 is the mathematics, not a hand bound.

**Algorithm (TRON template, Lin-Moré 1999).**
1. Projected-search generalized Cauchy point on {τ ≥ 0}.
2. Trust-region Newton on the free variables.
3. Sun-Nocedal ratio (7) with ε_f = band_f and r = 2/(1−c₂).

**Gradient and Hessian at small τ.**
- Use V_τ and V_ττ in cancellation-free form: the DR form of Prop. 1 (Gaussian), or `rail_face_limit`'s g_c and C plus the drift (LAML).
- Do not use −λg_ρ: its relative error is ε·r·λ/c (Check F′).

**Face certificate.**
- At τ_j = 0, the KKT test is c_j ≥ δ_j with δ_j = the derived band on c_j.
- That band is the error of c_j from `rail_face_limit`, of the same form as the trace-channel part of band_f, with no λ amplification.
- Strict complementarity c_j > δ_j gives finite identification (Calamai-Moré, Burke-Moré).
- If |c_j| ≤ δ_j the face is degenerate, and the certificate must include the second-order term V_ττ(0) = 2a₂ from `boundary-asymptotics.md`. Because V is analytic in τ this is decidable.

**Expected effect.**
- F1: #2 railed at 22.73 becomes τ₂ = 0 with c₂ checked.
- F4: the Weibull tail g ∝ e^{-ρ} becomes one projected step.
- F5: the lower rails are handled by the λ_j-chart face or flagged as separation.
- F3: see Section 6.3.

### 6.3 Build: Matérn chart (F3)

- Replace (ρ, ψ) by (μ = ρ − 2νψ, ψ). The criterion is flat in ψ at fixed μ to O(κ^{2min(ν,1)}).
- Treat κ → 0 as the intrinsic face t = κ^{2min(ν,1)} = 0, with the limit criterion computed from the intrinsic (generalized-covariance) model. The REML invariance to 11ᵀ with a fixed intercept makes it finite.
- The face KKT condition is ∂V/∂t|₀ ≥ δ.
- Status: [N] for ν = ½; [C] for general ν because of the fractional powers and logs in Gu-Wang-Berger Table 1.
- Delete the ψ box at `term_specs.rs:3770-3772`.
- The κ → ∞ end (range → 0, a pure nugget) is the τ-face of the spatial variance and is handled by Section 6.2.

### 6.4 Build: noise-tolerant acceptance and a noise-floor certificate (F1, F2, F4)

1. **ARC ratio at opt `lib.rs:7781`.**
   - Replace `rho = (f_k - f_trial) / denom` by `rho = (f_k - f_trial + r*eps_f) / (denom + r*eps_f)` with r = 2/(1−η₂), where η₂ is ARC's "very successful" threshold.
   - Pass eps_f = band_f from `outer_decrement_bands` (`decrement_bands.rs:67`).
   - `crates/gam-custom-family/src/joint_newton.rs:2162` already passes a derived noise floor to `noise_aware`. Make that the ε_f path everywhere.
2. **TrustRegionPolicy::update (opt `lib.rs:871-923`).** Same ratio. Remove the 1e-14 default.
3. **Armijo (opt `lib.rs:8481` `accept_armijo`).** Use f̃(x+αp) ≤ f̃(x) + c₁αg̃ᵀp + 2ε_f (Shi et al. (4.6)).
   - When 2ε_f dominates, switch to the Hager-Zhang approximate-Wolfe test, which uses g̃ᵀp and so ε_g, not ε_f.
4. **Noise-floor certificate.** On StepSizeTooSmall or MaxAttempts:
   - evaluate the analytic outer Hessian (for F1, curvature was "unavailable");
   - compute ½λ̂² = ½gᵀH⁻¹g on the free face;
   - if ½λ̂² ≤ 2·band_f, certify at rung NewtonDecrement, because Prop. 5 makes no further certified progress possible;
   - otherwise report a genuine failure.
   - Align `decrement_stationarity_bound` (`decrement_bands.rs:288`) to the same 2·band_f threshold, or justify band_f as the per-evaluation error and 2·band_f as the difference error, so no gap window exists.
5. **F2 (x1+cyclic(x2)).** ½λ̂² = 4.7e-5 is far above band_f, so F2 is not noise-limited. The mechanism predicted by Section 3.1 is a coordinate on an exponential tail: steps of about 1 e-fold, decrement × e^{-1} per step, and about ln(4.7e-5/3.95e-11) ≈ 14 more steps to reach band_f. [C] until the log's per-coordinate steps are checked. The fix is the τ chart of Section 6.2. Deuflhard's Θ_k ≈ 1 with ‖Δρ_k‖ ≈ 1 is the diagnostic that identifies it, and it is scale-free.

### 6.5 Derived tolerances (no magic constants)

| Quantity | Formula | Source |
|---|---|---|
| ε_f | band_f = B_channels + B_factor + \|½rᵀH⁻¹r\| | `decrement_bands.rs:67` (existing) |
| ε_g | ‖∇_β ½log\|H\|‖·‖H⁻¹r‖ + (quadratic g₃ part, O(‖r‖²)) | Ablin et al. Props 1, 3 (Section 3.7) |
| Armijo allowance | 2ε_f | Shi et al. (4.6); Berahas et al. (2.1) |
| TR ratio shift | rε_f, r = 2/(1−c₂) | Sun-Nocedal (7)-(8) |
| Noise-floor stationarity | ½gᵀH⁻¹g ≤ 2ε_f, i.e. ‖g‖ ≤ 2√(Mε_f) on steepest descent | Prop. 5; Shi et al. Thm 3.5 N₁ |
| Face multiplier band δ_j | error bound of c_j from its trace channels, without λ amplification | Prop. 1/2; `rail_face_limit.rs:56` |
| Inner tolerance | summable sequence tied to the outer decrement, e.g. E_r ≤ ½λ̂²_outer / k² | Pedregosa Thm 2 |

The classical algorithm constants c₁ (Armijo) and c₀ ≤ c₁ < c₂ (TR) are not tolerances. The theory holds for any values in their open intervals, and they already exist in opt.

### 6.6 Certificate content

A fit is certified when all of the following hold:
1. Every free coordinate satisfies ½gᵀH⁻¹g ≤ 2ε_f on the free face, with H positive definite there.
2. Every face coordinate satisfies c_j ≥ δ_j. In the degenerate case |c_j| ≤ δ_j it satisfies the second-order face test.
3. The Matérn intrinsic face, if active, satisfies ∂V/∂t ≥ δ_t.

The certificate should also record:
- the face set;
- the c_j values;
- ε_f and ε_g;
- the Deuflhard Θ of the last two steps, which should be < 1.

---

## 7. Open problems

1. **Global versus local face optimality.** Check D2 found 4/2000 null replicates with c > 0 at the face and a lower interior minimum near ρ ≈ −5, a bimodality margin up to 1.57.
   - A local KKT certificate cannot exclude this.
   - For one penalty, V_τ in Prop. 1 is a rational function of τ, so its real roots on [0, ∞) can be isolated exactly (Sturm sequences on the numerator polynomial of degree about 2r). That is a candidate global certificate that is not a grid [C].
   - For many penalties this is open.
2. **Non-analytic Matérn faces.** For ν ∉ {½, 3/2, …} the κ → 0 expansion has fractional powers, and at integer ν it has logs (Gu-Wang-Berger Table 1). Whether a first-order face test suffices in t = κ^{2min(ν,1)} is [C].
3. **Overlapping-penalty corners** (tensor products). The first-order term is 1-homogeneous and non-analytic in product charts (see `compactification.md`). No published bound-constrained method covers such corners.
4. **LAML with separation (F5).** Whether the LAML criterion has a finite limit as λ → 0 under separation, and whether a Jeffreys term (Kosmidis-Firth 2021) should be a model default, is unresolved. The literature treats it only for the MLE.
5. **ε_f for LAML log-determinants.** band_f's channels need a proof that they bound the evaluator error, including the pivoted-Cholesky and inner-residual interaction. See `fp-error-analysis.md`; that proof is not in the literature.
6. **ARC adaptation of the Sun-Nocedal ratio.** It is formally identical to the TR case, but I found no published proof for cubic regularization with a fixed noise floor. Bellavia et al. 2019 need relative accuracy that shrinks with the model decrease.
7. **F6 timeouts.** Whether they are inner-solve dominated, in which case tangent warm-starts (Section 3.8) help, is not visible from the log.

---

## Ranked top 10 ideas for gamfit

1. **τ-chart (or edf-fraction) face coordinates with the analytic KKT test c_j ≥ δ_j**, where c_j is the variance-component score. This deletes the ρ boxes, rail margin, tail-snap probes and tie band. Addresses F1, F4, F5 and, with idea 4, F3. Sources: Lin 1997; Crainiceanu-Ruppert 2004; Props 1-3.
2. **Sun-Nocedal noise-tolerant ratio (7)-(8) with ε_f = band_f** in ARC (opt `lib.rs:7781`) and TrustRegionPolicy (`:782`, `:871-923`). Addresses F4 (reject floor) and F2.
3. **Shi et al. relaxed Armijo (+2ε_f), plus a noise-floor certificate** ½gᵀH⁻¹g ≤ 2band_f on line-search failure, with the decrement and line-search thresholds aligned. Addresses F1, F4 (BFGS).
4. **Matérn microergodic chart (μ = ρ − 2νψ, ψ) with the intrinsic κ → 0 face**, which deletes the ψ box. Addresses F3. Sources: Zhang 2004; Kaufman-Shaby 2013; Gu-Wang-Berger 2018.
5. **Projected-Newton/TRON on τ ≥ 0 with finite active-face identification.** Addresses all rails. Sources: Bertsekas 1982; Calamai-Moré 1987; Burke-Moré 1988; Lin-Moré 1999.
6. **Cancellation-free V_τ and V_ττ** (the DR form, or `rail_face_limit`) instead of −λg_ρ. This removes the #2298 sign-flip class. Addresses F1, F4, F2.
7. **Derived ε_g from implicit-differentiation error orders, and summable inner tolerances.** Addresses all. Sources: Ablin et al. 2020; Pedregosa 2016; Grazzi et al. 2020.
8. **Deuflhard contraction Θ as a scale-free tail diagnostic** (Θ ≈ 1 with unit steps means a face in the wrong chart). Addresses F2, F5.
9. **Boundary-probability expectations in tests:** P(face) = P(Σμ_s w_s² ≤ Σμ_s) ≈ 0.68 on null smooths. Keeps tests from demanding interior optima. Addresses the test design behind F1 and F4.
10. **Tangent-predictor warm starts for inner solves** (continuation in ρ). Addresses F6 [C]. Sources: Park-Hastie 2007; Allgower-Georg 2003.
