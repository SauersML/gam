# Compactifying smoothing-parameter space: faces instead of rails

Slug: `compactification`. Scope: this report finds a closed, compact coordinate system for the smoothing parameters λ ∈ [0, ∞]^m. In it, λ_j = ∞ and λ_j = 0 are ordinary boundary faces of a smooth objective. The consequence is that the hand ρ-box, the rails, the tail-law asymptote certificate and the tail snap can all be replaced by textbook bound-constrained optimality theory: KKT conditions, projected Newton or trust region, and finite active-set identification.

Status labels used throughout:
- **[proven]**: a complete proof is given here, or a standard result is cited exactly.
- **[sketch]**: the proof idea is complete but some routine steps are left out.
- **[conj]**: a conjecture.
- **[num]**: checked numerically (Section 4). Scripts are in `SP/theory/compactification/`, where `SP` is the session scratchpad, run with `SP/theory/venv`.

---

## 1. Summary

- **One master lemma covers every boundary question** [proven for Gaussian REML; for LAML under assumption A2].
  - Split coefficients into Q = range(Σ_j S_j) and N = the common null space. Substitute ã = S_Q(λ) β_Q.
  - REML and LAML then become V = F(M), with M = S_Q(λ)⁻¹. F is **real-analytic on a neighbourhood of the whole closed cone M ⪰ 0**, including M = 0.
  - So the smoothness of V at any boundary equals the smoothness of the chosen chart ↦ M. It is never a property of V itself.
- **Upper face λ_j = ∞** [proven, num].
  - In τ_j = 1/λ_j, M is affine in τ_j, so V is analytic across τ_j = 0, and even extends to τ_j < 0.
  - The face slope c_j = ∂V/∂τ_j at 0 has a closed form: c = ½tr(Λ⁻¹Sch) − ½g_RᵀΛ⁻¹g_R, plus a dW term for LAML. Chebyshev differentiation agrees with it to a relative 1e-12 to 1e-15 for Gaussian fixed φ, Gaussian profiled φ and logistic LAML.
  - Under null-space truth, the face is the optimum (c > 0) in 68% of replicates, so faces are the common case, not an edge case.
- **Candidate charts compared.**
  - θ = √τ (lme4) makes V even in θ. Then V_θ(0) ≡ 0, so the face multiplier is identically zero and strict complementarity fails everywhere. Also, λ = 0 stays at θ = ∞.
  - σ² = φτ is equivalent to τ.
  - 1/(1+λ) is analytic at both ends but uses an arbitrary unit scale.
  - ρ = log λ sends both faces to infinity. In ρ, Newton with the exact Hessian provably advances exactly one e-fold per iteration toward a face and never reaches it: 13 to 29 iterations, stalling at a fake point ρ ≈ 33 to 37 [num].
- **Recommended chart: the edf fraction u_j = (1/r_j) Σ_i γ_ji/(γ_ji + λ_j) ∈ [0, 1]** [proven].
  - It is an analytic diffeomorphism of [0, ∞] onto [0, 1] whose inverse is analytic at both ends. In the Gaussian case with γ at the fit, u_j = (edf_j − M_p)/r_j.
  - γ_j comes from the existing `penalty_range_gammas_with_shared_nullspace` (`crates/gam-solve/src/estimate/rho_domain.rs:47`), so the chart has no tuning constant.
  - [0, 1] is the mathematical domain of the parameter, not a hand bound.
  - The face multiplier is O(1): g_u(0) = (r/2)(1 − g_RᵀΛ⁻¹g_R / Σγ) ≤ r/2. It was measured at 4.83 and 4.81 against a bound of 5.
- **Lower face λ_j = 0 has three cases** [proven, num].
  - Covered (range(S_j) ⊆ Σ_{k≠j} range(S_k)): V is analytic at u_j = 1.
  - Uncovered and identified: a barrier V ≈ −(r′/2) log(1 − u_j) → +∞. The minimiser is interior.
  - Exact fit with profiled φ: V → −∞. This must be a typed "REML unbounded" error, not a fit.
- **Corners of overlapping penalties (tensor products) are genuinely non-smooth in every product chart** [proven, num].
  - The first-order term is h(τ) = ½tr(M(τ)C), where M(τ) is an Anderson–Duffin parallel sum. h is positively 1-homogeneous and nonlinear.
  - For two penalties, V is analytic in the blow-up (r, s) = (τ₁ + τ₂, τ₁/(τ₁ + τ₂)) [sketch, num].
  - The corner KKT condition is min_{s ∈ [0,1]} h(s) > δ. h is rational in s, so this is decided by polynomial root isolation, not a grid.
  - Faces of non-overlapping penalties (different smooths) are smooth corners, and ordinary bound-constrained theory applies.
- **Optimiser.**
  - A bound-constrained trust region in the Lin–Moré (TRON) style on u ∈ [0, 1]^m. The other hyperparameters (ψ) keep their natural domains.
  - Steps: generalised Cauchy point by projected search, subspace Newton with the exact Hessian (including negative curvature), and exact placement on faces.
  - Certificate, face coordinates: an outward multiplier greater than its derived forward-error bound δ_j. That bound does not grow with λ; in ρ the SNR falls like e^{−2ρ}.
  - Certificate, free coordinates: H_FF ≻ 0 and the affine-invariant Newton decrement ½g_FᵀH_FF⁻¹g_F ≤ δV.
  - With strict complementarity, the active set is identified in finitely many steps and convergence is quadratic (Burke–Moré 1988; Lin–Moré 1999).
- **What the current failures are.** The box is [ln(√ε γ_min), ln(γ_max/√ε)] (`rho_domain.rs:142`–`159`).
  - Its edges sit exactly at the ρ where the naive ρ-gradient's rounding noise equals its signal c·e^{−ρ}. Measured crossover: ρ ≈ 22 [num].
  - An optimum on a face therefore rails at the box edge, with |Pg| of noise size just above the bound. This matches prostate-logit (2.28e-5 against 7.3e-6 at ρ = 22.73), the Matérn and statsmodels lower rails near −21 to −18, and the statsmodels upper rail at 21.63.
  - The Weibull-AFT and `x1+cc(x2)` failures are interior or indefinite, not face problems. They gain only from the exact-Hessian negative-curvature trust region and from deleting the tail snap.
- **Deletions.**
  - Delete entirely: `rho_optimizer/rail.rs` (230 lines), `rho_optimizer/asymptote_certificate.rs` (725 lines, including the magic 1e-8, 1e-6 and 1e-3 constants), the resolvability-box constructors, and the inverted-box validation in `run.rs`.
  - Keep and repurpose `rail_face.rs` and `reml/rail_face_limit.rs` as the τ-regular face evaluator. Ideally it merges into the main evaluator: the master-lemma form is valid at τ = 0 and at τ > 0 alike.
  - The opt crate needs: a generalised Cauchy point and projected search, typed face and degenerate certificates with caller-supplied error bounds replacing the sign-only `kkt_projected_gradient` (opt `lib.rs:3141`), negative-curvature trust-region steps, and removal of the literal 1e-8 and 1e-16 tolerances.

---

## 2. Setup and notation

**Criterion.** Coefficients β ∈ ℝ^p. Penalty S_λ = Σ_{j=1}^m λ_j S_j, with S_j ⪰ 0 and λ_j ≥ 0.
- Gaussian, fixed φ, with φ absorbed into (X, y):
  V = ½D_p + ½log|H| − ½log|S_λ|₊, where D_p = ‖y − Xβ̂‖² + β̂ᵀS_λβ̂ and H = XᵀX + S_λ.
- Profiled φ: V = ((n − M_p)/2) log D_p + ½log|H| − ½log|S_λ|₊.
- LAML: V = −ℓ(β̂) + ½β̂ᵀS_λβ̂ + ½log|H(W)| − ½log|S_λ|₊, where H(W) = XᵀW(η̂)X + S_λ.

Additive constants are dropped throughout.

**Splitting.** Let Q = range(Σ_j S_j) and N = Q^⊥ = ∩_j null(S_j), with M_p = dim N. Write β = U_Q β_Q + U_N b and A = XᵀWX in (Q, N) blocks. Then:
- the Schur complement is Sch = A_QQ − A_QN A_NN⁻¹ A_NQ;
- S_Q(λ) = U_Qᵀ S_λ U_Q, which is ≻ 0 on Q whenever every λ_j > 0;
- M(λ) = S_Q(λ)⁻¹.

For a single coordinate j, write S_j = U_R Λ U_Rᵀ, with Λ ≻ 0 (r_j × r_j) and the null space U_Z. Then:
- γ_j = eig(Λ^{−1/2} Sch_j Λ^{−1/2}), the generalised eigenvalues of the Schur complement against the penalty on range(S_j);
- g_R = X_Rᵀ(y − μ(X_Z b₀)) is the score of the penalised directions at the restricted fit b₀.

**Charts** for a single coordinate:
- ρ = log λ;
- τ = 1/λ;
- θ = √τ;
- σ² = φτ;
- τ_b = 1/(1 + λ);
- u = (1/r) Σ_i γ_i/(γ_i + λ).

**Assumptions.**
- (A1) A_NN(W) ≻ 0 at the fits considered. This means the unpenalised part is identified.
- (A2) The inner problem has a unique minimiser that depends analytically on the data of the system: −ℓ is real-analytic in η with a positive-definite Hessian, and the N-restricted fit b₀ exists, i.e. there is no separation in the unpenalised span.

  For Gaussian models, A2 follows from A1. For a C^k likelihood, replace "analytic" by C^{k−2} everywhere.
- (A3) The "identified" condition at λ_j = 0: XᵀWX + Σ_{k≠j} λ_k S_k ≻ 0 on the uncovered part of range(S_j). This is the property `unpenalized_fit_is_identified` (`rho_domain.rs:171`) tests.

Note on naming: the code's existing "τ-coordinates" (for example `reml/hyper.rs:1568`) refer to design-moving hyperparameters, not to 1/λ. The chart recommended here is called u to avoid a clash.

---

## 3. Results with proofs

### Lemma 0 (master lemma: V is an analytic function of M on the closed PSD cone) [proven for Gaussian; LAML under A2]

Substitute ã = S_Q(λ) β_Q, so that β_Q = Mã. The Gaussian normal equations become the **regular block system**

```
[ A_QQ M + I    A_QN ] [ ã ]   [ X_Qᵀ y ]
[ A_NQ M        A_NN ] [ b ] = [ X_Nᵀ y ]
```

with D_p = ‖y − X_Q Mã − X_N b‖² + ãᵀMã and

log|H| − log|S_λ|₊ = log|A_NN| + log|I + M Sch|.

**Claim.** Every entry is a polynomial in M. The system is nonsingular for every M ⪰ 0, including M = 0. So ã, b, D_p and the log-determinant pair are real-analytic in M on an open neighbourhood of {M ⪰ 0}.

**Proof.**
1. Eliminating b leaves (I + Sch M)ã = rhs.
2. The eigenvalues of Sch M are those of M^{1/2} Sch M^{1/2}, which are ≥ 0. Hence I + Sch M is invertible and A_NN ≻ 0 by (A1). Nonsingularity is an open condition, so it persists on a neighbourhood of the cone.
3. For the determinant, when M ≻ 0: |H| = |A_NN| · |S_Q + Sch|, and |S_λ|₊ = |S_Q| because S_λ vanishes on N and is positive definite on Q. Then log|S_Q + Sch| − log|S_Q| = log|I + M Sch|. The right-hand side extends analytically to all M ⪰ 0.
4. LAML: the inner stationarity in (ã, b) is

   ã − X_Qᵀ(y − μ(η)) = 0,  X_Nᵀ(y − μ(η)) = 0,  with η = X_Q Mã + X_N b.

   (The first equation is S_Q β_Q = X_Qᵀ(y − μ) rewritten.) At M = 0 its Jacobian is [[I, X_QᵀWX_N], [0, A_NN(W)]], which is nonsingular. For M ⪰ 0 it is similar to the positive-definite inner Hessian. The implicit function theorem (analytic version) gives analytic (ã, b)(M). W = W(η) is analytic, so log|A_NN(W)| + log|I + M Sch(W)| is analytic.
5. The penalised deviance term −ℓ + ½ãᵀMã is analytic by composition. ∎

**Consequence.** For any chart x ↦ λ(x), the smoothness of V at a boundary point is the smoothness of x ↦ M(λ(x)) there. The questions in the brief reduce to properties of the map from a chart to M.

### Theorem 1 (upper face λ_j = ∞ is an analytic face in τ_j) [proven, num]

Fix the other λ_k ∈ (0, ∞), possibly with penalties overlapping S_j. Then V is real-analytic in τ_j = 1/λ_j on a neighbourhood of τ_j = 0, including τ_j < 0.

**Proof.**
1. On Q, S_Q = λ_j S_jQ + R_Q, where R = Σ_{k≠j} λ_k S_k.
2. Work in the basis [range(S_j), rest of Q]. The generalised Schur complement of R onto the complement of range(S_j) has constant rank for λ_j ∈ (0, ∞]: the PSD Schur complement of λ_jΛ + R_RR is ≻ 0 for λ_j > 0, so its rank is r_j plus the rank of R's restriction.
3. Block inversion then gives M = S_Q⁻¹ as a rational function of τ_j with no pole at τ_j = 0. Its τ_j → 0 limit pins range(S_j), and the leading term is τ_j Λ⁻¹ on that block (Kato's analytic perturbation theory for the constant-rank pseudo-inverse; Kato 1995, ch. II).
4. Compose with Lemma 0. ∎

In the single-penalty case this reduces to the system actually implemented in the checks: [[τA_RR + Λ, A_RZ], [τA_ZR, A_ZZ]][ã; b] = [X_Rᵀy; X_Zᵀy], with log-determinant pair log|A_ZZ| + log|Λ + τSch| − log|Λ|. The radius of analyticity in τ is at least 1/γ_max.

**Face slope (closed form)** [proven, num].
- Envelope theorem: dD_p/dτ = −ãᵀΛã. At τ = 0, ã₀ = Λ⁻¹g_R.
- Gaussian, fixed φ:
  c = V_τ(0) = ½tr(Λ⁻¹Sch) − ½g_RᵀΛ⁻¹g_R = ½tr(Λ⁻¹C), with C = Sch − g_R g_Rᵀ.
- Gaussian, profiled φ:
  c = ½tr(Λ⁻¹Sch) − ((n − M_p)/2) · g_RᵀΛ⁻¹g_R / D_p0.
- LAML:
  c = ½tr(Λ⁻¹Sch(W₀)) − ½g_RᵀΛ⁻¹g_R + ½tr(A_ZZ⁻¹ X_Zᵀ diag(w′ ⊙ η′) X_Z),
  where η′ = (I − X_Z A_ZZ⁻¹ X_ZᵀW) X_R ã₀ and w′ = dw/dη. For the logit link, w′ = μ(1 − μ)(1 − 2μ).
- Tail law in ρ: g_ρ = −τV_τ = −c e^{−ρ} + O(e^{−2ρ}).

c is the REML score statistic for the variance component σ² = φτ at its boundary. It is positive, so the face is a KKT point, exactly when the observed quadratic form g_RᵀΛ⁻¹g_R falls below its null expectation tr(Λ⁻¹Sch) (fixed φ).

**Second derivative (exact, for the trust-region model)** [proven, num]. Let K = (Λ + τSch)⁻¹Sch. Then:
- V_τ = ½D_pτ + ½tr K;
- V_ττ = ½D_pττ − ½tr(K²);
- D_pτ = −ãᵀΛã and D_pττ = −2ãᵀΛã_τ, where ã_τ comes from differentiating the block system.

### Theorem 2 (the other candidate charts) [proven]

**(a) θ = √τ (lme4 relative-factor chart).** V(θ) = F(θ²) with F analytic, so V is even and analytic in θ. Hence V_θ(0) = 0 and V_θθ(0) = 2c.
- On θ ≥ 0 the Lagrange multiplier of the face is **identically 0**. Every face point satisfies first-order KKT regardless of the sign of c, and strict complementarity fails.
- Optimality is second-order only: θ = 0 is a strict local minimum iff c > 0; if c < 0 it is a local maximum along θ.
- Finite active-set identification theorems (Burke–Moré 1988; Calamai–Moré 1987) do not apply. The certificate becomes a curvature sign test on the same quantity c. So θ gives nothing that τ does not, and it loses the first-order face certificate.
- λ = 0 sits at θ = ∞, so the θ chart is not compact.
- lme4's use of θ (Bates et al. 2015, §3) is motivated by the Cholesky-factor parameterisation of the random-effects covariance, not by boundary regularity.

**(b) σ² = φτ.**
- With fixed φ, this is a linear rescaling of τ.
- With profiled φ, φ̂(τ) = D_p(τ)/(n − M_p) is analytic and positive at τ = 0 (D_p0 > 0 unless the fit is exact), so dσ²/dτ(0) = φ̂₀ > 0.

Either way it is an analytic reparameterisation of τ near the face, with the same face and the same sign of multiplier. The λ → 0 end is σ² → ∞, which is not compact.

**(c) τ_b = 1/(1 + λ).**
- At λ = ∞, τ_b = τ/(1 + τ), analytic with derivative 1.
- At λ = 0, 1 − τ_b = λ/(1 + λ), analytic with derivative 1.

So τ_b is an analytic compactification [0, ∞] → [0, 1]. Its defect is the fixed reference scale λ = 1, which is arbitrary in the units of S_j. For γ ≫ 1 the interesting λ sit at τ_b ≈ 1 − 1/γ, and resolution near 1 is lost: for λ below about ε, 1 − τ_b rounds to 0. The u chart below is τ_b with the scale derived from the data, averaged over the r_j directions. For r_j = 1 it is exactly 1/(1 + λ/γ).

**(d) ρ = log λ.** Both faces are at infinity. Near the upper face, g_ρ = −cτ + O(τ²) and H_ρρ = cτ + O(τ²). The exact Newton step is therefore Δρ = −g_ρ/H_ρρ = 1 + O(τ): **exactly one e-fold per iteration**, never reaching the face.

The decrement ½g²/H ≈ ½cτ falls below the value resolution at ρ ≈ ln(c/(8ε|V|)) + const ≈ 33 to 37 for the check-2 data. That is a "converged" point with an arbitrary λ. This is the mechanism behind rails and "gradient just over bound" (Section 6.1).

### Theorem 3 (the lower end λ_j → 0: three cases) [proven for Gaussian; LAML under A2, A3; num]

Hold the other λ_k fixed and let r′ = rank(S_λ) − rank(Σ_{k≠j} λ_k S_k), the dimension of range(S_j) not covered by the other penalties.

- **(i) Covered (r′ = 0).** S_Q stays ≻ 0 at λ_j = 0, so M is analytic in λ_j at 0 and V is analytic in λ_j through 0 (Lemma 0). The tail law is g_ρ = +c′e^{ρ}, with c′ = ∂V/∂λ_j(0).
- **(ii) Uncovered and identified (r′ > 0, A3).**
  - |S_Q(λ)| = λ_j^{r′} · (analytic, nonzero), while |S_Q + Sch| stays analytic and nonzero by A3.
  - Hence ½log|H| − ½log|S_λ|₊ = −(r′/2) log λ_j + analytic, and V → +∞ as λ_j → 0. So g_ρ → −r′/2, and V is coercive at this end.
  - In u: V = −(r′/2) log(1 − u_j) + analytic, a barrier. The face u_j = 1 is never optimal.
- **(iii) Exact fit with profiled φ** (y ∈ col(X), n > M_p + r′).
  - D_p(λ_j) = κλ_j + O(λ_j²) → 0, so V ≈ ((n − M_p − r′)/2) log λ_j → −∞.
  - REML is unbounded below and there is no REML estimate. This must surface as a typed non-identifiability error, not as a fit.
- **(iv) Not identified** (A3 fails; for example a basis wider than the data support). log|H| also loses rank as λ_j → 0 [sketch]. The log λ_j coefficient becomes (d − r′)/2, with d = dim null(XᵀWX + R) ≤ r′. If d = r′ the end is analytic, like case (i).

Logistic separation inside the covered block is the analogue of (iii) for LAML. It is not covered by A2 (see Section 7).

### Theorem 4 (the edf chart u) [proven]

For a coordinate with positive γ₁, …, γ_r (any positive vector gives a valid chart), define

u(λ) = (1/r) Σ_i γ_i/(γ_i + λ),  u(0) = 1,  u(∞) = 0.

**Claims.**
1. u is a strictly decreasing homeomorphism [0, ∞] → [0, 1], analytic on (0, ∞).
2. In the local chart τ at ∞: u = (1/r) Σ γ_iτ/(1 + γ_iτ), analytic at τ = 0 with u′(0) = Σγ/r > 0.
3. In the local chart λ at 0: 1 − u = (1/r) Σ λ/(γ_i + λ), analytic at λ = 0 with derivative Σγ⁻¹/r > 0.
4. By the analytic inverse function theorem, τ(u) and λ(1 − u) are analytic at the ends. So u is a global analytic coordinate on the compactification [0, ∞] with the smooth structure given by τ at ∞ and λ at 0, and it is the unique structure in which both faces are regular.
5. By Theorems 1 and 3, V is analytic at u = 0, analytic at u = 1 in the covered case, and a −(r′/2) log(1 − u) barrier at u = 1 in the uncovered case.

**Interpretation.** For a single-penalty Gaussian model with γ the Schur eigenvalues at the fit, Σ_i γ_i/(γ_i + λ) is the effective degrees of freedom of range(S_j). So u = (edf − M_p)/r is the fraction of the penalised span that is used. Indexing smoothness by degrees of freedom is classical (Hastie & Tibshirani 1990, ch. 3). Using it as a compactifying optimisation chart with certified faces is the new part here.

**Face multiplier (well scaled).**
g_u(0) = c/u′(0) = c r/Σγ.
With fixed φ, c = ½Σγ − ½q, where q = g_RᵀΛ⁻¹g_R, so

g_u(0) = (r/2)(1 − q/Σγ) ≤ r/2  (profiled φ: replace q by (n − M_p)q/D_p0).

The multiplier is a normalised score: dimensionless, bounded above by r/2, and independent of the units of S_j and of λ. Measured values were 4.834 and 4.813 against r/2 = 5 [num].

**Chart construction without constants.** Take γ from `penalty_range_gammas_with_shared_nullspace` (`crates/gam-solve/src/estimate/rho_domain.rs:47`) at the initial W, and freeze it for the run. Any positive γ gives a valid chart; γ only preconditions. So freezing γ is not an approximation and does not change the optimum.

**Inversion.** λ(u) solves f(λ) = Σ γ_i/(γ_i + λ) = r u.
- f is convex and decreasing in λ, so Newton started from λ = 0 increases monotonically to the root.
- In τ, h(τ) = Σ γ_iτ/(1 + γ_iτ) is concave and increasing, so Newton started from τ = 0 is also monotone.
- Use the τ iteration for u ≤ ½ and the λ iteration for u > ½. Either converges from its start; the symmetric split affects speed only, not correctness, so it is not a tuning constant. Both terminate on the root's forward-error bound.

**Chain rule.** Each u_j depends on τ_j alone, so the Jacobian is J = diag(u_j′(τ_j)), and

∇_u V = J⁻¹∇_τ V,  ∇²_u V = J⁻¹(∇²_τ V − diag(∇_u V ⊙ u″)) J⁻¹.

∇_τ V must come from the stable τ form (Section 6.3), not from −λ g_ρ, which reintroduces the cancellation.

**Representability near u = 1.** Double spacing near 1 limits λ resolution to about ε · r/Σγ⁻¹. In the covered case this is harmless: V is analytic at u = 1 with O(1) slope, so rounding u costs about |V_u| ε in V. In the uncovered case the barrier keeps iterates away from u = 1. An evaluation that rounds onto u = 1 returns +∞, and the trust region rejects the step as part of the algorithm, not as a fallback.

### Theorem 5 (corners of overlapping penalties) [first-order: proven; blow-up analyticity: sketch for |J| = 2, conj for |J| ≥ 3; num]

Let J be the set of coordinates sent to ∞ together (τ_j → 0 for j ∈ J), with the rest fixed. Let P_j be S_j restricted to the released subspace. Then M(τ) = (Σ_{j∈J} P_j/τ_j)⁺ (plus finite parts).

**First-order expansion.**
V(τ) = V(0) + h(τ) + O(|τ|²), with h(τ) = ½tr(M(τ) C) and C = Sch − g gᵀ.
- M(τ) is the Anderson–Duffin parallel sum of the τ_j P_j⁺. It is jointly concave, monotone and positively homogeneous of degree 1 (Anderson & Duffin 1969).
- So h is 1-homogeneous and Lipschitz, and it is concave whenever C ⪰ 0.
- V has the directional derivative V′(0; d) = h(d).

**Differentiability.**
- If the penalties are **non-overlapping** (a basis exists in which the P_j have disjoint supports; for example different smooths, or a double penalty with complementary ranges), M is block-diagonal and linear in τ. Then V is analytic at the corner and ordinary bound-constrained theory applies.
- If some direction is penalised by two coordinates (tensor products: S₁ = D₂ ⊗ I, S₂ = I ⊗ D₂), h is **not linear** and V is **not differentiable at the corner** in any chart that is a product of one-dimensional charts. Measured at 45°: h = −29.105, against −31.827 for the linear interpolant [num].

**Blow-up (|J| = 2).**
1. P₁ + P₂ ≻ 0 on the released space, so a congruence T diagonalises both: TᵀP₁T = diag(p₁), TᵀP₂T = diag(p₂) (Horn & Johnson 2013, §7.6).
2. Then M = T diag(m_k) Tᵀ with m_k = τ₁τ₂/(p₁ₖτ₂ + p₂ₖτ₁).
3. In (r, s), with τ₁ = rs and τ₂ = r(1 − s): m_k = r · s(1 − s)/(p₁ₖ(1 − s) + p₂ₖ s).
4. The denominator is positive on [0, 1] (p ≥ 0, not both zero); the pole lies outside [0, 1] at s* = p₁ₖ/(p₁ₖ − p₂ₖ). So m is analytic in (r, s) on a neighbourhood of {0} × [0, 1].
5. Finite penalties R give M = D(I + RD)⁻¹ with D = diag(m), which is analytic in D. By Lemma 0, V is analytic in (r, s).

The corner becomes the smooth edge r = 0. Every point of that edge is the same model, so ∂_s V(0, s) ≡ 0 and ∂_r V(0, s) = h(s, 1 − s). Conditioning in s is set by the eigenvalue ratios p₂ₖ/p₁ₖ, through the distance from s* to [0, 1] (measured 3.0e-2).

**Corner KKT.** τ = 0 is a feasible point of the orthant.
- **Necessary:** h(d) ≥ 0 for all d in the simplex. In particular h(e_j) ≥ 0: the slope of V along coordinate j alone, with the others pinned.
- **Sufficient** (strict local minimum): min over the simplex of h > 0. The uniform O(|τ|²) remainder comes from blow-up analyticity.
- When C ⪰ 0, h is concave on the simplex, so the minimum is at a vertex and min_j h(e_j) > 0 suffices. The strongest version, C ≻ 0 on the released space, is the test `certify_rail_face` already makes (`rho_optimizer/rail_face.rs:332`).
- For |J| = 2 with C indefinite (measured eigenvalues in [−262, 40]): h(s) = ½Σ_k Cd_k μ_k(s) is rational in s. Its minimum over [0, 1] is found exactly by isolating the real roots of the numerator of h′ (Sturm sequences or Descartes bisection) and comparing the endpoint and critical values.
- If min h < 0, the minimiser s* gives a feasible descent direction d = (s*, 1 − s*) out of the corner.

**Duplicated penalties** (S₁ = S₂ on a block) give h(e_j) = 0 along the edges: the direction is flat and unidentified. The redundant coordinate must be merged at model construction, not handled by the optimiser.

### Proposition 6 (existence on the compactified domain) [sketch]

Suppose the ψ-domain Ψ is compact or V is coercive in ψ, the exact-fit case (Theorem 3 iii) is excluded, and A2 holds. Then V is lower semicontinuous as an extended-real-valued function on [0, 1]^m × Ψ:
- analytic on the open part and on the upper faces;
- continuous (Lipschitz) at overlapping corners by Theorem 5;
- +∞ at uncovered lower faces by Theorem 3 (ii).

By Weierstrass, a minimiser exists. Continuity at mixed corners (some u_j = 0, others u_k = 1) is proven for non-overlapping penalties. For overlapping ones it is only sketched.

### Optimiser design and its theory

The problem is: minimise V(u, ψ) with u ∈ [0, 1]^m (smoothing) and ψ in its natural domain.

**Algorithm (TRON: Lin & Moré 1999; Conn, Gould & Toint 1988).** At the iterate x_k, with the exact g and H from the evaluator:
1. **Generalised Cauchy point.** Take a projected search along P[x_k − t g], with P the projection onto the box, until the sufficient-decrease condition of the quadratic model holds inside the radius Δ.
2. **Subspace step.** Fix the variables active at the Cauchy point. On the free set F, solve the trust-region subproblem with H_FF, using Moré–Sorensen for small m or Steihaug–Toint with negative-curvature exits. Then projected-search back into the box.
3. **Faces are exact.** Projection places u_j = 0.0 or 1.0 exactly, and the evaluator is valid there by Lemma 0. There is no "near-face" state.
4. **Ratio test and radius update** as usual. An evaluation returning +∞ (uncovered barrier) gives ratio −∞ and a rejected step.
5. **Corners of overlapping coordinates** (both u_j = u_k = 0, overlapping). Evaluate h on the edge in closed form.
   - If min h > δ, the pair is certified active.
   - Otherwise take the descent direction d = (s*, 1 − s*) from Theorem 5 as the Cauchy direction. The smooth theory is used everywhere else. Global convergence with this corner oracle is only sketched (Section 7).

**Convergence** [cited].
- Suppose the local minimiser x* satisfies strict complementarity (c_j ≠ 0 on every active face; generic by Theorem 1) and second-order sufficiency on the free set.
- Then projected-gradient-type methods identify the active set in finitely many iterations (Calamai & Moré 1987, Thm 4.1; Burke & Moré 1988).
- After identification, the iteration is Newton on the face, converging quadratically (Bertsekas 1982; Lin & Moré 1999, Thm 4.x for TRON).
- In ρ none of this applies, because the face is at infinity (Theorem 2 d).
- In θ the finite-identification theorems fail, because strict complementarity fails identically (Theorem 2 a).

**Certificate.** For each smoothing coordinate at the final point:
- **Face coordinate** (u_j = 0; similarly u_j = 1 with the sign reversed): certified active iff the outward multiplier g_{u_j} > δ_j, where δ_j is a rigorous forward-error bound on the computed g_{u_j} (Section 6.4).
  - δ_j comes from the τ-regular system, whose conditioning is κ₂ of the equilibrated [Λ, A_RZ; 0, A_ZZ] at τ = 0. It **does not depend on λ**.
  - Contrast ρ: the naive g_ρ has error about ε κ₀ r e^{ρ}/2 against a signal |c| e^{−ρ}, so the SNR falls like e^{−2ρ}. The crossover is at ρ_× = ½ ln(2|c|/(ε κ₀ r)).
- **Degenerate face** (|g_{u_j}| ≤ δ_j): the coordinate goes into the critical cone. The second-order condition is checked on F ∪ {degenerate}, with that coordinate restricted to its feasible half-line.
- **Free set:** certify H_FF ≻ 0 via Cholesky with λ_min(H_FF) > ‖δH‖₂ (Weyl). Then require the Newton decrement ½g_Fᵀ H_FF⁻¹ g_F ≤ δV, the forward-error bound on V.
  - The decrement is invariant under linear reparameterisation of F. So the choice of chart does not matter for the interior test; it matters only for the face test, where u makes the multiplier O(1).
  - In ρ, the same decrement test falsely certifies the fake tail points of Theorem 2 (d). Those are value-optimal to within δV but have arbitrary λ.
- **Overlapping corner:** min over [0, 1] of h(s) > δ_h, from exact root isolation.

The certificate follows from the standard second-order sufficiency theorem for bound constraints (Bertsekas 1982, Prop. 1–2; Nocedal & Wright 2006, Thm 12.6), applied with perturbation bounds.

---

## 4. Numerical checks

Setup: n = 200, a 12-column cubic B-spline basis with uniform knots extended beyond the data (so that the D2 null space is exactly the linear functions), D2 penalty with r = 10, and M_p = 2. The "instrument" is Chebyshev interpolation of V on [−a, a] with a = 0.5/γ_max, used only as an independent test of analyticity and derivatives. Outputs are in `SP/theory/compactification/check{1,2,3,3b}.out`.

### 4.1 Upper face is analytic in τ; the closed-form c is exact (`check1_face.py`)

| case | degree at which Chebyshev coefficients reach 1e3·ε·abs(c₀) | mean decay ratio | relative error of V′(0) (Chebyshev vs closed-form c) |
|---|---|---|---|
| Gaussian fixed φ, null truth | 19 | 0.275 | 8.4e-13 |
| Gaussian profiled φ, null truth | 18 | 0.271 | 2.1e-12 |
| Gaussian fixed φ, sin truth | 18 | 0.227 | 1.2e-13 |
| Gaussian profiled φ, sin truth | 17 | 0.223 | 7.4e-15 |
| logistic LAML, null truth | 19 | 0.285 | 1.8e-13 (omitting the dW term: 4.0e-3) |
| logistic LAML, smooth truth | 16 | 0.212 | 2.1e-13 (omitting the dW term: 5.4e-3) |

Geometric decay over a window symmetric about τ = 0 is the numerical signature of real-analyticity across the face, including τ < 0.

**θ chart.**
- Maximum odd Chebyshev coefficient: 2.7e-14 to 2.0e-13, so V is even in θ.
- V_θ(0) is between 4e-11 and 5e-10 (zero to rounding).
- V_θθ(0)/(2c) = 1.000000000 in all four Gaussian cases.

**Tail law and naive ρ-gradient** (profiled φ, null truth):

| ρ | stable g_ρ | −c e^{−ρ} | naive g_ρ | relative error of naive |
|---|---|---|---|---|
| 12 | 1.808666e-3 | 1.817677e-3 | 1.808662e-3 | 2.2e-6 |
| 16 | 3.328888e-5 | 3.329191e-5 | 3.364025e-5 | 1.1e-2 |
| 20 | 6.097617e-7 | 6.097627e-7 | 8.107123e-6 | 1.2e+1 |
| 24 | 1.116819e-8 | 1.116819e-8 | −5.583674e-4 | 5.0e+4 (wrong sign) |

**Face frequency.** Under null-space truth (y = 1 + 2x + noise), c > 0, meaning the face is a KKT point and the REML optimum, in 0.682 of R = 400 replicates with fixed φ and 0.680 with profiled φ. This is consistent with the point mass at the boundary for variance components documented by Crainiceanu & Ruppert (2004); for this design the mass exceeds the Self & Liang (1987) 50:50 value.

### 4.2 Charts compared under a 1-D projected Newton method with the exact Hessian (`check2_charts.py`)

(A) Exact τ derivatives against the instrument, at τ ∈ {−4.3e-5, 0, 5.7e-5}: V_τ agrees to 11 digits and V_ττ to 9 digits.

(B) Projected Newton (Bertsekas-style projected arc with Armijo), from ρ₀ ∈ {−5, 0, 5, 10, 20}:

| dataset | ρ chart | τ chart | u chart |
|---|---|---|---|
| face, fixed φ (c = 1948) | 17–29 iterations, stops at fake ρ = 36.5–37.2 | 1 iteration, face KKT | 1–6 iterations, face KKT, g_u = 4.834 |
| face, profiled φ (c = 174.6) | 13–28 iterations, stops at fake ρ = 32.9–33.4 | 1 iteration, face KKT | 1–3 iterations, face KKT, g_u = 4.813 |
| interior, fixed φ | 5–21 iterations, ρ* = 1.34 | 7–16 iterations | 4–8 iterations |
| interior, profiled φ | 5–23 iterations, ρ* = −1.21 | 7–15 iterations | 4–5 iterations |

The ρ-chart counts on face data are one e-fold per iteration, as Theorem 2 (d) predicts. The u chart is uniformly the fastest in the interior. It is also the only chart with both faces finite and an O(1) multiplier. The τ chart has the lower face at τ = ∞.

(C) Noise law of the naive ρ-gradient at a face optimum:

| ρ | measured error of g_ρ | ε κ(H) r/2 | signal c e^{−ρ} |
|---|---|---|---|
| 10 | 5.2e-11 | 3.5e-12 | 8.8e-2 |
| 14 | 3.2e-9 | 1.9e-10 | 1.6e-3 |
| 18 | 3.8e-8 | 1.0e-8 | 3.0e-5 |
| 22 | 2.8e-6 | 5.6e-7 | 5.4e-7 |
| 26 | 1.6e-4 | 3.1e-5 | 1.0e-8 |

The error grows like e^{ρ}, within a factor of 5 to 15 of the κ-prediction. The signal decays like e^{−ρ}. SNR ≈ 1 at ρ ≈ 22. In u the face multiplier is 4.834 with a λ-independent error.

### 4.3 Lower end and corners (`check3_lower_corner.py`)

- **(i-a) Covered** (S₂ = I, λ₂ = 1). Chebyshev coefficients in λ₁ on [−a, a] decay geometrically (1.1e2 down to 3.3e-9 by degree 13), so V is analytic in λ₁ through 0. dV/dλ₁(0) = −25.21, and g_ρ₁/e^{ρ₁} = −25.196, −25.210 and −25.210 at ρ₁ = −10, −15 and −20.
- **(i-b) Uncovered** (S₂ = U_ZU_Zᵀ). g_ρ₁ = −4.937, −4.99956, −5.000001 and −5.0005 at ρ₁ = −5, −10, −15 and −20, against −r/2 = −5. This is the barrier. The drift at −25 (−4.970) is finite-difference noise in the test.
- **(i-c) Exact fit, profiled φ.** g_ρ = 83.6, 93.80, 93.9987 and 94.0000 at ρ = −5, −10, −15 and −20, against (n − M_p − r)/2 = 94. V → −∞.
- **(ii) Tensor corner** (S₁ = D₂ ⊗ I, S₂ = I ⊗ D₂; 6 × 6 marginals).
  - h(e₁) = −17.38, h(e₂) = −27.63 and h(e₁ + e₂) = −41.16, while h(e₁) + h(e₂) = −45.01.
  - Finite-difference directional derivatives match h(d) to 5 digits at 0°, 22.5°, 45°, 67.5° and 90°. The linear interpolant is off by 2.0, 2.7 and 1.8 at the three interior angles. So V is not differentiable at the corner.
  - C has eigenvalues in [−262, 40], so it is indefinite.
  - Blow-up: Chebyshev coefficients in r (at s = 0, 0.3 and 1) decay to about 1e-15 by degree 11, so V is analytic in r. In s on [0, 1] at fixed r, the coefficients decay to about 1e-14 by degree 60, slower because the nearest pole of m_k(s) is 3.0e-2 from [0, 1].
  - In this dataset every h is negative, so the corner is not optimal. The check tests smoothness, not optimality.

---

## 5. Literature

- D. P. Bertsekas (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM J. Control Optim.* 20(2):221–246. Projected Newton, active-set identification, superlinear convergence.
- A. R. Conn, N. I. M. Gould, Ph. L. Toint (1988). Global convergence of a class of trust region algorithms for optimization with simple bounds. *SIAM J. Numer. Anal.* 25(2):433–460. doi:10.1137/0725029. Generalised Cauchy point.
- T. F. Coleman, Y. Li (1996). An interior trust region approach for nonlinear minimization subject to bounds. *SIAM J. Optim.* 6(2):418–445. An affine-scaling alternative. It is not recommended here, because it keeps iterates strictly interior and so never lands exactly on a face.
- C.-J. Lin, J. J. Moré (1999). Newton's method for large bound-constrained optimization problems. *SIAM J. Optim.* 9(4):1100–1127. TRON: projected search plus subspace trust region, finite identification and quadratic convergence.
- P. H. Calamai, J. J. Moré (1987). Projected gradient methods for linearly constrained problems. *Math. Program.* 39:93–116.
- J. V. Burke, J. J. Moré (1988). On the identification of active constraints. *SIAM J. Numer. Anal.* 25(5):1197–1211.
- J. J. Moré, G. Toraldo (1991). On the solution of large quadratic programming problems with bound constraints. *SIAM J. Optim.* 1(1):93–113. [citation not re-verified]
- J. Nocedal, S. J. Wright (2006). *Numerical Optimization*, 2nd ed. Springer. Ch. 12 (second-order conditions) and §16.7 (bound constraints).
- D. Bates, M. Mächler, B. Bolker, S. Walker (2015). Fitting linear mixed-effects models using lme4. *J. Stat. Softw.* 67(1):1–48. The θ relative-factor parameterisation and its boundary.
- S. N. Wood (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS B* 73(1):3–36. doi:10.1111/j.1467-9868.2010.00749.x. Stable log|S|₊ reparameterisation; ρ-chart derivatives.
- W. N. Anderson, R. J. Duffin (1969). Series and parallel addition of matrices. *J. Math. Anal. Appl.* 26(3):576–594. doi:10.1016/0022-247X(69)90200-5. Concavity and homogeneity of the parallel sum.
- A. van der Sluis (1969). Condition numbers and equilibration of matrices. *Numer. Math.* 14:14–23. doi:10.1007/BF02165096. Diagonal equilibration is optimal to within a factor √(#blocks), so choosing between the τ and λ forms is a scaling decision, not a threshold.
- S. G. Self, K.-Y. Liang (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA* 82(398):605–610.
- C. M. Crainiceanu, D. Ruppert (2004). Likelihood ratio tests in linear mixed models with one variance component. *JRSS B* 66(1):165–185. Point mass of the (RE)ML estimate at the boundary.
- T. Kato (1995). *Perturbation Theory for Linear Operators*, reprint of the 2nd ed. Springer. Ch. II, analytic perturbation of constant-rank operators. [standard; edition not re-verified]
- R. A. Horn, C. R. Johnson (2013). *Matrix Analysis*, 2nd ed. Cambridge Univ. Press. §7.6, simultaneous diagonalisation by congruence. [section number from memory]
- T. J. Hastie, R. J. Tibshirani (1990). *Generalized Additive Models*. Chapman & Hall. Ch. 3, degrees of freedom as a smoothing index.
- N. J. Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. Forward-error constants γ_n for the δ bounds.

---

## 6. Consequences for gamfit

All paths are relative to `/home/user/gam` at HEAD 486fd7441a. The opt crate is `~/.cargo/git/checkouts/opt-4a38fa79856f3ac9/53ce029/opt/src/lib.rs`. The failure log is `SP/q1561/all-tests.log`.

### 6.1 What the current failures are, in this picture

The box is `resolvability_interval` = [ln(√ε γ_min), ln(γ_max/√ε)] (`crates/gam-solve/src/estimate/rho_domain.rs:142`–`159`). Its fallback is `precision_box` = [ln √ε, −ln √ε] (`crates/gam-problem/src/log_strength.rs:27`).

At the top edge, τγ_max = √ε. The naive g_ρ noise there is about ε κ₀ e^{ρ} r/2 and the signal is c e^{−ρ}. With c ≈ ½Σγ, their ratio is O(1). In other words, **the box edge is placed at the SNR ≈ 1 crossover ρ_×** (Section 4.2 C). An optimum that is really on the face therefore:
- rails at the box edge;
- shows |Pg| equal to the noise, just above a bound derived from the same ε;
- leaves the asymptote machinery to fit a tail law in a region where the gradient is noise.

| cluster (log line) | observation | diagnosis | fixed by |
|---|---|---|---|
| prostate logit (324, 871, 886; unrailed 786, 822) | railed at ρ = 22.73 (box top), abs(Pg) = 2.28e-5 vs 7.30e-6; BFGS StepSizeTooSmall; unrailed variant 9.6e-6 vs 7.3e-6 | u = 0 face optimum seen at SNR ≈ 1; BFGS curvature y/s ≈ cτ decays geometrically | u chart, exact face evaluation, face-sign certificate |
| statsmodels additive (15518) | rails at 21.63 (upper) and −18.41 (lower) | upper: u = 0 face. Lower: λ = 0 end, covered (analytic u = 1 face) or uncovered (barrier; then the lower rail is a false stop caused by the noise floor) | u chart; Theorem 3 decides |
| iso-kappa Matérn (14600), mgcv Matérn (15831), sklearn GP (33534) | lower rails near −20.8 to −21 with abs(Pg) up to 3.6 (not noise-sized) | a λ = 0 end plus a possible (ρ, ψ) microergodic ridge. The large abs(Pg) says the box cuts a descent direction, which is exactly what a hand bound does | u chart removes the cut; the ridge is for the Matérn identifiability report |
| multinomial (13475) | many ρ near −15 | the family floor `rho_lower_bound` (`crates/gam-model-api/src/families/custom_family/options.rs:447`) acts as a hand bound; the λ → 0 end is covered or uncovered per Theorem 3 | replace the floor with u = 1 (the domain) plus a typed non-identifiability check; defer to the multinomial report for the floor's derivation |
| Weibull AFT by-factor (33697) | unrailed, abs(Pg) = 6.99e-2 vs 1.86e-3; tail snap declined ("ρ = 5.16 more than 18 e-folds inside the box"); curved variant λ_min(H) = −6.4e-4 | interior non-convergence or indefinite curvature, not a face | exact-Hessian trust region with negative-curvature steps (opt); deleting the tail snap removes a misleading decline path |
| `x1+cc(x2)` (1009) | ARC decrement stall | interior | not a compactification issue |

### 6.2 Deletions

**Delete entirely.**

- `crates/gam-solve/src/rho_optimizer/rail.rs` (230 lines). It contains `outer_coordinate_is_railed` (31), `RailTest::evaluate` (72), `is_railed` (96, with the margin test), the "box=[..] margin" Display (109), `railed_coordinate_facts` (135) and `rail_test_summary` (158). Rails exist only because the box exists.
- `crates/gam-solve/src/rho_optimizer/asymptote_certificate.rs` (725 lines). It contains:
  - `DEFAULT_ASYMPTOTE_WINDOW` = 12 (81) and `MIN_TAIL_SAMPLES` = 3 (86);
  - `AsymptoteSide` (90) and `tail_constant` (117);
  - the magic constants `EXP4_INTERIOR_GRAD_TOL` = 1e-8 (221), `EXP4_TAIL_NOISE_FLOOR` = 1e-6 (226) and `EXP4_TAIL_DRIFT_REL` = 1e-3 (230);
  - `assess_coordinate` (340).

  The tail law it fits is g_ρ = −c e^{−ρ}. Its constant c is now computed exactly at the face (Theorem 1).
- `crates/gam-solve/src/rho_optimizer/bridges.rs:3809`, `coordinate_rail_margin`.
- The box constructors in `crates/gam-solve/src/estimate/rho_domain.rs`:
  - `resolvability_interval` (144);
  - `coordinate_domain` (190);
  - `resolvability_domain_from_gram_blocks` (297);
  - `resolvability_domain_from_design` (337);
  - `resolvability_domain_and_limit_faces_from_design` (363).

  Keep `penalty_range_gammas_from_gram` (37) and `penalty_range_gammas_with_shared_nullspace` (47): they become the u-chart γ. Keep `unpenalized_fit_is_identified` (171): it decides the Theorem 3 case at u = 1.
- Per-family box builders, each to be replaced by the chart constructor:
  - `crates/gam-solve/src/gaussian_reml.rs:3752` `resolvability_rho_domain`;
  - `crates/gam-custom-family/src/fit.rs:692` `resolvability_rho_domain`, `:718` `resolvability_rho_domain_and_limit_faces` and `:883` `per_block_resolvability_rho_domain`;
  - `crates/gam-models/src/survival/base.rs:2318` `resolvability_rho_domain`;
  - `crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs:40` `joint_rho_resolvability_domain`, `:84` `realized_blocks_rho_domain` and `:109` `penalized_block_rho_domain`.
- `spatial_optimization.rs:6`, `SPATIAL_PSI_BFGS_STEP_CAP` = LN_2 (used at 1872 and 5871). This is a step cap, which the SPEC forbids. The trust-region radius replaces it.
- `crates/gam-solve/src/rho_optimizer/run.rs`:
  - the inverted-box check (#2370) inside `run_outer_uncertified`, at 8165–8233;
  - `outer_bounds` (8695), `outer_model_domain_bounds_template` (8706) and `outer_search_bounds_template` (8718).

  `install_objective_domain` (8732) becomes "install u-chart plus ψ domain". The `project_to_bounds` call sites (2766, 3032) now project onto [0, 1] in u.
- Tests to delete or rewrite as face and chart tests, all under `crates/gam-solve/src/rho_optimizer/`: `asymptote_rail_certify_tests.rs`, `rail_projection_tests.rs`, `lower_bound_outward_tests.rs`, `inverted_rho_box_tests.rs` (and the module at `run.rs:8686`–`8687`), and `run_plan_certificate_face_tests.rs`. Also remove the box text in `gam-report/src/lib.rs`, `model_types/result_types.rs` and `spatial_curvature_inference.rs`.
- The family floor `rho_lower_bound` (`options.rs:447`) is a hand bound on the lower end. Theorem 3 says what that end is.

`LOG_STRENGTH_MIN/MAX` (`crates/gam-problem/src/log_strength.rs:32`) is the representability of exp and can stay as a type invariant. The optimiser no longer sees it.

**Repurpose.**

- `crates/gam-solve/src/rho_optimizer/rail_face.rs` (`split_face_penalties` 607, `face_release_bases` 683, `assemble_face_limit` 785, `gaussian_rail_face_limit` 956, `laml_rail_face_limit` 1113) and `crates/gam-solve/src/reml/rail_face_limit.rs` (`rail_face_limit` 56, `laml_rail_face_limit_via_limit_fit` 154) already compute the τ = 0 limit model. They become the face evaluator, returning V, ∇V and ∇²V at u_j = 0 exactly.
- The `OutsideClosedForm` declines (`rail_face_limit.rs:91`–`141`) must go. Lemma 0 covers every family satisfying A2. A decline is either a typed A2 failure (separation) or a missing implementation.
- `certify_rail_face` (`rail_face.rs:332`, the C ≻ 0 test) becomes the sufficient corner test of Theorem 5.
- `subspace_split_threshold` (`rail_face.rs:150`) should use the structural rank of S_j, which is known at construction from the penalty's declared null space, rather than a numerical threshold.

### 6.3 Builds

1. **Chart (gamfit, `estimate/rho_domain.rs`).** `EdfChart { gammas: Vec<f64>, r }` per smoothing coordinate, with:
   - u(τ) and u(λ);
   - the monotone Newton inverse from Theorem 4;
   - u′ and u″;
   - the chain rule of Theorem 4.

   It is built once from `penalty_range_gammas_with_shared_nullspace` at the initial W, together with a per-coordinate lower-end kind (covered, uncovered-identified or unidentified) from Theorem 3 via `unpenalized_fit_is_identified` and rank(Σ_{k≠j}S_k). Exact-fit profiled REML is detected, and reported as a typed error, when D_p → 0 at a lower face.
2. **Evaluator in master-lemma form (gam-solve REML/LAML).** Solve the regular block system of Lemma 0 in (ã, b), with M built from τ for coordinates in τ-form and from λ for those in λ-form.
   - The choice of form is a diagonal scaling of the same system, so use van der Sluis (Jacobi) equilibration rather than a switch threshold.
   - Faces (τ_j = 0) are ordinary evaluations. The separate "limit fit" code path merges into the main evaluator.
3. **Stable gradient identity (no rank cancellation)** [proven, algebraic]. For Gaussian with fixed φ:

   ∂V/∂ρ_j = ½λ_j β̂ᵀS_jβ̂ − ½λ_j tr[(S_Q + Sch)⁻¹ Sch S_Q⁻¹ S_jQ],

   using tr(H⁻¹S_j) − tr(S_λ⁺S_j) = −tr[(S_Q + Sch)⁻¹ Sch S_Q⁻¹ S_jQ]. So V_τj = −λ_j V_ρj is O(1) with no cancellation. For LAML add the dW term.

   The existing fused eigenpair-wise subtraction (`crates/gam-solve/src/reml/reml_outer_engine/objective.rs:995`–`1135`, #2331) removes dependence on summation order. It still starts from an eigendecomposition of the assembled H = A + λS, whose rounding error ε‖H‖ ≈ ελ‖S‖ destroys A_RR's contribution in range(S_j) at the rate measured in Section 4.2 (C). Only the regular form avoids assembling A + λS. This last claim is a backward-error argument; the fused path itself was not tested numerically.
4. **Exact second derivatives** in τ (Theorem 1 formula), with the LAML d²W terms obtained by differentiating the regular system (analytic by Lemma 0). Cross terms ∂²V/∂τ_j∂x_F at τ_j = 0 are needed for the trust-region model.
5. **Corner oracle.** For overlapping active pairs: simultaneous diagonalisation of (P₁, P₂), the closed-form h(s), and real-root isolation of the numerator of h′ on [0, 1].

### 6.4 opt crate (general outer-optimiser work)

- **Generalised Cauchy point and projected search** (CGT 1988; Lin–Moré 1999). The current trust-region loop (`lib.rs:6580`–`6660`) takes a Steihaug–Toint step on the free set from `active_mask` and then `project_point`. There is no Cauchy point along the projected path, so faces are found by clipping rather than by the model. This needs rewriting as TRON.
- **Typed face certificate.** `kkt_projected_gradient` (`lib.rs:3141`) zeroes g at a bound using a sign test g ≥ 0 with no error bound. `BoxSpec::active_mask` (3176) and `second_order_active_mask` (3189, 6862, 10667) use strict signs. Replace them with a per-coordinate verdict {Active, Degenerate, Free} computed against caller-supplied forward-error bounds δ_j. Degenerate coordinates enter the critical-cone second-order test.
- **Objective interface.** The objective returns (V, δV, g, δg, H, δH), and every tolerance is derived from these. The literal 1e-16 and 1e-8 constants (`lib.rs:6392`–`6393`, 6522, 6642, 6648, 6880, 6990, 6995, 7120–7365) go.
- **Negative curvature.** Moré–Sorensen for small m, or Steihaug–Toint exiting to the trust-region boundary along negative curvature on the free subspace. This is needed for the Weibull-AFT cluster (λ_min(H) = −6.4e-4).
- **Decrement certificate.** Keep `DecrementBands` / `newton_decrement_verdict` (2566, 2654), but restrict them to the free set.
- **Chart.** opt stays chart-agnostic: it sees a box [0, 1]^m × Ψ. The chart lives in gamfit (item 6.3.1).
- **BFGS** (about line 7977) should not be used for smoothing coordinates once the exact Hessian is available. In the ρ chart its curvature estimate lags the geometric decay of H_ρρ near a face, which is the prostate StepSizeTooSmall.

### 6.5 Derived tolerances (structure; the constants come from the floating-point report)

- **Face multiplier bound δ_j** at u_j = 0, first order:

  δ_j ≈ (r_j/Σγ_j) · [γ_p ε κ₂(Ẽ) (t₁ + q) + 2‖Λ^{−1/2}g_R‖ · ‖Λ^{−1/2}X_RᵀWX_Z A_ZZ⁻¹ ρ_Z‖],

  where:
  - Ẽ is the equilibrated τ = 0 block matrix [Λ, A_RZ; 0, A_ZZ];
  - t₁ = ½tr(Λ⁻¹Sch) and q = g_RᵀΛ⁻¹g_R are the two terms of c;
  - γ_p = pε/(1 − pε) (Higham);
  - ρ_Z is the inner residual of the restricted fit. The second term is the inner-solve contribution.

  No factor of λ appears.
- **δV:** the forward-error bound of V from the same system; for LAML add the inner-residual term.
- **δH:** from the second-derivative evaluation. H_FF ≻ 0 is certified by λ_min > ‖δH‖₂.
- **δ_h (corner):** the forward error of h(s) at the isolated critical points.

---

## 7. Open problems

1. **|J| ≥ 3 overlapping corners** [conj]. Three or more PSD matrices are not in general simultaneously congruence-diagonalisable, so M(τ) = (Σ P_j/τ_j)⁻¹ is algebraic but not a sum of simple rational terms.
   - Conjecture: V is analytic after iterated blow-ups.
   - For commuting Kronecker penalties (te/ti), which are simultaneously diagonalisable, the permutohedral (toric) blow-up should suffice.
   - A corner certificate needs min over the simplex of h > δ. For commuting penalties h is rational on the simplex. The general case needs polynomial optimisation.
2. **Global convergence with the corner oracle.** V is non-smooth only on the codimension-2 set of overlapping active pairs, and an exact directional-derivative oracle is available there. A proof that TRON with that oracle converges globally is missing. Alternatively, run the trust region in (r, s) blow-up coordinates for such pairs.
3. **Separation (LAML).** When the restricted fit b₀ does not exist, the u = 0 face does not exist (A2 fails), and V's limit may be −∞. A typed separation detector (an LP on the N-restricted design) is needed before the face evaluator.
4. **Multi-coordinate u-chart conditioning.** The chart uses γ from the initial W, frozen. Whether ∇²_u V stays uniformly well scaled in the interior across families (conjectured, since u is the edf fraction) and with ψ coordinates present has not been proven. Only 1-D iteration counts were measured.
5. **(ρ, ψ) Matérn geometry.** For the Matérn/GP clusters the natural compactification is joint: the microergodic combination σ²/range^{2ν} together with range. That is deferred to the Matérn identifiability report. The u chart alone removes the hand cut but not a ridge.
6. **Explicit δ constants.** Only the structure is given here; the γ_n constants and the inner-residual coupling belong to the floating-point report.
7. **Degenerate strict complementarity** (abs(c_j) ≤ δ_j). The critical-cone second-order test is specified, but the rate of convergence to a degenerate face is linear, not quadratic. The count of such cases in the test suite has not been measured.
8. **Non-analytic likelihoods.** Some survival or transformation links are C^k only. V is then C^{k−2} at faces, and the exact-Hessian theory needs k ≥ 4. The families involved have not been audited.
9. **Mixed corners** (some u_j = 0, others u_k = 1) for overlapping penalties: continuity is sketched, not proven (Proposition 6).
