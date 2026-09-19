# Convergence theory: transformation and survival models

Lane: `transformation-survival`. Scope: the monotone conditional transformation model (CTN, `transformation_normal`), the survival transformation / Royston–Parmar models (Weibull working scaffold, by-factor smooths, cause-specific competing risks), the latent-frailty survival family (individual frailty, interval censoring), and location-scale survival.

Every claim is tagged with one of three labels:
- **[P]** proven here, or a standard theorem with a precise citation;
- **[N]** checked numerically (the scripts are in `SP/theory/transformation-survival/`, where SP = `/tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad`);
- **[C]** conjectured.

Code references are relative to `SP/main_src/crates/` unless stated.

Failing clusters covered:

| id | tests (q1561) | symptom |
|----|---------------|---------|
| W | `lifelines_weibull_aft_by` ×2 | dim=6, no rails, \|Pg\|=0.0699 vs 0.00186, MaxAttempts after 106 its; curved arm ARC λ_min=−6.4e-4 |
| T | `scipy_pit_transformation_normal`, `scipy_yeojohnson`, `scipy_boxcox_univariate_lambda`, `r_tram_smooth_continuous_covariate_transformation` | 600 s timeout |
| F | `coxph_frailty` ×2, `lifelines_interval_censored` | 600 s timeout |
| K | `competing_risks_truth_recovery` (timeout), `competing_risks_cif` (METRIC_OFF) | |
| L | `gamlss_gaussian_survival_ls` (timeout on real data, METRIC_OFF synthetic) | |
| R | `rstpm2 smooth_covariate` | METRIC_OFF (overfit) |

---

## 1. Summary

1. **The CTN inner problem is self-concordant, and its minimizer is unique [P].** The inner objective is Σ½h_i² − Σ log h'_i + ½θᵀS_λθ, with h and h' linear in θ. This is a convex quadratic plus a sum of −log(affine) terms, so it is self-concordant. It follows that:
   - damped Newton with step 1/(1+δ) converges globally, and quadratically once δ < 1/4;
   - the minimizer is unique whenever the Hessian is PD on the feasible face, so the "multiple equivalent modes" comment at `gam-models/src/transformation_normal/fit.rs:413` describes inner-termination noise, not the geometry;
   - inner error propagates to LAML as |ΔV| ≤ δ² + p·|log(1−δ/(1−δ))|, which gives a *derived* inner tolerance δ* = ε_V/(2p), plus an explicit gradient version (Thm 1.4).

   The model has no location or scale flat direction (F_Z = Φ is fixed). Its penalty null space {a + b·y} is a strictly concave Gaussian-linear transformation model, so "penalize by default" recovers a proper null model.
2. **Overlapping CTN penalties put every face at ρ = −∞ as a plateau [P, N].** When a covering subset of the penalties spans the common range, V is real-analytic in λ (not in ρ) at λ_j = 0. So ∂V/∂ρ_j = λ_j ∂V/∂λ_j → 0 whatever the sign of ∂V/∂λ_j. Two consequences:
   - A ρ-chart descent walks to ρ_j → −∞ forever. This is the Box-Cox, Yeo-Johnson and PIT timeout mechanism, given `UNBOUNDED_OUTER_ITERATIONS = usize::MAX` at `gam-solve/src/rho_optimizer/run.rs:420`.
   - A ρ-chart gradient test |∂V/∂ρ_j| ≤ ε *falsely certifies* non-KKT points with λ_j ≈ 0. That is a mechanism for over-fitting (cluster R [C]).

   The certificate is the one-sided KKT condition ∂V/∂λ_j ≥ 0 at λ_j = 0, which is exact and finite. The null-recovery end (λ → ∞ along a common scaling) is analytic in μ = 1/λ, with a proof valid for *any* smooth likelihood whose null model is strictly concave.
3. **The Khatri-Rao monotonicity cone is massively redundant [P].** The cone at `gam-problem/src/constraint_set.rs:423` / `transformation_normal/custom_family.rs:269` has n·(p_resp−1) rows. Identical rows ψ(x_i) = ψ(x_j) make exactly parallel active constraints, which violates LICQ (non-unique multipliers, degenerate working sets). Replacing the rows by the extreme rays of cone{ψ(x_i)} leaves the feasible set identical. For intercept-only models (`price ~ 1`) the n·(K−1) rows reduce to K−1 simple bounds.
4. **The by-factor null-space ridge is an exact LAML identity [P, N].** For `x + s(x, by=group)` with G = 2 levels, parametric x, and per-level null-space strengths (λ_A, λ_B), V depends only on λ_eff = λ_Aλ_B/(λ_A+λ_B), exactly and for any likelihood. The outer Hessian is singular at the optimum and indefinite on the side where λ_eff exceeds its optimum. This explains W: no rails, ARC with λ_min < 0, and a line search that dies. The fix is a single-strength contrast block orthogonal to x, which gives the same V with no ridge. The rationale at `gam-terms/src/smooth/term_design.rs:2149-2150` ("do NOT project the overlapping continuous axis out") is wrong for the *null* part: the projection keeps the by-factor contrast exactly.
5. **Scaffold aliasing creates an asymptote in V [P, N].** Suppose the tangent of a working-baseline offset (Weibull θ at `fit_orchestration/fit.rs:3450-3570`; the latent chart θ at `survival/latent/survival/mod.rs:975-1225`) lies in the span of the inner time design. Then:
   - either θ is exactly flat, or
   - once the compensating direction carries a shrinkage λ_s (the Marra-Wood penalty at `mod.rs:1675`), V(ρ_s) = min_η[F(η) + ½log(1 + I(η)e^{−ρ_s})] is *strictly decreasing with no minimizer*, and ∂V/∂ρ_s ∝ −e^{−ρ_s}.

   This accounts for the ρ ≈ 20–25 coordinates in W. The fix is a direct sum: the scaffold *is* the unpenalized {1, log t} block, the outer θ is deleted, and no shrinkage is put on a direction whose null model (log-t slope 0, i.e. zero hazard) lies on the boundary of the monotonicity domain.
6. **An individual lognormal frailty with log σ as an unpenalized inner coefficient is ill-posed [P, N].** This is the setup at `mod.rs:1613-1645`. The log-likelihood is analytic in s = σ², with ℓ(s) = ℓ(0) + a·s + O(s²) and a = ½Σ[(d_i − H_i)² − H_i] (the Commenges-Andersen score). Consequences:
   - The flat-prior integral over τ = log σ diverges for *every* data set.
   - If a < 0, inner Newton steps in τ tend to −½ forever.
   - The Laplace log-det term becomes ½log(2ε): the stopping tolerance sets V, not the data.

   The fix is to make s = σ² ≥ 0 an outer hyperparameter (a domain constraint) with the KKT test at s = 0. A *shared* frailty is simply a Gaussian random-effect block with λ = 1/s. The `coxph_frailty` test formula `Surv(t, event) ~ x` (test file line 232) omits the group, so the model is misspecified: it fits an individual frailty to shared-frailty data.
7. **Interval censoring is concave, so the obstacle is identification [P, N].** For any log-concave error law (normal, Gumbel-min/PH, logistic/PO), log[F(z_R) − F(z_L)] is jointly concave (Pratt 1981). The lognormal-frailty mixture is log-concave too, because it is a convolution of log-concave densities. So the latent interval likelihood is concave in (θ, β) for fixed σ. The obstacle is the *non-identification* of σ against the flexible baseline in `SurvInterval(L,R,event) ~ 1`, which the test file itself documents (lines 33–37). The fix is to not estimate an unidentified parameter: fix s = 0 there structurally, or keep s as an outer coordinate certified by KKT.
8. **Cause-specific competing risks are exactly separable [P].** Each cause has its own coefficient and penalty blocks (`fit_orchestration/fit.rs:2606-2760`), so V(ρ) = Σ_c V_c(ρ_c) exactly. A joint outer optimization is strictly worse: the joint |Pg| is dominated by the worst cause, and one asymptotic cause (item 5) blocks all of them. The fix is to solve and certify each cause independently.
9. **Location-scale survival has an exact 2-dimensional gauge [P]; attributing METRIC_OFF to the slice choice is [C].** The model z = (h(t) − η(x))/σ(x) has the gauge (h, η, σ) ↦ (ch + a, cη + a, cσ). LAML is not invariant across gauge slices, because the flat prior on unpenalized coordinates picks up a non-constant Jacobian. The code's slice (pin the σ intercept, `survival/location_scale/prepare.rs:504-534`) is one choice. The recommended slice pins the log-t slope of h to 1 (currently only in the rank-2 reduction of `time_block.rs`).
10. **The timeouts come from iteration count, not per-iteration cost [P for the mechanism, C for each individual log].** Every structure above (plateau, asymptote, ridge, improper integral) gives an outer sequence with no finite limit in the ρ chart, and the outer loop has no termination other than certification. Once items 2, 5 and 6 are fixed, V extends as a C^ω function to the compactified domain λ ∈ [0, ∞] (charts λ and μ = 1/λ). A second-order method with exact derivatives then reaches an ε-KKT point in O(ε^{−3/2}) iterations (Cartis-Gould-Toint). That is a termination *theorem*, not a cap.

---

## 2. Setup and notation

- **Penalty and LAML.** θ ∈ ℝ^p are inner coefficients. S_λ = Σ_j λ_j S_j with S_j ⪰ 0, λ_j = e^{ρ_j}. N = ∩_j ker S_j is the common null space and R = N^⊥. The LAML criterion (minimized) is

  V(λ) = f(θ̂) + ½θ̂ᵀS_λθ̂ + ½ log det H(θ̂) − ½ log|S_λ|_+,  H = ∇²f + S_λ,

  where f = −ℓ and |·|_+ is the product of the eigenvalues on R.
- **CTN.** h(y | x) = Σ_{k,l} θ_{kl} a_k(y) ψ_l(x) = b(y, x)ᵀθ, and h' = ∂_y h = d(y, x)ᵀθ. With Z = h(Y|X) ~ N(0, 1):

  f(θ) = Σ_i [½(b_iᵀθ)² − log(d_iᵀθ)],  d_iᵀθ > 0.

  The response basis a_k is cumulative (I-spline type) for k ≥ 1, and a_0 ≡ 1. Monotonicity is enforced as c_k(x_i) := ψ(x_i)ᵀθ_k ≥ 0 for k ≥ 1 and all i. This is the Khatri-Rao cone.
- **Newton decrement.** δ(θ) = (gᵀH^{−1}g)^{1/2}, where g is the inner gradient.
- **Survival transformation model.** log H(t | x) = o_θ(t) + X(t, x)β, where o_θ is a working-baseline offset. For the Weibull scaffold, o_θ = θ_k(log t − log θ_s).
- **Latent frailty model** (`mod.rs:1-18`). log H_0 = q(a) = B(a)γ, H(a | U) = H_0(a)e^U, U ~ N(μ, σ²), μ = Xβ.

---

## 3. Results

### 3.1 CTN: convexity, uniqueness, inner tolerance, identifiability

**Theorem 1.1 (self-concordance) [P].** f + ½θᵀS_λθ is standard self-concordant on {θ : d_iᵀθ > 0 ∀i}. It is strictly convex iff M := BᵀB + DᵀW D + S_λ ≻ 0, where W = diag((d_iᵀθ)^{−2}).

*Proof.* The function −log(d_iᵀθ) is the composition of −log with an affine map, which is standard self-concordant (Nesterov & Nemirovskii 1994, Prop. 2.1.1 and Cor. 2.1.1). A convex quadratic has zero third derivative, so it is self-concordant. Sums of self-concordant functions are self-concordant. The Hessian is M, so strict convexity is equivalent to M ≻ 0. ∎

The same argument applies on any face of the monotonicity cone, because the constraints are linear.

**Corollary 1.2 [P].** The damped Newton iteration θ⁺ = θ − (1+δ)^{−1}H^{−1}g decreases f by at least δ − log(1+δ). Once δ ≤ 1/4, full steps satisfy δ⁺ ≤ δ²/(1−δ)² (Boyd & Vandenberghe 2004, §9.6.2–9.6.4). Hence the number of inner iterations is bounded by (f(θ₀) − f*)/(0.25 − log 1.25) + log₂log₂(1/δ*), and the minimizer is unique. The "two modes" that the q1561 log shows differing by 1.39e-4 in LAML, against a 2.27e-5 audit bound, are two inexact inner terminations of a strictly convex problem.

**Theorem 1.3 (inner error in V) [P].** Let θ be an inner iterate with decrement δ < 1/2, and let θ* be the exact minimizer at fixed λ. Then

  |V(θ) − V(θ*)| ≤ δ² + p·|log(1 − δ/(1−δ))| = δ² + pδ + O(δ²).

*Proof.* The penalized part obeys 0 ≤ F(θ) − F(θ*) ≤ δ² for δ ≤ 0.68 (B&V 2004, eq. 9.50). The iterate satisfies r := ‖θ − θ*‖_θ ≤ δ/(1−δ) (Nesterov 2004, Thm 4.1.13), and for r < 1 the Hessians obey (1−r)²H(θ) ⪯ H(θ*) ⪯ (1−r)^{−2}H(θ) (Nesterov 2004, Thm 4.1.6). So |log det H(θ) − log det H(θ*)| ≤ 2p|log(1−r)|. The term log|S_λ|_+ does not depend on θ. ∎

*Derived inner tolerance.* Allot half of the outer value resolution ε_V to the inner solve; ε_V itself comes from the fp-error lane (`docs/convergence_theory/fp-error-analysis.md`). This gives δ* = ε_V/(2p). There is no constant: p is the coefficient count and ε_V is derived elsewhere.

**Theorem 1.4 (inner error in ∇V) [P to first order in δ].** The analytic gradient uses the envelope formula ∂_jV = λ_j[½θᵀS_jθ + ½tr(H^{−1}S_j) − ½tr(S_λ^+S_j)] + (third-derivative term). At an inexact θ it omits gᵀ∂θ̂/∂ρ_j = −λ_j gᵀH^{−1}S_jθ. By Cauchy–Schwarz in the H-norm, |gᵀH^{−1}v| ≤ δ‖v‖_{H^{−1}}. The trace terms change by at most a factor (1−r)^{∓2}. Hence

  |Δ∂_jV| ≤ δ·(‖λ_jS_jθ‖_{H^{−1}} + tr(H^{−1}λ_jS_j)) + O(δ²),

and the inner tolerance that keeps the gradient error below ε_g/2 is

  δ*_g = ε_g / [2 max_j (‖λ_jS_jθ‖_{H^{−1}} + tr(H^{−1}λ_jS_j))].

Every quantity is available at the current iterate.

**Proposition 1.5 (no location/scale invariance; proper null model) [P].**
- For θ ↦ θ' with h' = α + βh, β > 0 constant, f changes by Σ[½((α+βh_i)² − h_i²)] − n log β. This is not identically zero, so there is no flat direction in θ.
- Take the null space {h = a + b(y − m)}, which for intercept-only models is shared by S1 (order-1 affine-invariant, `response_basis.rs:155`), S2 (order-2) and S3 (the double penalty, `kronecker_design.rs:443-457`). On it, f(a, b) = ½Σ(a + b(y_i − m))² − n log b, whose Hessian is PD for n ≥ 2 distinct y. So the λ → ∞ limit is the Gaussian-linear transformation model, which exists and is unique.

**Proposition 1.6 (existence at λ = 0, an LP certificate) [P].** The unpenalized CTN MLE on the cone 𝒦 exists and is unique iff there is no v ≠ 0 with Bv = 0, Dv ≥ 0 and v in the recession cone of 𝒦.

*Proof.*
- f is closed, proper and convex.
- Along a direction v with Bv ≠ 0 the quadratic term diverges, so such v are not recession directions.
- If Bv = 0 and Dv ≥ 0 with Dv ≠ 0, then f(θ + tv) → −∞.
- If Bv = 0 and Dv = 0, v is a lineality direction.

Rockafellar (1970) Thm 27.1(b) completes the argument. ∎

The check is a single LP: maximize 1ᵀDv subject to Bv = 0, Dv ≥ 0, v ∈ rec 𝒦, 1ᵀDv ≤ 1 (a homogeneous normalization, not a box). It certifies that the λ = 0 face is admissible before any KKT test there. For a spline with K ≤ #distinct(y) it always passes.

### 3.2 Overlapping penalties: faces at ρ = −∞

**Assumption A (cover).** There is a subset J with range(Σ_{j∈J} S_j) = R.

**Theorem 2.1 (λ-analyticity at the face) [P].** Under A, fix λ_J in the open orthant and suppose H(θ̂) ≻ 0. Then V is real-analytic in (λ_j)_{j∉J} on a neighbourhood of λ_j = 0, including negative λ_j small enough that S_λ|_R stays PD.

*Proof.*
- log|S_λ|_+ = log det(Q_RᵀS_λQ_R), and Q_RᵀS_λQ_R ≻ 0 is guaranteed by λ_J alone.
- θ̂(λ) is analytic by the implicit function theorem, since ∇θ(∇f + S_λθ) = H ≻ 0 and f is analytic (for CTN on the open domain h' > 0).
- f, ½θᵀSθ and log det H are then analytic compositions. ∎

**Corollary 2.2 (plateau; false certification) [P].**
- ∂V/∂ρ_j = λ_j ∂V/∂λ_j with ∂V/∂λ_j → c_j finite. Hence |∂V/∂ρ_j| ≤ ε holds for all ρ_j < log(ε/|c_j|), *whatever the sign of c_j*.
- If c_j > 0, the face optimum is at λ_j = 0. A ρ-chart descent still never reaches it: every step gains only ~e^{ρ_j}.
- If c_j < 0, the point is not a KKT point (V decreases as λ_j grows), yet a ρ-chart gradient test accepts it. The result is under-smoothing, i.e. an overfit.

**Theorem 2.3 (dominated plateau) [P].** Suppose S_k alone covers R and λ_k → ∞ with the other λ fixed. Then ∂V/∂λ_j = O(λ_k^{−2}) and ∂V/∂ρ_j = O(λ_jλ_k^{−2}).

*Proof sketch.*
- θ̂_R = O(1/λ_k), so ½θ̂ᵀS_jθ̂ = O(λ_k^{−2}).
- Write the R-Schur complement of H as λ_kS_k|_R + A, where A is O(1) (likelihood curvature on R given N, plus the other penalties). Both tr(H^{−1}S_j) and tr(S_λ^+S_j) have the expansion tr((λ_kS_k)^{−1}S_j) + O(λ_k^{−2}) with the *same* leading term, which cancels in V.
- The third-derivative term is O(θ̂_R) · O(λ_k^{−1}). ∎

**Theorem 2.4 (null-recovery end, any smooth likelihood) [P].** Let λ = c/μ along a fixed direction c > 0 with Σ_j c_jS_j|_R ≻ 0. Assume the null model f|_N has a unique minimizer with ∇²_{NN}f ≻ 0. Then V is analytic in μ at μ = 0, including the endpoint.

*Proof.*
- Substitute θ_R = μu. Stationarity becomes ∇_Nf(θ_N, μu) = 0 and ∇_Rf(θ_N, μu) + S_cu = 0 (with S_c = Σc_jS_j), which is analytic in (θ_N, u, μ).
- At μ = 0 the Jacobian with respect to (θ_N, u) is block-triangular: [[∇²_{NN}f, 0], [*, S_c]], which is nonsingular. The implicit function theorem gives (θ̂_N, û) analytic in μ.
- log det H − log|S_λ|_+ = log det(I + (μ)S_c^{−1}A_{RR}) + log det(H_{NN} − H_{NR}(S_c/μ + A_{RR})^{−1}H_{RN}), which is analytic in μ; the log det(S_c/μ) terms cancel exactly.
- f(θ̂) + ½θ̂ᵀS_λθ̂ = f(θ̂) + ½μûᵀS_cû is analytic. ∎

**Certificate for a face (λ-chart KKT) [P].** Let Z be the set of zero coordinates, I the interior set and M the set at infinity. A candidate is certified when:
- (i) |∂V/∂ρ_i| ≤ e_i for i ∈ I;
- (ii) ∂V/∂λ_j|_{λ_j=0} > e_j for j ∈ Z, evaluated by the *bracket* ½θ̂ᵀS_jθ̂ + ½tr(H^{−1}S_j) − ½tr(S_λ^+S_j) + third-derivative term, without the λ_j factor;
- (iii) ∂V/∂μ_m|_{μ_m=0} > e_m for m ∈ M;
- (iv) the reduced Hessian on I is PD.

Each e is the gradient-error bound of Thm 1.4 plus the fp-error-lane bound. If |∂V/∂λ_j| ≤ e_j, strict complementarity fails and the case is weakly active: resolve it with the bound-constrained Newton QP in the λ chart (λ ≥ 0 is a domain constraint), not by a threshold.

*Derived chart origin* (a scale, not a bound): ρ_j⁰ = log(tr(Q_Rᵀ∇²f Q_R)/tr(S_j)), which balances penalty and information.

### 3.3 The Khatri-Rao cone: redundancy and LICQ

**Proposition 3.1 [P].** For each k ≥ 1, {θ_k : ψ(x_i)ᵀθ_k ≥ 0 ∀i} = 𝒦_ψ^*, where 𝒦_ψ = cone{ψ(x_1), …, ψ(x_n)}. If E ⊆ {ψ(x_i)} is a set of extreme-ray generators of 𝒦_ψ, then E^* = 𝒦_ψ^*. The feasible set is unchanged and the constraint count drops from n to |E|.

**Proposition 3.2 [P].** If ψ(x_i) = αψ(x_j) with α > 0 and the constraint is active, the active gradients are linearly dependent. LICQ fails and the multiplier set is a nontrivial polytope, which is unbounded in the parallel-pair direction only through the sum. Differentiability of θ̂(ρ) by the implicit function theorem requires LICQ (or a stronger condition such as strong regularity) together with strict complementarity (Fiacco 1976; Robinson 1980; Kyparisis 1985 for uniqueness of multipliers).

- *Intercept-only models* (`price ~ 1`: the Box-Cox, Yeo-Johnson and PIT cases): ψ ≡ 1, so every row of the cone is identical. The cone at `custom_family.rs:269` has n·(K−1) rows that reduce *exactly* to K−1 bounds α_k ≥ 0.
- *B-spline ψ* (tram): E ⊂ {ψ(x_i)} is computable by an LP per candidate row, or exactly by duplicate-row merging followed by conic-hull pruning. Coefficientwise θ_{kl} ≥ 0 (as in mlt) is only *sufficient*, since R^q_+ ⊆ 𝒦_ψ^*, and it is strictly smaller in general.

### 3.4 The by-factor null-space ridge

**Theorem 4.1 [P, N].** Suppose the design contains a parametric column x (flat prior), per-level columns x·1_A and x·1_B with Gaussian priors of precisions λ_A and λ_B, and other terms. Then LAML depends on (λ_A, λ_B) only through λ_eff = λ_Aλ_B/(λ_A+λ_B), for any likelihood.

*Proof.*
- Let γ_g = β + b_g. The likelihood depends on (γ_A, γ_B) only.
- Change variables to (β, γ_A, γ_B), which has unit Jacobian. The negative log-integrand is q(β, γ) + r(γ, other), where q = ½λ_A(γ_A−β)² + ½λ_B(γ_B−β)² is exactly quadratic in β.
- Laplace in (β, γ) equals exact Gaussian integration in β followed by Laplace in γ, because det H_full = det(H_ββ)·det(Schur_γ) and the Schur complement is the Hessian of r + min_β q.
- min_β q = ½λ_eff(γ_A−γ_B)², with H_ββ = λ_A + λ_B.
- The terms in V that remain explicitly in (λ_A, λ_B) are ½log(λ_A+λ_B) − ½log λ_A − ½log λ_B = −½log λ_eff. ∎

**Corollary 4.2 (outer geometry) [P].** Let Ṽ(ρ_eff) be the reduced criterion, with ρ_eff = −log(e^{−ρ_A} + e^{−ρ_B}) and w = softmax(−ρ). Then ∇²V = Ṽ'·∇²ρ_eff + Ṽ''·∇ρ_eff∇ρ_effᵀ, where ∇²ρ_eff = −w_Aw_B[[1, −1], [−1, 1]]. The curvature along the ridge tangent is −Ṽ'·w_Aw_B:
- at the optimum (Ṽ' = 0), the Hessian is singular: the minimizer is not isolated, and Newton or ARC certification that needs a PD Hessian is impossible;
- wherever Ṽ' > 0 (λ_eff above its optimum), the Hessian is indefinite.

This matches the Weibull-by curved arm (ARC, λ_min = −6.4e-4, no rails). It also matches the linear arm's line-search MaxAttempts: V is constant along a curved valley, and inner noise (Thm 1.3) dominates the decrease condition.

**Proposition 4.3 (G ≥ 3) [P].** The contrast precision P = Λ − λλᵀ/Σλ determines λ, because P_12P_13/P_23 = −λ_1²/Σλ, etc. So there is no exact ridge. The problem is still weak identification: G strengths for a (G−1)-dimensional random effect [C].

**Fix [P].** Replace the per-level null columns by the contrast block C = x·(1_g − n_g/n)-type columns orthogonal to x (G−1 columns) with *one* exchangeable strength.
- For G = 2 this is exact: V is identical and the ridge disappears.
- For G ≥ 3 it is the exchangeable prior (a modelling choice that removes G−1 weakly identified coordinates).

The wiggle (range) parts per level keep their own strengths. The comment at `term_design.rs:2149-2150` conflates the *null* part with the range part: projecting x out of the per-level null columns keeps the contrast γ_A − γ_B, which *is* the by-factor linear signal, exactly.

### 3.5 Scaffold aliasing

**Theorem 5.1 [P, N].** Let η = o_θ + Xβ with ∂o_θ/∂θ ⊂ span(X) for all θ. For the Weibull scaffold, ∂o/∂θ_k = log t − log θ_s and ∂o/∂θ_s = −θ_k/θ_s, both in span{1, log t}.
- (i) If the compensating coefficients are unpenalized, then f(θ, β̂(θ)) and log det H are constant in θ, because H = XᵀW(η)X + S depends on θ only through η. θ is an exact flat direction.
- (ii) Suppose the compensating direction s carries a penalty λ_s (a null-space shrinkage) and θ is an outer coordinate optimized on V. Then V(ρ_s) = min_η[F(η) + ½log(1 + I_s(η)/λ_s)], where I_s is the Schur-complement information of coordinate s. It is strictly decreasing in λ_s, has no finite minimizer, and has ∂V/∂ρ_s = −½I_s/(λ_s + I_s) ≈ −½I_s e^{−ρ_s}.

*Proof of (ii).*
- Moving (θ, β_s) along the aliasing direction leaves η, and hence f and XᵀWX, unchanged, while ½λ_sβ_s² strictly decreases. So the joint minimizer has β̂_s = 0.
- The Schur decomposition gives log det H = log det H_{−s} + log(λ_s + I_s(η)). The −½log λ_s from |S_λ|_+ combines with it to give ½log(1 + I_s/λ_s).
- For λ_1 < λ_2, V(λ_2) = G(η_2, λ_2) < G(η_2, λ_1) ≤ V(λ_1). ∎

**Proposition 5.2 (the shrinkage target is improper) [P].** On the survival time block, the null direction that the function-space shrinkage at `latent/survival/mod.rs:1647-1675` penalizes is the affine trend d log Λ/d log t. Its shrinkage limit, slope 0 with no offset slope, is H(t) constant in t: zero hazard, on the boundary of the monotonicity domain h' > 0. So "penalize to the null" is not a proper model there. The shrinkage only acquires a proper target *through* the scaffold offset, which is exactly the aliasing of Thm 5.1(ii).

**Fix (direct sum) [P].**
- θ_k(log t − log θ_s) = θ_k log t + c is *linear* in (θ_k, c). The Weibull scaffold is therefore just the unpenalized block {1, log t}, with the domain constraint slope > 0 (part of the monotone constraint).
- Delete the outer θ, and project the time smooth's range off span{1, log t}.
- The λ → ∞ null model is then exactly the Weibull (or the latent PH-lognormal) baseline, the "penalize by default" null.
- Check (c3): the direct-sum V equals lim_{ρ_s→∞}V to 4e-11.

### 3.6 Frailty

**Theorem 6.1 [P, N].** For a row with event indicator d_i, cumulative hazard H_i and frailty e^{σz}, z ~ N(0, 1): L_i(σ) = E[(h_ie^{σz})^{d_i}exp(−H_ie^{σz})]. This is an even analytic function of σ, hence analytic in s = σ², and

  log L_i(s) = log L_i(0) + ½s[(d_i − H_i)² − H_i] + O(s²).

*Proof.* Write κ = σz(d_i − H_i) − ½H_iσ²z² + O(σ³). Then E e^κ = 1 + Eκ + ½Eκ² + O(σ⁴) = 1 + ½σ²[(d_i−H_i)² − H_i] + O(σ⁴), because the odd moments vanish. ∎

This is the Commenges & Andersen (1995) homogeneity score, before the correction for estimated nuisance parameters.

**Corollary 6.2 (log σ as an unpenalized inner coefficient is ill-posed) [P].** With τ = log σ and a := Σ_i ½[(d_i−H_i)² − H_i]:
- (a) ∫e^{ℓ(τ)}dτ = +∞ *for every data set*, since e^{ℓ(τ)} → e^{ℓ(0)} > 0 as τ → −∞. The flat-prior Laplace integrand at `mod.rs:1613-1645` (penalties empty, initial coefficient ln σ) is improper.
- (b) If a < 0 then ℓ'(τ) = 2ae^{2τ} and ℓ'' = 4ae^{2τ}, so Newton steps are exactly −½ asymptotically.
- (c) At the stopping rule |ℓ'| = ε, ½log(−ℓ'') = ½log(2ε): the Laplace log-det contribution of τ is set by the tolerance.

**Fix [P].**
- *Individual frailty:* s = σ² is an outer hyperparameter on the domain s ≥ 0, with V analytic at s = 0. It is certified at the boundary by ∂V/∂s|_{s=0} ≥ e_s, where ∂V/∂s|_0 = −a(θ̂_0) + ½tr(H_0^{−1}∂_sH|_0). The row score's η-derivatives are explicit: ∂_η[½((d−H)² − H)] = −H(d−H) − ½H.
- *Shared frailty:* a Gaussian random-effect block z_g ~ N(0, s), i.e. a penalized block with S = I_G and λ = 1/s, inside the existing LAML machinery. λ → ∞ recovers the no-frailty model (Therneau, Grambsch & Pankratz 2003; Ripatti & Palmgren 2000).
- The Self & Liang (1987) and Stram & Lee (1994) boundary mixtures concern testing only. The optimization certificate is the KKT sign.

**Test mismatch [P, from the source].** `quality_vs_survival_coxph_frailty_hazard_multiplier.rs:135-145` simulates *shared* frailty over 12 groups, but line 232 fits `Surv(t, event) ~ x`, an individual frailty with no group. The fitted model cannot represent the data-generating process.

*Weak identification [C].* Individual frailty against a flexible baseline is identified only through the mixed-proportional-hazards structure (Elbers & Ridder 1982). It is known to be fragile (Heckman & Singer 1984), so even after the fix V(s) may be nearly flat.

### 3.7 Interval censoring

**Theorem 7.1 [P, N].** If F has a log-concave density, g(a, b) = log(F(b) − F(a)) is jointly concave on b > a (Pratt 1981; this follows from Prékopa 1973, since f(u)·1{a ≤ u ≤ b} is log-concave in (u, a, b)). This covers the normal (CTN), Gumbel-min (PH) and logistic (PO) laws.

The latent mixture F_σ(z) = E_U[1 − exp(−e^{z+U})] is the cdf of W − U, a convolution of log-concave densities, so it is log-concave for every fixed σ. With q(a) = B(a)γ linear (`mod.rs:1-18`) and μ = Xβ, the latent interval likelihood is concave in (γ, β) for fixed σ. The monotonicity constraint is linear.

**Consequence.** The `lifelines_interval_censored` timeout is not a concavity failure. The test file (lines 33–37) documents that σ is *not identified* against the monotone baseline. For any σ, B_σ = S_σ^{−1}∘S reproduces the marginal exactly in the nonparametric limit [P], and the spline span only weakly separates candidates. So the outer problem contains a near-flat σ direction plus the θ-chart axes of §3.5 under the `precision_box` (`mod.rs:1001`). For `~ 1` with no clustering, the principled model has no σ: fix s = 0 structurally. If the user requests a frailty, s is an outer coordinate with the KKT certificate of §3.6.

### 3.8 Competing risks

**Theorem 8.1 [P].** The cause-specific likelihood Π_c λ_c(t)^{1[D=c]}exp(−Σ_cΛ_c(t)) = Π_c[λ_c^{1[D=c]}e^{−Λ_c}] factorizes (Prentice et al. 1978; Kalbfleisch & Prentice 2002, §8.2). The code builds per-cause blocks, per-cause penalties and per-cause λ (`fit_orchestration/fit.rs:2640-2740`, penalty names `cause_specific_survival_cause_{c}_penalty_{j}`). So H, S_λ and |S_λ|_+ are block-diagonal and V(ρ) = Σ_cV_c(ρ_c) exactly.

Consequences:
- The joint outer is a product problem whose conditioning is the worst ratio across causes.
- Its |Pg| is dominated by the worst cause.
- A single cause on an asymptote (§3.5) or plateau (§3.2) prevents joint termination.

Solving each cause separately gives the same optimum with independent certificates. CIF METRIC_OFF is attributed to the per-cause time-block asymptote [C].

### 3.9 The location-scale survival gauge

**Proposition 9.1 [P].** z = (h(t) − η(x))/σ(x), with density φ(z)h'(t)/σ, is invariant under (h, η, σ) ↦ (ch + a, cη + a, cσ) for c > 0. This is an exact 2-parameter gauge when h and η both contain constants and σ contains an intercept.

**Proposition 9.2 [P].** The penalties transform covariantly: λ_h ↦ c^{−2}λ_h, λ_η ↦ c^{−2}λ_η, and the log σ wiggle is shift-invariant. But the flat prior on the unpenalized coordinates (the affine part of h and the intercept of η) produces an uncompensated Jacobian. The Jacobian of the map between two gauge slices depends on the point (c = 1/slope_h is a function of the coefficients), so V differs between slices by a non-constant function. The slice choice therefore changes the LAML optimum.

**Recommendation [C].** Use the slice "log-t slope of h = 1" and free the σ intercept. Its null model (λ → ∞) is log T ~ N(η, σ) with h = log t, which is exactly the parametrization of the reference lognormal AFT, and under it the unpenalized coordinates enter linearly. The current slice pins the σ intercept (`location_scale/prepare.rs:504-534`), and the slope pin exists only in the rank-2 reduction of `location_scale/time_block.rs` (`prepare_identified_time_block`, 1233+). Attributing rmse_loc = 5.09 to the slice is [C].

### 3.10 Outer termination

**Theorem 10.1 [P, conditional].** Suppose that after the fixes above V extends to a C² function on the compactified domain K = Π_j[0, ∞]_j (charts λ_j near 0, μ_j = 1/λ_j near ∞, justified by Thm 2.1 and Thm 2.4), with Lipschitz Hessian on K. Then ARC or a projected trust-region method with exact derivatives reaches an ε-KKT point of the bound-constrained problem in O(ε^{−3/2}) iterations (Cartis, Gould & Toint 2011, for the unconstrained case, applied chartwise; the bound constraints λ ≥ 0 and μ ≥ 0 are domain constraints).

In the current ρ chart, none of the hypotheses hold for clusters T, W, F and K:
- the plateau (Cor. 2.2), asymptote (Thm 5.1), ridge (Thm 4.1) and improper integral (Cor. 6.2) each yield an outer sequence with no limit point in ℝ^m;
- `UNBOUNDED_OUTER_ITERATIONS = usize::MAX` (`run.rs:420`, used at 429 and 540) turns each of these into a wall-clock timeout.

The per-iteration cost is not the cause. The Box-Cox case has n ≈ 30 and p ≈ 10, so each iteration costs microseconds to milliseconds, and 600 s means a very large iteration count [P for the arithmetic, C per test, because the timeout logs carry no diagnostics].

---

## 4. Numerical checks

The scripts are in `SP/theory/transformation-survival/` and use the venv at `SP/theory/venv`. Finite differences appear only in these tests.

**`check_a_by_ridge.py`** (Poisson; design [1, x, 1_B, x1_A, x1_B, wiggle_A, wiggle_B]) [N]:
- V along λ_eff = const has maximum spread **2.8e-14** over ρ_A ∈ [1, 25].
- At the stationary ridge (ρ_eff* = −0.6139) the Hessian eigenvalues are {≈1e-8, 0.25–0.47}: singular.
- At a non-stationary ridge point (ρ_eff = 0.7, ρ_A = 2) the eigenvalues are (−0.38, 0.70): indefinite.
- The single-strength contrast model reproduces V with difference 0 to printed precision.

**`check_b_ctn_overlap.py`** (intercept-only CTN, n = 30, cubic B-spline K = 8, penalties S2, S1, S3) [N]:
- (b1) Damped Newton takes 6 iterations, final decrement 1.9e-15, min eig H = 0.584.
- (b2) dV/dρ_2 as ρ_3 = 2, 4, 6, 8 is 1.4e-2, 4.9e-4, 4.4e-6, 6e-8: a ratio of about e^{−4} per +2 step, consistent with O(λ_k^{−2}) in Thm 2.3. The last ratios sit near the finite-difference noise floor.
- (b3) The one-sided slope in λ_2 at 0 is finite (0.00799), as Thm 2.1 predicts.
- (b4) The slope in μ at the null end is finite (−1.13), as Thm 2.4 predicts.
- (b5) BFGS in ρ from 0 ends at ρ = [1.18, −12.0, −18.8] with "precision loss": the plateau walk of Cor. 2.2.
- (b6) The face optimum ρ_1* = 1.18143 gives V = 15.0835485, with one-sided dV/dλ_2 = 0.0814 > 0 and dV/dλ_3 = 0.1346 > 0. The face is certified and matches the BFGS ρ_1.

**`check_c_scaffold.py`** (Weibull offset plus [1, log t, x], with λ_s on log t) [N]:
- V(ρ_s) falls strictly: 135.08 → 131.615 as ρ_s goes −2 → 10, with slopes −0.4985 … −0.0098.
- The implied I_s is constant at ≈ 139.2, confirming V = V_∞ + ½log(1 + I/λ_s).
- With λ_s = 0 the NLL is identical for every θ_k (θ_k + b_1 = 1.31691).
- Direct sum: V = 131.61180092 against the limit 131.61180096.

**`check_d_frailty_boundary.py`** (80-node Gauss–Hermite, n = 120) [N]:
- On a no-frailty draw with score a = −6.276, (ℓ(√s) − ℓ(0))/s → −6.277 and dℓ/dσ|_0 = 0.
- Newton in τ steps −0.468, −0.446, …, −0.5036 (→ −½).
- At iteration 11, ℓ' = −7.37e-5 and ℓ'' = −1.46e-4, so ½log(−ℓ'') ≈ ½log(2|ℓ'|).
- On a frailty draw, a = 22.75 and the interior optimum is ŝ = 0.51.

**`check_e_interval_concavity.py`** (analytic Hessians on a 141×80 grid) [N]:
- The maximum top eigenvalue of ∇²log(F(b) − F(a)) is −4.7e-8 (normal), 0 (Gumbel-min) and −1.3e-2 (logistic).
- For the latent mixture with σ ∈ {0, 0.5, 1, 2, 3} it is ≤ 0: concave for every fixed σ.

---

## 5. Literature

**Transformation models**
- Hothorn, Kneib & Bühlmann (2014), "Conditional transformation models", JRSSB 76(1):3–27. The tensor basis b(y)⊗ψ(x) and monotone coefficients.
- Hothorn, Möst & Bühlmann (2018), "Most likely transformations", Scand. J. Stat. 45(1):110–134, doi:10.1111/sjos.12291. The log-likelihood log f_Z(h) + log h' and the coefficientwise monotone (sufficient) constraint.
- Hothorn (2020), "Most likely transformations: the mlt package", JSS 92(1).

**Survival splines**
- Royston & Parmar (2002), Stat. Med. 21(15):2175–2197. The log-cumulative-hazard spline with a Weibull null.
- Liu, Pawitan & Clements (2018), "Parametric and penalized generalized survival models", SMMR 27(5):1531–1546, doi:10.1177/0962280216664760 (rstpm2).
- Pya & Wood (2015), "Shape constrained additive models", Stat. Comput. 25:543–559.
- Marra & Wood (2011), "Practical variable selection for GAMs", CSDA 55(7):2372–2387. The null-space double penalty; see Prop. 5.2 for why it is improper on a survival time block.

**LAML**
- Wood (2011), JRSSB 73(1):3–36.
- Wood, Pya & Säfken (2016), JASA 111(516):1548–1563.

**Convex analysis and optimization**
- Nesterov & Nemirovskii (1994), *Interior-Point Polynomial Algorithms in Convex Programming*, SIAM, §2.1.
- Nesterov (2004), *Introductory Lectures on Convex Optimization*, Thms 4.1.6 and 4.1.13.
- Boyd & Vandenberghe (2004), *Convex Optimization*, §9.6 (eq. 9.50).
- Rockafellar (1970), *Convex Analysis*, Thm 27.1.
- Nocedal & Wright (2006), *Numerical Optimization*, §12.2–12.3 (LICQ, KKT).
- Fiacco (1976), Math. Programming 10:287–311.
- Robinson (1980), "Strongly regular generalized equations", Math. Oper. Res. 5(1):43–62.
- Kyparisis (1985), "On uniqueness of Kuhn–Tucker multipliers", Math. Programming 32:242–246.
- Cartis, Gould & Toint (2011), "Adaptive cubic regularisation methods… Part II", Math. Programming 130:295–319 (complexity O(ε^{−3/2})).

**Log-concavity**
- Pratt (1981), "Concavity of the log likelihood", JASA 76(373):103–106.
- Prékopa (1973), "On logarithmic concave measures and functions", Acta Sci. Math. (Szeged) 34:335–343.

**Frailty**
- Commenges & Andersen (1995), "Score test of homogeneity for survival data", Lifetime Data Anal. 1(2):145–156.
- Self & Liang (1987), JASA 82(398):605–610.
- Stram & Lee (1994), Biometrics 50(4):1171–1177.
- Therneau, Grambsch & Pankratz (2003), "Penalized survival models and frailty", JCGS 12(1):156–175.
- Ripatti & Palmgren (2000), Biometrics 56:1016–1022.
- Elbers & Ridder (1982), RES 49(3):403–409.
- Heckman & Singer (1984), Econometrica 52(2):271–320.

**Competing risks**
- Prentice, Kalbfleisch, Peterson, Flournoy, Farewell & Breslow (1978), Biometrics 34:541–554.
- Kalbfleisch & Prentice (2002), *The Statistical Analysis of Failure Time Data*, 2nd ed., §8.2.

---

## 6. Consequences for gamfit

Each item gives what to delete, what to build, and the certificate or tolerance. Cluster letters refer to the table at the top.

### 6.1 Transformation-normal (clusters T and R)

1. **Delete the κ hand box.** Remove `transformation_normal/fit.rs:296` (`lower_bounds_aniso_from_data`) and `:311` (`clamp_to_bounds`). The κ coordinates are then governed by the face-KKT certificate of §3.2 (charts λ and μ).
2. **Deduplicate the cone.**
   - Where: `gam-problem/src/constraint_set.rs:423` (`KhatriRaoConeConstraints`) and the construction at `transformation_normal/custom_family.rs:227-278` (`::new(factor, (1..p_resp).collect(), p_resp)` at 269).
   - Build E, the extreme rays of cone{ψ(x_i)}: (a) merge exactly equal rows (a hash of the bit patterns, which is exact); (b) drop row i if the LP ψ(x_i) = Σ_{j≠i}ν_jψ(x_j), ν ≥ 0, is feasible, solved in exact-arithmetic-safe form by the existing active-set LP. Emit |E|·(p_resp−1) rows.
   - For intercept-only models, emit simple bounds.
   - Then `active_set.rs:1816` (`exactly_parallel_representative`), `:1878` (`khatri_rao_cone_reduced_face`) and `MAX_FEASIBILITY_REPAIR_DEPTH = 16` at `:1085` become unnecessary and should be deleted. LICQ then holds on each face (Prop. 3.2).
3. **λ-chart face certification for the overlapping response penalties** (`kronecker_design.rs:401` `build_tensor_penalties_kronecker`, with the projector at 443–457; `response_basis.rs:155`).
   - The outer works in ρ in the interior. A coordinate j is moved to the λ_j chart when λ_j·|bracket_j| < e_j, i.e. when the ρ-gradient is below its own error bound.
   - Certify by (i)–(iv) of §3.2, with the bracket computed *without* the λ_j factor.
   - Never accept |∂V/∂ρ_j| ≤ ε as stationarity for a coordinate whose bracket is negative (Cor. 2.2): that is the over-fit mode.
   - The chart origin is ρ_j⁰ = log(tr(Q_Rᵀ∇²f Q_R)/tr(S_j)).
4. **Inner tolerance.**
   - Replace any fixed inner tolerance by δ* = min(ε_V/(2p), δ*_g) (Thms 1.3 and 1.4).
   - Use damped Newton with step (1+δ)^{−1} (already licensed by `custom_family.rs:21`, `inner_objective_is_self_concordant`), capped only by the feasibility step `max_feasible_step_size` (`:196`), which is a domain constraint.
   - Delete the "multiple equivalent modes" logic and comment at `fit.rs:413`.
5. **Magic constants.**
   - `response_basis.rs:369` (6.0 in `transformation_complexity_knot_budget`) and `:381` (the n/10 sample cap). The knot count is a representation choice. With the penalty properly selected, the resolution argument needs K ≤ #distinct(y) (Prop. 1.6). Use that, not n/10.
   - `config.rs`: `MONOTONICITY_EPS = 1e-8`, `H_ABS_MAX = 1e6`, `RIDGE_FLOOR = 1e-8`, and the 160/320 widths. The monotone domain is h' > 0, enforced by the barrier itself (−log h' → ∞). A ridge floor is unnecessary once Prop. 1.5 (null model PD) and the LP certificate of Prop. 1.6 hold.
6. **λ = 0 admissibility.** Run the recession LP of Prop. 1.6 once per design. If it fails, f has no minimizer at λ = 0, so that face is excluded *by the mathematics*, not by a box.

### 6.2 Survival transformation and Weibull-by (clusters W and K)

7. **Contrast reparametrization for by-level null blocks** (`term_design.rs:2128-2160` `factor_by_level_gate`; `:2161+` `build_parametric_constraint_block_for_term`).
   - When a parametric main effect x (or its span) is present, replace the G per-level null columns x·1_g (G strengths) by G−1 contrast columns orthogonal to x in the design metric, with one strength.
   - Keep the per-level range (wiggle) penalties.
   - Correct the comment at 2149–2150.
   - This is exact for G = 2 (Thm 4.1) and removes the singular or indefinite outer Hessian.
8. **Direct-sum scaffold** (`fit_orchestration/fit.rs:3397-3435` seed; `:3450-3570` baseline profile, including the "≲10 evaluations" comment at 3470).
   - Delete the θ-profile BFGS. The Weibull scaffold becomes the unpenalized block {1, log t} with the domain constraint slope > 0.
   - Project the time smooth's range off span{1, log t}.
   - Do not install null-space shrinkage on {1, log t} (Prop. 5.2).
   - Certificate: the outer has no θ coordinate, so the remaining ρ are certified by §3.2.
9. **Per-cause outer** (`fit.rs:2606` `fit_cause_specific_survival_transformation_custom`, routed from `:3574-3588`).
   - Run C independent LAML problems, one per cause, on the shared design `x_exit` (read-only), each with its own certificate.
   - The fit is certified iff every cause is.
   - Remove the gauge-priority ordering across causes (2744–2760), which exists only because of the joint stacking.
10. **Termination.** Replace `UNBOUNDED_OUTER_ITERATIONS` (`gam-solve/src/rho_optimizer/run.rs:420`, uses 429 and 540) with chart-switching ARC or trust-region iterations that terminate on the face-KKT certificate (Thm 10.1). The iteration count is then bounded by theory, not by a cap. Cross-lane: the compactified-coordinates, boundary-asymptotics and inexact-oracle lanes supply the chart machinery and ε_g.

### 6.3 Latent frailty and interval censoring (cluster F)

11. **σ as an outer domain coordinate** (`survival/latent/survival/mod.rs:1613-1645` `build_log_sigma_blockspec`; `sigma_link.rs:43/126`).
    - Remove ln σ from the inner coefficient vector.
    - Add s = σ² ≥ 0 as an outer coordinate: the domain is [0, ∞), with the μ_s = 1/s chart at infinity if needed.
    - Certificate at s = 0: ∂V/∂s|_0 = −a(θ̂_0) + ½tr(H_0^{−1}∂_sH|_0) > e_s, using the explicit row score ½[(d−H)² − H] and its η-derivatives (§3.6).
12. **Latent θ axes** (`mod.rs:975-1225` `fit_latent_baseline_axes`; `precision_box` at 1001).
    - Delete the `precision_box`: it is a hand box.
    - Delete the θ chart axes whose offset tangent lies in span(time design). This is the direct sum of item 8.
    - Do not install `install_latent_time_nullspace_shrinkage_penalty` (`mod.rs:1675`) on a scaffold-aliased direction. Its stated purpose, making the MAP unique on the interval warm-start, is met instead by the direct sum plus the concavity of Thm 7.1.
13. **Interval, intercept-only** (`quality_vs_lifelines_interval_censored_truth_recovery.rs`). σ is unidentified (the test file's own lines 33–37). The model for `SurvInterval(L,R,event) ~ 1` should carry s = 0 structurally. The likelihood is then concave (Thm 7.1), and the inner problem is a monotone-constrained concave program.
14. **Tests.**
    - `quality_vs_survival_coxph_frailty_hazard_multiplier.rs:232` should fit the group as a shared frailty, i.e. a random-effect block on `g` with λ = 1/s, matching the DGP at 135–145.
    - As written, the test demands recovery of a model the formula cannot represent.

### 6.4 Location-scale survival (cluster L)

15. **Gauge slice** (`location_scale/prepare.rs:504-534`; `time_block.rs:1233+`).
    - Replace the σ-intercept pin with the linear pin "log-t coefficient of h = 1" for all rank reductions, not only rank-2, and free the σ intercept.
    - Certificate: the gauge is fixed by one linear equality, so the outer Hessian has no gauge null direction.
    - Check it by confirming that the Jacobian of (h, η, σ) ↦ z on the slice has full rank [P].
    - That the slice resolves METRIC_OFF is [C].

### 6.5 Derived tolerances used above

| quantity | formula | source |
|---|---|---|
| inner value tolerance | δ* = ε_V/(2p) | Thm 1.3; ε_V from the fp-error lane |
| inner gradient tolerance | δ*_g = ε_g/[2 max_j(‖λ_jS_jθ‖_{H^{−1}} + tr(H^{−1}λ_jS_j))] | Thm 1.4 |
| face/KKT margin e_j | inner gradient bound (Thm 1.4) + fp-error bound of the bracket | §3.2 |
| chart switch ρ → λ | λ_j·\|bracket_j\| < e_j | Cor. 2.2 |
| chart origin | ρ_j⁰ = log(tr(Q_Rᵀ∇²fQ_R)/tr(S_j)) | §3.2 |
| frailty boundary | ∂V/∂s\|_0 > e_s | §3.6 |

No constant in this table is chosen by hand.

---

## 7. Open problems

1. **G ≥ 3 by-factor.** There is no exact ridge (Prop. 4.3), but G strengths are identified by G−1 contrasts, so the outer Hessian is near-singular. The open question is whether the exchangeable single-strength block is always LAML-preferable, or whether per-level strengths should be kept with a certified weakly-identified subspace [C].
2. **Individual frailty against a flexible baseline.** The mixed-PH identification (Elbers-Ridder) is asymptotic. A finite-sample criterion that decides when s is estimable, such as a curvature of V(s) above its error bound, is still needed. The same applies to the σ of the interval model once covariates are present.
3. **Multi-rate corners.** Thm 2.4 covers a common-rate path to λ → ∞. Corners where several λ go to ∞ at different rates need the full compactification (the product of μ charts). Joint analyticity at such corners is proven here only when each λ_k separately covers R [C in general].
4. **Location-scale slice.** A proof that the "slope = 1" slice gives the LAML with the correct improper prior, meaning invariance of the flat prior under the residual gauge, is still missing. Prop. 9.2 shows only that the slices differ.
5. **Attributions.** The timeout logs carry no diagnostics. The attributions of individual timeouts (tram, frailty, competing_risks_truth_recovery, gamlss on_real_data), and of the METRIC_OFF cases (rstpm2 via Cor. 2.2 false certification; CIF via Thm 5.1), should be confirmed by re-running with the outer trace after fixes 2, 3, 8 and 10.
6. **Non-log-concave links.** Thm 7.1 needs a log-concave error law. For a transformation family with a non-log-concave F, such as Student-t errors, interval censoring loses inner concavity, and the inner certificate of Thm 1.3 does not apply.
