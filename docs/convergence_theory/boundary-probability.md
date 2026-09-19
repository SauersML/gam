# Boundary probability: the law of λ̂_REML at and near λ = ∞, and what it implies for the certifier and the smooth tests

Lane: `boundary-probability` (convergence theory team). Scripts and raw outputs are in
`$SP/theory/boundary-probability/` (`bplib.py`, `tables.py→tables.out`, `calibrate.py→calib_*.out`,
`glm.py→glm.out`, `resolution.py→resolution.out`, `localalt.py→localalt.out`,
`scoretest.py→scoretest.out`), where `$SP` is this session's scratchpad. All of it is test or theory
code: it uses grids, finite differences and Monte Carlo, none of which is proposed for production.

Every claim below carries one of three tags:

- **[proven]**: a proof is given here, or a precise reference is cited.
- **[checked]**: verified numerically by the named script.
- **[conjecture]**: stated without proof, with the supporting evidence given.

---

## 1. Summary

1. **Boundary optima are the common case, not an edge case.** Take a penalized spline whose true function lies in the penalty null space. Its REML estimate λ̂ is exactly **+∞** with probability about **0.66–0.68**. This holds for every P-spline order m = 1, 2, 3, for basis sizes k = 5…40, for n = 50…1000, and in the n → ∞ limit **[checked, `tables.out`]**. The value is *not* ½: the Self–Liang/Stram–Lee 50:50 law needs many comparable eigenvalues, and spline eigenvalues decay like s^{−2m} **[proven, §3.2]**. With the default double penalty, *both* penalties of a null smooth sit at λ = ∞ with probability about 0.5, which gives f̂ ≡ 0 **[checked, `calibrate.py`]**.

2. **In γ = 1/λ = e^{−ρ} ≥ 0 the criterion is analytic at the boundary.** It expands as V = V₀ + aγ + ½bγ² + …, and this holds for Gaussian REML and for LAML alike **[proven, §3.3]**. The boundary is optimal iff a ≥ 0. The sign of *a* is the whole decision. In ρ = log λ, both ∂V/∂ρ = −a·e^{−ρ} and the curvature vanish exponentially, and they do so *whatever the sign of a*. So no ρ-gradient tolerance can tell a boundary optimum from a missed interior optimum. The "tail laws", rail stalls and iteration caps are artefacts of the ρ chart.

3. **gamfit's face certificate refuses almost every legitimate boundary optimum.** The certificate is `C ≻ 0` (`rail_face.rs:376`). For a one-penalty face under H0, P(C ≻ 0) = P(Beta(K/2, (n−p−K)/2) < 1/(n−p)) ≈ P(χ²_K < 1) **[proven, §3.4]**. With a cubic P-spline and k = 10 this is **1.5×10⁻³**. With k = 20 it is **~10⁻⁹**. The true boundary probability is 0.67 **[checked]**. The necessary and sufficient first-order condition is the KKT condition c_j ≥ 0: ½tr(S⁻¹C) ≥ 0 on each released block, i.e. tr(M(t)C) ≥ 0 over the face simplex. This is exactly the Crainiceanu–Ruppert condition. C ≻ 0 is sufficient but nowhere near necessary.

4. **The lower rail (λ_j → 0, ρ_j → −∞) is never optimal when the penalty's own directions are identified.** In that case V ≈ −½(r_j − h_j)ρ_j → +∞ **[proven, §3.5]**. A lower-rail optimum exists only when r_j = h_j, for example when range(S_j) ⊆ range(S_{−j}), as with a Matérn full-rank mass penalty. Its KKT condition is then taken in λ_j ≥ 0, not in ρ_j. The module doc at `asymptote_certificate.rs:8–10`, which says "null-space shrinkage wants λ → 0", has the sign of the theory wrong for identified null spaces.

5. **The criterion has a statistical resolution, and the current tolerance ignores it.** V is a negative log-likelihood defined up to an additive constant: under y → cy it shifts by (n−p)·log c. The Fisher information for ρ is **O(edf), not O(n)**, e.g. I_ρ ≈ 1.0 at edf = 3 and ≈ 2.2 at edf = 6 for every n from 100 to 6400 **[checked, `resolution.out`]**. A criterion gap τ moves every linear functional of the hyperparameters by at most √(2τ) sampling standard deviations **[proven, §3.6]**. The current rule `rel·(1+|V|)` (`run.rs:8839`, `decrement_bands.rs:131`) is not invariant to the units of y: rescaling y by 10⁶ multiplies the tolerance by 10 **[checked]**. It also scales wrongly in n.

6. **Proposed derived tolerance: τ_stat = 1/(2n) in absolute log-likelihood units.** Equivalently, the fit must be within n^{−1/2} sampling SDs of the exact optimum. This is the accuracy order of the Laplace approximation and of first-order inference, so resolving the criterion further buys nothing. Certify when ½·δ² + band ≤ τ_stat, where δ² is the Newton decrement (taken in γ coordinates on faces, with projection) and band is the numerical band. If the numerical band alone exceeds τ_stat, the fit fails with a typed error: numerically unresolvable at statistical precision. Applied to the failing multinomial fit (N = 200, ½δ² = 4.7×10⁻⁵ against the current resolution 2.7×10⁻⁵), this would certify it.

7. **gamfit's smooth-term p-values are conservative under the default double penalty**, and the brief counts conservative p-values as a bug. The Wood-type Wald statistic has an atom at 0 of mass ≈ 0.5, because f̂ ≡ 0 when both λ̂ = ∞. Its χ²/F(ref_df) reference ignores that atom. Simulated size under H0 is about half of nominal (§4.3). The equivalent chi-bar ½χ²₀ + ½χ²₁ reference for the RLRT is also conservative for splines by a factor of about 2 (size 0.025 at nominal 0.05) **[checked, `tables.out`]**.

8. **Calibrated replacement: the test statistic is the boundary certificate's own first-order quantity.** The variance-component score U = y'P₀ZZ'P₀y / y'P₀y has an *exact* Gaussian null law, a ratio of quadratic forms, computed deterministically by Imhof inversion. For GLMs it has the first-order null law Σκ_s χ²₁ with κ = eig(Z'P_WZ) (Lin 1997). The Gaussian version is exactly calibrated **[proven]** and the simulations match nominal (§4.4). Other options: the pivotal-T exact null (simulation) and the RLRT with the Crainiceanu–Ruppert exact null.

9. **GLM/LAML analogue.** V'(0) = −½[r'ZZ'r − tr(Z'P_WZ)] + ½tr((X₀'WX₀)⁻¹X₀' diag(w'·η̇) X₀). The first term is Lin's score; the second is the Laplace weight-drift term, which is a relative O(n^{−1/2}) correction. The formula matches finite differences to a median relative error of 3×10⁻⁵. The GLM boundary probability matches the κ-spectral prediction: 0.6745 vs 0.677 at n = 100 **[checked, `glm.out`]**.

---

## 2. Setup and notation

**Model.** Consider one penalized block of a GAM, written in mixed-model form. The penalty is S = U₊D U₊' (plus null space U₀).

- Unpenalized design: X₀ = [parametric | X_f U₀], with p = rank X₀.
- Whitened penalized design: Z = X_f U₊ D^{−1/2} (n × K), so the block's coefficients are b and the penalty is λ‖b‖².
- γ = 1/λ = e^{−ρ} ≥ 0 is the variance ratio. γ = 0 is the boundary λ = ∞.
- For the Gaussian case, y = X₀β + Zb + ε with b ~ N(0, σ²γI) and ε ~ N(0, σ²I).
- Contrast basis: A (n × (n−p)) with A'X₀ = 0 and A'A = I. P₀ = AA'.
- Spectrum: A'ZZ'A = Σ_s μ_s e_s e_s', with μ₁ ≥ … ≥ μ_K > 0 and μ_s = 0 for s > K. The μ_s are the nonzero eigenvalues of Z'P₀Z.
- Spectral data: w_s = e_s'A'y/σ. Under H0 (γ = 0), w ~ N(0, I_{n−p}).

**Profiled REML** (Crainiceanu & Ruppert 2004, eq. (5)), as a function to *maximize*:

  ℓ_R(γ) = −((n−p)/2)·log( Σ_{s≤K} w_s²/(1+γμ_s) + Σ_{s>K} w_s² ) − ½ Σ_{s≤K} log(1+γμ_s).  (2.1)

Other definitions:

- **Criterion.** gamfit minimizes V = −ℓ_R + const. With a known scale, drop the log and use −½Σw_s² γμ_s/(1+γμ_s) + ½Σlog(1+γμ_s) for V.
- **Several penalties.** S_λ = Σ_j λ_j S_j. A *face* F is a set of coordinates with λ_j = ∞, i.e. γ_j = 0. The released subspace is range(Σ_{j∈F} S_j) restricted to ⋂_{j∉F} …; the details are in `rail_face_limit.rs:1–40`.
- **C.** The first-order form of V on a face in gamfit's convention (`rail_face.rs`). In the notation of `rail_face_limit.rs:25–27` it is the profiled-Gaussian Schur form C = Schur_Z(K) − Schur_Z(S_R) − g_Q g_Q'/φ̂, plus the rank-2 drift g_Q d_Q' + d_Q g_Q' for LAML.
- **c_j** = ½tr((Q_j'S_jQ_j)⁻¹Q_j'CQ_j) (`rail_face.rs:385–436`). On the tail, ∂V/∂ρ_j → −c_j e^{−ρ_j}.
- **GLM.** η = X₀β + √γ·Zv with v ~ N(0, I); W = diag(w(μ)); P_W = W − WX₀(X₀'WX₀)⁻¹X₀'W; κ_s = eig(Z'P_WZ).

**Boundary probabilities.** P_loc is the probability that the first-order KKT condition holds at γ = 0, i.e. that γ = 0 is a local optimum. P_glob = P(λ̂ = ∞) is the probability that γ = 0 is the global optimum. Always P_glob ≤ P_loc.

---

## 3. Results and derivations

### 3.1 Exact finite-sample law of the boundary event (Gaussian)

**Proposition 1 [proven].** Under H0, γ = 0 is a local maximizer of (2.1) iff

  Σ_{s≤K} μ_s w_s² ≤ (Σμ_s/(n−p)) · Σ_{s≤n−p} w_s².  (3.1)

Hence P_loc = P( Σ_{s≤K}(μ_s − c)·χ²₁,s − c·χ²_{n−p−K} ≤ 0 ), with c = Σμ/(n−p). This is a linear combination of independent χ² variables and is computed exactly by Imhof (1961) inversion (`bplib.p_local_boundary`).

With a known scale, the condition is Σμ_s(w_s² − 1) ≤ 0.

*Proof.* Differentiate (2.1) at γ = 0:

  ℓ_R'(0) = ((n−p)/2)·Σμ_s w_s² / Σw_s² − ½Σμ_s.

- The domain is γ ≥ 0. The boundary is a local maximum iff ℓ_R'(0) < 0, and ℓ_R'(0) = 0 is a null event.
- ℓ_R is analytic at 0 (§3.3), so the one-sided first-order condition is decisive whenever ℓ_R'(0) ≠ 0.
- Rearranging ℓ_R'(0) ≤ 0 gives (3.1).
- Since w ~ N(0, I), the difference of the two sides of (3.1) is the stated χ² combination. ∎

The same condition appears as the "probability mass at zero" in Crainiceanu & Ruppert (2004, §3, their eq. for P(λ̂ = ∞) via the first-order condition). They also note that P_glob requires a supremum check.

**Local vs global [checked].** The Monte Carlo P_glob in `tables.out` uses an exact spectral grid; this is test code. P_glob falls short of P_loc by 0.004–0.012. In those cases γ = 0 is a local optimum, but a separate interior maximum beats it. So the likelihood in γ is not unimodal, and a certifier that verifies only first-order KKT at a face can certify a local, non-global optimum about 1% of the time under H0. §7 gives the open problem.

### 3.2 Large-n limit: why splines are not 50:50

**Proposition 2 [proven, given convergence of μ_s/n].** Let k be fixed and n → ∞. Assume μ_s/n → κ_s, which holds for regression splines under a continuous design density, by convergence of the empirical Gram matrix. Then φ̂-ratios converge, Σw²/(n−p) → 1 a.s., c/n → Σκ, and

  P_loc → π_∞(κ) := P( Σ_s κ_s(χ²₁,s − 1) ≤ 0 ).  (3.2)

*Proof.* Divide (3.1) by n and apply Slutsky's theorem. The limit law is continuous at 0. ∎

**Why this is not ½.** The Self & Liang (1987) and Stram & Lee (1994) 50:50 law is the case where the normalized score Σκ_s(χ²₁ − 1)/√(2Σκ_s²) is asymptotically N(0, 1). By Lindeberg, that holds iff max_s κ_s² / Σκ_s² → 0.

- For a random intercept with J → ∞ clusters, κ_s ≈ equal, the condition holds, and π_∞ → ½.
- For a P-spline of order m, κ_s ≍ s^{−2m} (Speckman 1985; Utreras 1983 for smoothing-spline eigenvalues). So κ₁²/Σκ² stays bounded away from 0 even as k → ∞. There is **no LAN in the direction of γ**, the score is not asymptotically normal, and π_∞ ≠ ½.
- In the extreme case of one dominant eigenvalue, π_∞ = P(χ²₁ ≤ 1) = 0.6827.

Numerical values [checked, `tables.out`, limits computed exactly by Imhof]:

| m | k=5 | k=10 | k=20 | k=40 | Σκ²/(Σκ)² (k=20) |
|---|---|---|---|---|---|
| 1 | 0.6793 | 0.6646 | 0.6592 | 0.6579 | 0.478 |
| 2 | 0.6825 | 0.6777 | 0.6765 | 0.6762 | 0.727 |
| 3 | 0.6796 | 0.6806 | 0.6803 | 0.6802 | 0.821 |

Finite n, with n = 50 → 1000:

- **P-splines:** P_loc(REML) is 0.654–0.682 and P_glob is 0.648–0.681. The values increase slowly towards π_∞, and the difference between the REML and known-scale values is O(1/n).
- **Other smooths:** truncated-power linear splines (the Crainiceanu–Ruppert example) give P_loc = 0.671–0.676, and a uniform random design gives the same values as an equispaced one.
- **Random intercept:** J = 5 gives 0.57–0.59, J = 20 gives 0.53–0.54, and J = 100 gives 0.513–0.518. This is the Stram–Lee regime converging to ½, as the theory predicts.

**Conjecture 1 [conjecture; checked on 400 random κ vectors in `scoretest.out`].** For any κ with at least one positive entry, ½ < π_∞(κ) ≤ P(χ²₁ ≤ 1) = 0.6827.

- The lower bound says the median of a mean-zero, positive-weighted, centred χ² sum is negative, i.e. the sum is right-skewed.
- The upper bound says the single-atom case is extremal.
- Two equal weights give P(χ²₂ ≤ 2) = 1 − e^{−1} = 0.632.

**Local alternatives [checked, `localalt.out`, m = 2, k = 20].**

*Fixed-function truth.* Take w₁ = δ₁ + ε₁, i.e. signal of size δ₁σ on the leading eigenvector, which is a function with RMS δ₁σ/√n. Condition (3.1) becomes a noncentral χ² combination.

| δ₁ | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| P(λ̂ = ∞), n = 100 | .674 | .618 | .470 | .300 | .156 | .023 | .001 |
| P(λ̂ = ∞), n = 1000 | .674 | .620 | .473 | .302 | .156 | .023 | .001 |

The table is n-invariant. So the natural local alternatives are ‖f‖_RMS = O(σn^{−1/2}), which is the parametric rate, because the leading spline direction behaves like a single fixed effect. **A real but small signal still gives λ̂ = ∞ with substantial probability:** 47% at δ₁ = 1 and 16% at δ₁ = 2. Boundary optima are therefore legitimate fits under the alternative too, not merely under H0.

*Random-effect truth.* Take w_s ~ N(0, 1 + γ₀μ_s). As γ₀μ₁ runs over 0, .1, .3, 1, 3, 10, P(λ̂ = ∞) is .674, .652, .611, .510, .369, .209. Even when the between-component variance exceeds the noise variance tenfold along the leading direction, one fit in five lands on the boundary.

### 3.3 Analyticity in γ; the ρ-flatness is an artefact of the chart

**Proposition 3 [proven].**

(a) *Gaussian REML.* ℓ_R(γ) in (2.1) is real-analytic on γ > −1/μ₁. In particular it is analytic at γ = 0.

(b) *LAML.* Assume the inner problem min_{β,v} −ℓ(X₀β + sZv) + ½‖v‖² is strictly convex, with an analytic log-likelihood and canonical link (or any link for which the penalized Hessian is positive definite at the mode). Then V is real-analytic in γ at γ = 0.

*Proof of (b).* Put s = √γ.

- The inner objective is invariant under (s, v) → (−s, −v). By uniqueness of the mode, v̂(−s) = −v̂(s) and β̂(−s) = β̂(s), so η̂, W and μ̂ are even in s.
- The Laplace term is ½log det H(s) with H(s) = [[X₀'WX₀, sX₀'WZ], [sZ'WX₀, I + s²Z'WZ]], and det H = |X₀'WX₀|·|I + s²Z'P_WZ|. This is even in s.
- By the analytic implicit function theorem, the mode is analytic in s at s = 0, where H(0) is positive definite. Hence V(s) is analytic and even, so it is an analytic function of s² = γ. ∎

**Consequences.** Write V(γ) = V₀ + aγ + ½bγ² + O(γ³).

1. γ = 0 is a local minimum of V on γ ≥ 0 iff a > 0 (with a = 0 a null event). If a < 0 and b > 0, the interior stationary point is γ* ≈ −a/b, and the gap is V₀ − V(γ*) ≈ a²/(2b). For REML with V = −ℓ_R, RLRT = 2(V₀ − V(γ̂)). So **the gap equals RLRT/2**, and a near-boundary interior optimum is statistically meaningful only if a²/(2b) is large compared with the resolution of §3.6.

2. In ρ = −log γ: V = V₀ + a e^{−ρ} + ½b e^{−2ρ} + …, so

  ∂V/∂ρ = −a e^{−ρ} + O(e^{−2ρ}),  ∂²V/∂ρ² = a e^{−ρ} + O(e^{−2ρ}).

   Both vanish exponentially **for either sign of a**. Now take a quasi-Newton step in ρ against a gradient tolerance ε.
   - It reaches |∂V/∂ρ| < ε at ρ ≈ log(|a|/ε), whether that point is the boundary optimum (a > 0) or sits on a descending slope beyond which an interior optimum was skipped (a < 0 is impossible on the tail itself, but the rail can be reached from the wrong basin).
   - The ρ-Hessian is also ≈ 0 there, so Newton decrements, BFGS curvature pairs and the "indefinite/flat" diagnostics all degenerate.

   This is the mechanism behind the #2299-type rail stalls, the exponential "tail laws" (`asymptote_certificate.rs`) and the iteration-cap failures. They are properties of the chart, not of the criterion.

3. **The correct object is the KKT condition in γ.** Use V_γ(0) = a ≥ 0, with a computed *at* γ = 0 by the limit fit (which `rail_face_limit.rs` already builds: a = c_j), and V_γγ(0) = b for the second-order gap. For interior points, the chain rule gives

  V_γ = −e^{ρ}V_ρ,  V_γγ = e^{2ρ}(V_ρρ + V_ρ).  (3.3)

### 3.4 Faces with several penalties; the correct certificate and what C ≻ 0 costs

**Setting.** Let the face F have coordinates j ∈ F released together along γ_j = εt_j, with t in the simplex Δ_F. The penalty on the released subspace is (1/ε)·Σ_j t_j^{−1}A_j, where A_j is S_j compressed onto the released subspace.

**Proposition 4 [proven for A(t) := Σt_j^{−1}A_j nonsingular on the released subspace].** The one-sided directional derivative of V along t is

  D_tV = ½ tr( M(t)·C ),  M(t) = (Σ_j t_j^{−1}A_j)^{−1},

which is the parallel sum of the t_jA_j^{−1}. M is homogeneous of degree 1 in t, and it is matrix-concave in t (Anderson & Duffin 1969). The face is a first-order local optimum iff

  min_{t∈Δ_F} tr(M(t)C) ≥ 0.  (3.4)

*Proof.*

- Expand the criterion at the limit fit to first order in the released penalty inverse. This is the same expansion `rail_face_limit.rs` uses for a single ray: V = V_∞ + ½tr(S_F^{−1}C) + o(‖S_F^{−1}‖).
- Substitute S_F = ε^{−1}A(t). Homogeneity follows because M(τt) = τM(t).
- The face is optimal iff no ray into the orthant decreases V, and the rays are parameterized by Δ_F. ∎

**Corollaries [proven].**

- *Disjoint ranges.* If the A_j have mutually orthogonal ranges (true for double-penalty blocks, whose bending and null-space ranges are orthogonal), then M(t) = ⊕ t_jA_j^+. This gives tr(M(t)C) = 2Σ_j t_jc_j, so (3.4) ⟺ **c_j ≥ 0 for all j ∈ F**. The per-coordinate constants that gamfit already computes (`rail_face.rs:385–436`) are the certificate.
- *Sufficiency of C ≻ 0.* C ≻ 0 implies every tr(M(t)C) > 0, so it is sufficient. It is not necessary.
- *What C ≻ 0 costs (one penalty, S_R = 0, Gaussian, H0).* In the whitened spectral coordinates, g_Q = Z'P₀y with components √μ_s·w_sσ, and φ̂ = σ²Σ_{all}w²/(n−p).
  - C ≻ 0 ⟺ g'(Z'P₀Z)^{−1}g < φ̂ ⟺ Σ_{s≤K}w_s² < Σ_{s≤n−p}w_s²/(n−p). This is equivalent to a Beta(K/2, (n−p−K)/2) variable being below 1/(n−p) (`bplib.p_face_certificate_C_pd`).
  - By contrast c = ½tr(C) ≥ 0 ⟺ Σμ_sw_s² ≤ Σμ_s·φ̂/σ², which is exactly (3.1).

Probability that the current certificate accepts, against the true boundary probability [proven formula; values in `tables.out`]:

| design (m = 2, n = 100) | K | P(C ≻ 0) | P_loc (truth) |
|---|---|---|---|
| k = 5 | 3 | 0.19 | 0.680 |
| k = 10 | 8 | 1.5×10⁻³ | 0.675 |
| k = 20 | 18 | 1.4×10⁻⁹ | 0.674 |
| k = 40 | 38 | 1.3×10⁻²⁵ | 0.674 |

For the random intercept with J = 100 clusters, P(C ≻ 0) ≈ 10⁻⁸¹. So for any realistic basis, the C ≻ 0 gate refuses *every* boundary optimum it meets. The optimizer is then left in ρ-space, chasing the exponentially flat tail of §3.3, until a stall detector, a cap or a tolerance intervenes. This follows the pattern of the rail/stall test clusters in §6.

*Degenerate c_j ≈ 0.* By §3.3, a boundary coordinate with c_j < 0 but small has an interior optimum only c_j²/(2b_j) below the boundary value. It is statistically indistinguishable from the boundary if c_j²/(2b_j) ≤ τ_stat (§3.6). A certificate that accepts it at the boundary, reporting the gap, is correct to within τ_stat.

### 3.5 The lower rail (λ_j → 0)

**Proposition 5 [proven].** Write V = −ℓ(β̂) + ½β̂'S_λβ̂ + ½log|H_λ| − ½log|S_λ|₊ + const, with H_λ = X'WX + S_λ. This covers REML (Gaussian) and LAML. For coordinate j, define:

- r_j = dim of the directions penalized *only* by S_j, i.e. range(S_j) ∩ null(S_{−j}), compressed;
- h_j = the number of those directions that the likelihood does not identify, i.e. the null directions of X'WX + S_{−j} within them.

As λ_j → 0 (ρ_j → −∞), with the other coordinates fixed:

  V = −½(r_j − h_j)·ρ_j + O(1).

*Proof.*

- The pseudo-determinant satisfies log|S_λ|₊ = r_j log λ_j + O(1), because the rank of S_λ is constant for λ_j > 0 and exactly r_j eigenvalues scale with λ_j.
- log|H_λ| = h_j log λ_j + O(1), because only unidentified directions keep an O(λ_j) eigenvalue.
- The fit terms converge to the λ_j = 0 fit.
- Collect the terms: ½h_jρ_j − ½r_jρ_j. ∎

**Consequences.**

- If r_j > h_j, V → +∞ on the lower rail. So a lower rail is **never** an optimum for directions that the data identify. This covers every double-penalty null-space block whose null space (linear trend, etc.) is estimable, which is the usual case.
- The module doc `asymptote_certificate.rs:8–10` says that "a null-space shrinkage coordinate wants λ → 0 … so it does not shrink a real signal". The criterion itself forbids that limit. With a strong null-space signal, λ̂_j goes to a finite, *small but interior* value, and the −½r_jρ_j term keeps it off the rail.
- A lower-rail optimum requires r_j = h_j. The typical case is r_j = 0: range(S_j) ⊆ range(S_{−j}), as for a full-rank Matérn/mass penalty paired with a derivative penalty. V is then bounded as λ_j → 0, the boundary λ_j = 0 is a genuine *domain* boundary (λ_j ≥ 0), and the KKT condition is ∂V/∂λ_j ≥ 0 at λ_j = 0.
- By the same chain rule as (3.3), ∂V/∂ρ_j = λ_j∂V/∂λ_j → 0 exponentially. The same chart artefact therefore appears on the lower side, and the remedy is the same: use the coordinate λ_j ≥ 0 there.

### 3.6 Statistical resolution of the criterion; a derived tolerance

**Invariance.** Under y → cy (Gaussian), V shifts by (n−p)·log|c|, and every λ̂, edf, fitted curve divided by c, and p-value is unchanged. So **the only invariant tolerances are on differences of V**, in absolute log-likelihood units. A tolerance of the form rel·(1+|V|) is not invariant [proven].

`resolution.out` gives the numbers for rel = 10⁻⁷ at n = 1000:

| y scaled by | 10⁻³ | 1 | 10³ | 10⁶ |
|---|---|---|---|---|
| \|V\| | 5472 | 1415 | 8302 | 15189 |
| τ_rel | 5.5×10⁻⁴ | 1.4×10⁻⁴ | 8.3×10⁻⁴ | 1.5×10⁻³ |

The same fit is certified or refused depending on the units of y.

**Fisher information in ρ [proven].** For a single block, Gaussian, profiled scale, the expected information is

  I_ρ = ½[ Σ_s t_s² − (Σ_s t_s)²/(n−p) ],  t_s = γμ_s/(1+γμ_s).

Here t_s ∈ (0, 1) is the per-eigendirection edf and Σt_s = edf of the block. So I_ρ ≤ ½·edf, whatever n is. The sampling SD of ρ̂ is ≥ (edf/2)^{−1/2}, which is **of order one e-fold** for typical smooths.

Checked in `resolution.out` (m = 2, k = 20, n = 100…6400):

| edf | I_ρ | 1/√I_ρ | MC sd(ρ̂) | robust sd | P(railed) |
|---|---|---|---|---|---|
| 3 | 0.99–1.03 | 0.98–1.01 | 1.29–1.37 | 1.02–1.14 | 8% |
| 6 | 2.08–2.26 | 0.67–0.69 | 0.86–0.94 | 0.71–0.75 | 0.3% |

The information does not grow with n, and a real signal of edf = 3 still produces a railed λ̂ 8% of the time.

**Proposition 6 [proven].** Let the criterion near θ̂ = (ρ̂ or γ̂) be quadratic with Hessian H ≻ 0. Since V is a (restricted or Laplace) negative log-likelihood, H ≈ observed information and Var(θ̂) ≈ H^{−1}. Then for any θ with V(θ) − V(θ̂) ≤ τ and any vector L,

  |L'(θ − θ̂)| ≤ √(2τ)·√(L'H^{−1}L) = √(2τ)·se(L'θ̂).

*Proof.* Cauchy–Schwarz in the H-inner product, together with ½(θ−θ̂)'H(θ−θ̂) ≤ τ. ∎

The same bound transfers, through the delta method, to any smooth functional of θ such as edf, fitted values or AIC. So a gap τ means the reported hyperparameters sit within η = √(2τ) sampling SDs of the exact optimum, simultaneously in every direction.

**Choosing η: why τ_stat = 1/(2n).** This is a principled choice, not a theorem; §7 lists it as an open problem. The criterion and everything computed from it are first-order objects:

- LAML is the Laplace approximation to the log marginal likelihood. Its error is O(n^{−1}) absolutely at fixed dimension (Tierney & Kadane 1986), and Shun & McCullagh (1995) show when it degrades with growing dimension.
- Wald/edf inference built on λ̂ has O(n^{−1/2}) error.

Setting η = n^{−1/2} makes the optimization error in every hyperparameter functional smaller than the intrinsic first-order error of the inference built on it, uniformly in the units of y and in the parameterization. Hence

  **τ_stat = η²/2 = 1/(2n)**, with n the number of observations. For grouped or weighted data, use the sum of prior weights, i.e. the Fisher-information count.

**Certificate.** At a candidate point, let δ² be the projected Newton decrement:

- in ρ for interior coordinates;
- in γ, with the constraint γ ≥ 0, for face coordinates, using a = c_j and b_j from the limit fit.

The quadratic model promises that no feasible point improves V by more than G_model = ½δ². Then:

  certify ⟺ G_model + band_V ≤ τ_stat,  (3.5)

- band_V is the numerical objective band, i.e. what `decrement_bands.rs` already computes.
- If band_V > τ_stat, the numerical error alone exceeds the statistical resolution, and the fit must fail with a **typed** "numerically unresolvable at statistical precision" error. This is not a retry.
- If band_V ≤ τ_stat but G_model > τ_stat − band_V, the optimizer has more to do.

(3.5) replaces two things: the unit-dependent `rel·(1+|V|)` and the uniform ρ-gradient tolerance. It implies a gradient tolerance ‖g‖_{H⁻¹} ≤ √(2(τ_stat − band_V)), which is derived and not chosen.

**Error on reported p-values [proven].** If the test statistic is T = 2·(criterion gap), as for the RLRT, an optimization error τ changes T by at most 2τ. The p-value then changes by at most f_T(t)·2τ ≤ (sup f_T)/n. For a score statistic evaluated at the certified boundary, there is no optimization error at all.

### 3.7 The exact null for smooth-term tests with a boundary λ̂

**The atom [proven, checked].** With the default double penalty (both the null space and the range penalized), H0: f ≡ 0 puts both γ_j at 0 with probability ≈ 0.5. This is 0.505 at n = 100 in the smoke test; the final figure is in §4.3. At such a fit f̂ ≡ 0 exactly and any Wald statistic T = 0. The null law of T is therefore

  L_H0(T) = π₀·δ₀ + (1 − π₀)·L_H0(T | T > 0),  π₀ = P_H0(all γ̂_j = 0).

**Correct p-value.** The correct p-value is p(t) = P_H0(T ≥ t), where p(0) = 1. No χ² or F reference with continuous d.f. has this form. gamfit's present reference (`smooth_test.rs:180–199`) is χ²_{ref_df} or F(ref_df, n − edf), with rank = round(edf) floored at 1 and ref_df ≥ rank_used. It treats T as a continuous χ²-like quantity with at least one d.f., and so it is conservative whenever π₀ is large. The simulated size is roughly half of nominal (§4.3).

**Pivotality (Gaussian, one tested smooth, parametric X₀) [proven].** Under y → ay + X₀c:

- λ̂ is invariant, because REML uses only the contrasts A'y and a scale-free profile.
- β̂_smooth scales by a, and φ̂ and Vb scale by a².

So T = β̂'Vb^{−1}β̂ (truncated or otherwise) is invariant. Under H0, y = X₀β + σε, so T is a function of ε alone: **pivotal**. Its exact null can be generated from ε ~ N(0, I) by refitting the spectral form (2.1) per draw. Each refit is a one- or two-dimensional analytic γ ≥ 0 problem solved by the production projected-Newton step of §6, not a grid. The p-value is then (1 + #{T_b ≥ t})/(B + 1), which is exactly valid.

**Score test (recommended; exact and deterministic) [proven].** The certificate quantity at the boundary is the variance-component score. In the Gaussian profiled case,

  U = y'P₀ZZ'P₀y / y'P₀y,  P_H0(U ≥ u) = P( Σ_{s≤K}(μ_s − u)·χ²₁,s − u·χ²_{n−p−K} ≥ 0 ),  (3.6)

which is exact, evaluated by Imhof or Davies inversion (a deterministic 1-D integral), with no simulation, refit or atom. This is Lin's (1997) score test in its exact finite-sample Gaussian form. The distribution is exact because U is a ratio of quadratic forms in the same normal vector.

- **Other terms with unknown λ.** Use the Greven et al. (2008) pseudo-likelihood argument: condition on the other λ̂_{−j}, i.e. replace I by the fitted marginal covariance of the other components, and apply (3.6) to the transformed data. This is asymptotically exact.
- **Double penalty.** The H0 f ≡ 0 involves both blocks. The joint score with the unit-weight total penalty, Z = [Z_bend, Z_null], whitened by S_bend + S_null, gives a single ratio (3.6) with the combined spectrum. The choice of weights determines the direction of power but not the calibration.

**RLRT alternative.** RLRT = 2 sup_γ[ℓ_R(γ) − ℓ_R(0)] has the exact spectral null of Crainiceanu & Ruppert (2004, Theorem 1): replace w by N(0, I) in (2.1) and take the supremum. Its atom at 0 has mass exactly P_glob (≈ 0.66). The chi-bar ½χ²₀ + ½χ²₁ approximation has size .025 at nominal .05 for splines (§4.2), so it is conservative by a factor of about 2. Scheipl, Greven & Küchenhoff (2008) recommend the simulated exact null (their RLRTSim) over chi-bar.

### 3.8 GLM / LAML analogues

**Proposition 7 [proven; checked].** For η = X₀β + √γZv (Bernoulli, logit), with the LAML criterion

  V(γ) = −ℓ(β̂, v̂) + ½v̂'v̂ + ½log|X₀'WX₀| + ½log|I + γZ'P_WZ|,

the right derivative at γ = 0 is

  V'(0) = −½[ r'ZZ'r − tr(Z'P_WZ) ] + ½ tr( (X₀'WX₀)^{−1} X₀' diag(w'⊙η̇) X₀ ),  (3.7)

where:

- r = y − μ̂₀ is the null-fit residual;
- w' = dw/dη = w(1 − 2μ) for the logit link;
- η̇ = dη̂/dγ|₀ = (I − X₀(X₀'WX₀)^{−1}X₀'W)·ZZ'r.

*Proof.* By §3.3, V is analytic in γ, so V'(0) exists. Write û = √γ·v̂.

- The inner mode satisfies û = γZ'(y − μ̂) + O(γ²), so η̂ = η̂₀ + γ·η̇ + O(γ²), with η̇ as stated after projecting out the X₀-refit.
- −ℓ + ½v̂'v̂ = −ℓ₀ − ½γr'ZZ'r + O(γ²), by the envelope theorem.
- ½log|I + γZ'P_WZ| = ½γtr(Z'P_WZ) + O(γ²).
- ½log|X₀'WX₀| changes through W only: ½tr((X₀'WX₀)^{−1}X₀' diag(w'⊙η̇) X₀)·γ. ∎

The first term is Lin's (1997) score for the variance component. The second is the Laplace weight-drift term, the scalar analogue of gamfit's rank-2 `g_Qd_Q' + d_Qg_Q'` correction (`rail_face_limit.rs:29–33`). The drift term is O_p(1), while the score term is O_p(n^{1/2}) for spline κ, so it is a relative O(n^{−1/2}) correction.

**Boundary probability [checked, `glm.out`].** Binomial logit, n = 100, k = 10, 2000 simulations, with η_true linear so that H0 holds for the penalized part:

| quantity | value |
|---|---|
| P(boundary), exact local LAML | 0.6745 |
| P(boundary), Lin score only | 0.6775 |
| P(boundary), Lin + drift | 0.6745 |
| P(boundary), global (grid) | 0.6595 |
| κ-spectral prediction π_∞(κ) | 0.6773 (MC s.e. 0.0105) |
| \|V'(0) − (3.7)\| / E\|Lin\| | median 2.9×10⁻⁵, max 9×10⁻⁴ |
| \|drift\| / E\|Lin\| | median 4.6×10⁻² |
| sign disagreement, Lin vs exact | 2.0% |

The n = 400 run is in `glm.out` if it completed; see §4.5.

**GLM null law for tests.** r'ZZ'r → Σκ_sχ²₁ with κ = eig(Z'P_WZ) at the null fit (Lin 1997; Zhang & Lin 2003). The p-value P(Σκ_sχ²₁ ≥ r'ZZ'r) is deterministic by Imhof and first-order accurate. §4.4 checks its size.

---

## 4. Numerical checks

All scripts use `$SP/theory/venv`. Seeds are fixed in the scripts.

### 4.1 Boundary probabilities (`tables.py → tables.out`, 40 000 spectral draws per row)

These are summarized in §3.2 and §3.4.

- The exact Imhof P_loc agrees with the Monte Carlo local frequency to MC error in every row, e.g. 0.6753 vs 0.6783 at m = 2, k = 10, n = 100, where the s.e. is 0.0023.
- P_glob is 0.4–1.2 points below P_loc everywhere.

### 4.2 RLRT null quantiles vs chi-bar (`tables.out`)

For P-splines the RLRT 0.95 quantile is 1.55–2.00, against 2.71 for ½χ²₀ + ½χ²₁. The actual size of the chi-bar test at nominal .10 / .05 / .01 is:

| model | size at .10 / .05 / .01 |
|---|---|
| m = 1 | .053–.064 / .025–.031 / .004–.006 |
| m = 2 | .047–.058 / .022–.028 / .004–.006 |
| m = 3 | .049–.057 / .023–.027 / .004–.005 |
| random intercept J = 100 | .091–.094 / .045–.047 / .009 |

The random-intercept row approaches nominal, as Stram–Lee predicts.

### 4.3 Calibration of gamfit's smooth-term p-value (`calibrate.py → calib_*.out`)

`calibrate.py` replicates the `wood_smooth_test` path (`smooth_test.rs:119–209`):

- the fitted covariance H⁻¹φ̂, as in `result_types.rs:5543`;
- whitening by the intercept-projected Gram (`smooth_test.rs:273–311`);
- rank = round(edf), clamped to [max(null_dim, 1), dim] (`smooth_test.rs:164–167`);
- the truncated quadratic (`smooth_test.rs:320–350`);
- ref_df = max(tr(F)²/tr(F²), rank_used) (`smooth_test.rs:180–183`, `:352`);
- F(ref_df, n − tr F) (`smooth_test.rs:187–199`).

REML is maximized exactly over γ ≥ 0 using test-code L-BFGS-B, with the boundary included. The "exact-mix" p-value is the empirical P_H0(T ≥ t) from independent reference draws. The model is Gaussian with a cubic B-spline, k = 10, and f ≡ 0.

CALIB_RESULTS_PLACEHOLDER

### 4.4 Score-test calibration (`scoretest.py → scoretest.out`)

SCORE_RESULTS_PLACEHOLDER

### 4.5 LAML derivative and GLM boundary law (`glm.py → glm.out`)

GLM_RESULTS_PLACEHOLDER

### 4.6 Resolution and local alternatives

These are reported inline in §3.6 (`resolution.out`) and §3.2 (`localalt.out`).

---

## 5. Literature

- **Crainiceanu, C. M. & Ruppert, D. (2004).** Likelihood ratio tests in linear mixed models with one variance component. *JRSS-B* 66(1), 165–185. Supplies the spectral form (2.1), the probability of λ̂ = ∞, the exact finite-sample null of the (R)LRT, and the finding that 0.5:0.5 fails for splines. That paper reports P(λ̂ = ∞) ≈ 0.65–0.70 for penalized splines, which matches §3.2.
- **Crainiceanu, C. M., Ruppert, D., Claeskens, G. & Wand, M. P. (2005).** Exact likelihood ratio tests for penalised splines. *Biometrika* 92(1), 91–103. Exact RLRT for a zero-variance smooth, and local power.
- **Self, S. G. & Liang, K.-Y. (1987).** Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA* 82(398), 605–610. The chi-bar mixtures, which need i.i.d. or LAN structure.
- **Stram, D. O. & Lee, J. W. (1994).** Variance components testing in the longitudinal mixed effects model. *Biometrics* 50(4), 1171–1177. The ½χ²₀ + ½χ²₁ law for a random intercept with J → ∞.
- **Chernoff, H. (1954).** On the distribution of the likelihood ratio. *Ann. Math. Statist.* 25(3), 573–578. The boundary-cone limit.
- **Shapiro, A. (1985).** Asymptotic distribution of test statistics in the analysis of moment structures under inequality constraints. *Biometrika* 72(1), 133–144. Chi-bar weights.
- **Greven, S., Crainiceanu, C. M., Küchenhoff, H. & Peters, A. (2008).** Restricted likelihood ratio testing for zero variance components in linear mixed models. *JCGS* 17(4), 870–891. The pseudo-likelihood reduction to one component when other λ are present.
- **Scheipl, F., Greven, S. & Küchenhoff, H. (2008).** Size and power of tests for a zero random effect variance or polynomial regression in additive and linear mixed models. *CSDA* 52(7), 3283–3299. A simulation comparison in which the exact-null RLRT is recommended and chi-bar is conservative.
- **Lin, X. (1997).** Variance component testing in generalised linear models with random effects. *Biometrika* 84(2), 309–326. The score test and its Σκχ² null.
- **Zhang, D. & Lin, X. (2003).** Hypothesis testing in semiparametric additive mixed models. *Biostatistics* 4(1), 57–74. Score tests for smooth terms, with a bias correction.
- **Wood, S. N. (2013a).** On p-values for smooth components of an extended generalized additive model. *Biometrika* 100(1), 221–228. The rank = round(edf) Wald test that gamfit implements.
- **Wood, S. N. (2013b).** A simple test for random effects in regression models. *Biometrika* 100(4), 1005–1010.
- **Wood, S. N., Pya, N. & Säfken, B. (2016).** Smoothing parameter and model selection for general smooth models. *JASA* 111(516), 1548–1563. LAML and its derivatives.
- **Imhof, J. P. (1961).** Computing the distribution of quadratic forms in normal variables. *Biometrika* 48(3/4), 419–426. Also Davies, R. B. (1980), *Appl. Statist.* 29(3), 323–333 (algorithm AS 155).
- **Tierney, L. & Kadane, J. B. (1986).** Accurate approximations for posterior moments and marginal densities. *JASA* 81(393), 82–86. The O(n^{−1}) Laplace error.
- **Shun, Z. & McCullagh, P. (1995).** Laplace approximation of high dimensional integrals. *JRSS-B* 57(4), 749–760.
- **Speckman, P. (1985).** Spline smoothing and optimal rates of convergence in nonparametric regression models. *Ann. Statist.* 13(3), 970–983. Eigenvalue decay of order s^{−2m}. Also Utreras, F. (1983), *Numer. Math.* 42, 107–117.
- **Reiss, P. T. & Ogden, R. T. (2009).** Smoothing parameter selection for a class of semiparametric linear models. *JRSS-B* 71(2), 505–523. REML vs GCV near the boundary, and the multiple-minima behaviour of GCV.
- **Anderson, W. N. & Duffin, R. J. (1969).** Series and parallel addition of matrices. *J. Math. Anal. Appl.* 26, 576–594. The concavity and homogeneity of M(t) used in §3.4.
- **Huang, J. Z. & Zhang, L. (2008).** Testing in nonparametric regression: a review (extracted text in `$SP/theory/boundary-probability/scheipl_testing_poly.txt`), *Statistics Surveys* 2, 154–169.

---

## 6. Consequences for gamfit

These recommendations are consistent with the SPEC:

- no boxes, grids, finite differences, retries or magic constants;
- the domain constraint γ = e^{−ρ} ≥ 0, which is allowed.

Each item says what to delete, what to build, and which failing cluster in `$SP/q1561/all-tests.log` it addresses.

### C1. Replace the C ≻ 0 face gate with the face KKT condition

- **Delete** the positive-definiteness refusal at `crates/gam-solve/src/rho_optimizer/rail_face.rs:364–383`, including `if !(min_curvature > curvature_margin)` at line 376. This is the gate that accepts a true boundary optimum with probability 10⁻³ to 10⁻²⁵ (§3.4).
- **Promote** the per-coordinate constants c_j (`rail_face.rs:385–436`) to *the* certificate. For blocks with mutually orthogonal released ranges (all double-penalty smooths), certify iff c_j ≥ −m_j for all j ∈ F, where:
  - m_j is the numerical margin, from the same backward-error bound as now, i.e. q·ε·‖C‖·(1 + cond) propagated through the trace;
  - a coordinate with −m_j ≤ c_j < 0 is additionally admissible iff c_j²/(2b_j) + band ≤ τ_stat (C3), where b_j = ∂²V/∂γ_j² at the limit fit;
  - the gap is reported in the proof object (`RailFaceProof.value_gap`, `rail_face.rs:223`).
  
  The comment at line 424, "`C ≻ 0` makes every compression positive definite, so this can only be reached by…", becomes false and must be rewritten. The line-424 test becomes the primary test.
- **Overlapping ranges.** Certify min_{t∈Δ_F} tr(M(t)C) ≥ −m (3.4), where M(t) = (Σ_j t_j^{−1}A_j)^{−1}. The exact gradient is ∂/∂t_j tr(M(t)C) = t_j^{−2}·tr(M A_j M C). Solve by projected Newton on the simplex. This is a |F|-dimensional smooth problem; |F| ≤ the number of penalties on one term.
- **What C ≻ 0 still means.** Keep it as an informational field: a *strict second-order-free* sufficient condition. It is not a gate.
- **Degenerate coordinates.** `FaceCoordinateKind::Unidentified` (released rank 0) remains certified, as now.
- **Tests fixed:**
  - binomial-logit rail, `quality_vs_interpretml_ebm_binomial_logit` (log line 324): coordinate #2 railed at ρ = 22.73 with |Pg| = 2.3×10⁻⁵;
  - the multinomial penguins custom family (line 13475, "declined a certified optimum that an evaluated state beats");
  - every double-penalty fit with a null smooth.

### C2. Optimize boundary coordinates in γ, not ρ (active set on γ ≥ 0)

- **Build.** When a coordinate's ρ_j exceeds the point where e^{−ρ_j}μ₁ becomes negligible, i.e. where ρ_j-curvature information has vanished by (3.3), switch that coordinate's chart to γ_j = e^{−ρ_j} ≥ 0.
  - "Negligible" is itself derived: ½γ_j²·b_j ≤ τ_stat means the quadratic term cannot matter.
  - Then take projected-Newton steps with the active set {γ_j = 0}.
- **Inputs, all exact and closed-form:**
  - V_γ = −e^{ρ}V_ρ and V_γγ = e^{2ρ}(V_ρρ + V_ρ) from (3.3), for γ_j > 0;
  - V_γ(0) = c_j and V_γγ(0) = b_j from the limit fit at γ_j = 0 (`crates/gam-solve/src/reml/rail_face_limit.rs:56`), with b_j obtained by one more implicit-function derivative of the same limit fit.
  
  For the Gaussian spectral case, b = Σ_sμ_s²(w_s² − ½) with a known scale; the profiled version adds the scale term.
- **The move rule.** A boundary coordinate leaves the active set iff c_j < 0 and c_j²/(2b_j) > τ_stat − band. This is the finite, exact KKT test. There is no tail extrapolation.
- **Delete.** The exponential tail-law machinery exists to certify a ρ that "stopped moving": the fitted tail law, the stall detector and the rail boxes. This covers `asymptote_certificate.rs` Thm 2.1/2.2, and every hand-set ρ box such as [−17.36, 22.73] in log line 324. At γ = 0 the face is represented exactly.
- **Tests fixed:**
  - the binomial EBM rail (line 324);
  - the transformation/survival fits with indefinite Hessians and large |Pg| (lines 33697 and 33712, Weibull-AFT, dim = 6, |Pg| = 7.0×10⁻² and 0.27), where the ρ-chart flatness corrupts the quasi-Newton model;
  - the timeouts driven by exponentially slow tail approach.

### C3. Replace the relative cost floor with the absolute statistical tolerance

- **Delete** `resolution = outer_rel_cost_floor(config) * (1.0 + cost.abs())` at `decrement_bands.rs:131`. Also delete its siblings `outer_rel_cost_floor` (`run.rs:8839–8844`) and `criterion_curvature_resolution` = 2·rel·(1+|V|) (`run.rs:8857`), wherever they gate certification. The same applies to the defaults `tolerance: 1e-5` (`run.rs:359`, `run.rs:459`) and `COST_STALL_REL_TOL_FLOOR = 1e-7` (`bridges.rs:302`) where they act as criterion resolutions.
- **Build** τ_stat = 1/(2·n_eff), with n_eff = Σ prior weights, i.e. the observation count. Then apply (3.5):
  1. band_V > τ_stat gives the typed error `NumericallyUnresolvable { band_V, tau_stat }`.
  2. ½δ² + band_V ≤ τ_stat certifies. δ² is the projected Newton decrement, in γ for active or near-boundary coordinates (C2).
  3. Otherwise the optimizer continues. No cap, retry or jitter is involved: if the optimizer cannot make progress while (2) fails, that is a typed non-convergence.
- The negative-curvature adjudication (`criterion_curvature_resolution`'s use) likewise becomes: a direction with ½|λ_min|·α_max² ≤ τ_stat − band is irrelevant. Here α_max is the largest statistically meaningful step, which is ~1 sampling SD, i.e. α² = 1/I, so the test is |λ_min|/(2I) ≤ τ_stat − band. This is derived, not chosen.
- **Tests fixed:**
  - `gam_multinomial_recovers_true_class_simplex` (line 1009; N = 200; ½δ² = 4.7×10⁻⁵ refused against rel·(1+|V|) = 2.7×10⁻⁵; τ_stat = 2.5×10⁻³ certifies, and railing coordinate 2 would *raise* V by 0.135, so that fit is interior and correct);
  - the binomial prostate fits (lines 786 and 822, |Pg| = 9.6×10⁻⁶ refused with no rails);
  - any test whose certification flips with the units of y.

### C4. Calibrated smooth-term p-values

- **Replace** the χ²/F(ref_df) p-value at `crates/gam-terms/src/inference/smooth_test.rs:180–199` (and the `rank = round(edf)` truncation at `:164–167`, which is only needed to make the χ² reference plausible). Use one of the following:
  - **(a) Primary, deterministic, exact for Gaussian.** The variance-component score test (3.6) for the term, with the spectrum μ_s = eig(Z'P̃Z).
    - P̃ is the projection or marginal precision of the null model, with the other terms at their certified λ̂ (Greven et al. conditioning).
    - The p-value comes from Imhof/Davies inversion of a linear combination of χ²₁ and χ²_{n−p−K}.
    - For a double-penalty term, use Z whitened by the unit-weight sum of its penalties.
    - For GLMs, use Lin's r'ZZ'r with κ = eig(Z'P_WZ) at the null fit, which is first-order. Zhang & Lin's (2003) bias correction is optional.
    - The integration tolerance is derived from the p-value resolution required, e.g. an absolute error ≤ 10⁻² · p at the reported p. Imhof and Davies have explicit error bounds (Davies 1980), so the tolerance is stated, not tuned.
  - **(b) Keep the Wood-type T, but use its exact law.** T is pivotal (§3.7). Its null is π₀δ₀ plus the continuous part, generated by refitting the spectral form. This needs simulation plus a deterministic seed, so (a) is preferred.
  - **(c) RLRT with the Crainiceanu–Ruppert exact null** (atom P_glob at 0).
- **Delete** the ref_df floor ("never below rank_used", `smooth_test.rs:180–183`) and the `rank_used == 0 → None` refusal (`:172–176`). A term with f̂ ≡ 0 is a valid fit; its p-value under (a) is simply P(U ≥ u_obs), with u_obs available from the data whatever λ̂ is.
- **Addresses:** the conservativeness of the default double-penalty tests (§4.3), and every test of "p-value calibration" or "null smooth has uniform p".

### C5. Lower rails: correct the theory, fix the chart

- **Fix the doc** at `asymptote_certificate.rs:8–10`. By Proposition 5, a coordinate with r_j > h_j cannot have a lower-rail optimum: V → +∞.
- **Build** the (r_j, h_j) count per coordinate. It is a rank computation on the penalty ranges and on X'WX restricted to them, both of which are already formed.
  - If r_j > h_j, the domain for λ_j is effectively (0, ∞) with V → ∞ at both ends of the lower side, so no lower-rail certificate should ever be issued.
  - If r_j = h_j (e.g. range(S_j) ⊆ range(S_{−j}): Matérn mass + derivative), use the chart λ_j ≥ 0 near λ_j = 0. The KKT condition is ∂V/∂λ_j|₀ ≥ 0, computed at the λ_j = 0 fit; this is the mirror of C2.
- **Tests fixed:** the iso-kappa Matérn lower rails, which is the joint ρ + κ case owned by the Matérn lane:
  - log line 14600: ρ = −20.77, |Pg| = 3.62;
  - line 15518: rails [3, 4], |Pg| = 3.7×10⁻²;
  - line 15831: ρ = −21.17, |Pg| = 0.33;
  - line 33534: ρ = −20.95, |Pg| = 2.8×10⁻².
  
  A state at ρ ≈ −21 with |Pg| = O(1) is not stationary in any chart. By Proposition 5, either the coordinate has r_j = 0, in which case C5's λ_j ≥ 0 chart applies, or the optimizer is being pulled there by the κ coordinate, which is outside this lane.

### The certificate, assembled

A fit is certified iff all four of the following hold.

- **(i) Interior coordinates.** The projected Newton decrement satisfies ½δ² ≤ τ_stat − band_V, in ρ.
- **(ii) Upper-boundary coordinates** (γ_j = 0). The face KKT condition (3.4) holds, i.e. c_j ≥ −m_j for disjoint ranges. Any c_j < 0 must have Σ c_j²/(2b_j) within the same budget.
- **(iii) Lower-boundary coordinates** (λ_j = 0, allowed only when r_j = h_j). ∂V/∂λ_j ≥ −m_j, with the analogous budget.
- **(iv) Numerics.** band_V ≤ τ_stat; otherwise there is a typed failure.

Tolerance derivations:

- τ_stat = 1/(2n), from the Laplace and first-order-inference accuracy (§3.6).
- m_j comes from backward error.
- band_V comes from the existing decrement-band analysis.

Nothing is chosen by hand.

---

## 7. Open problems

1. **Local vs global boundary optima.** About 1% of null datasets have a local KKT boundary optimum beaten by an interior optimum (P_loc − P_glob ≈ 0.004–0.012). A first-order face certificate cannot see this.
   - Is there a cheap, non-grid, *global* certificate in one γ dimension? The Gaussian ℓ_R(γ) is a sum of logs of rational functions. Its stationary points are roots of a polynomial of degree ≤ 2K − 1, which could be isolated exactly by Sturm sequences. This is not a grid and would be a proof.
   - For LAML there is no analogous structure.
2. **General overlapping-face KKT.** Minimizing tr(M(t)C) over the simplex when C is indefinite and ranges overlap is a difference of concave functions. Is the minimum always at a vertex or an edge for the penalty structures gamfit builds (tensor products, Matérn)?
3. **The choice of η in τ_stat = η²/2.** η² = 1/n is justified by first-order accuracy. For Gaussian REML the criterion is exact, with no Laplace error, and a case could be made for η² = 1/edf or 1/(n − p). This matters by a constant factor only. A decision-theoretic derivation, e.g. η chosen so that the p-value error f_T·2τ is below the Monte Carlo or integration error of C4, would settle it.
4. **GLM null beyond first order.** The Σκχ²₁ law for the Lin score is first-order. Does the exact small-n law, e.g. at the n = 100 binomial size in §4.4, need a saddlepoint correction to meet "conservative = bug" at α = 0.01? For which families?
5. **Joint ρ + κ (Matérn range).** The boundary theory here fixes κ. At a lower rail, κ and ρ interact through log|S|₊, and Proposition 5's count r_j − h_j may depend on κ. This belongs to the Matérn lane.
6. **Conjecture 1** (½ < π_∞ ≤ 0.6827) needs a proof. The lower bound is a mean–median inequality for weighted centred χ² sums.
7. **Power of the score test vs the RLRT for splines.** The score is locally most powerful at γ = 0. Crainiceanu et al. (2005) show that the RLRT is better against moderate alternatives. The choice between C4(a) and C4(c) is a power question, not a calibration question.
