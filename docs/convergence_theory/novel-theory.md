# Novel theory for the outer REML/LAML smoothing-parameter problem

Lane: `novel-theory`. Scripts: `SP/theory/novel-theory/*.py`, with SP = the session scratchpad. They run in the shared venv `SP/theory/venv`. Finite differences and high-precision differentiation appear only in these test scripts; the proposed production path has none.

This lane asks, from first principles, what the ideal outer algorithm looks like, without assuming the current architecture. The ideas examined are:

- exact algebraic structure (Gaussian);
- certified homotopy;
- Morse theory on the compactified orthant;
- exact one-dimensional minimization;
- MM/EM/Fellner–Schall fixed points;
- information geometry and charts.

Where another lane already owns a result, it is cited rather than re-derived. The lanes cited are compactification, boundary-asymptotics, certificate-theory, optimizer-dynamics, fp-error-analysis, glm-laml-landscape and boundary-probability.

---

## 1. Summary

- **Single-penalty Gaussian REML is exactly log-rational in μ = 1/λ.** Its stationarity condition is one polynomial P(μ) with exact degree 2s−1, where s is the number of distinct penalized eigenvalues, and a positive leading coefficient r·r0.
  - The number of local minima on the closed half-line [0, ∞], null face included, equals ⌊N/2⌋+1. Here N is the number of positive roots of P.
  - The null face is a local minimum iff P(0) > 0, which is equivalent to a score statistic T < 1.
  - Certified root isolation (Arb on the exact rational P) therefore gives a **global** certificate for m = 1 at about 15 ms per instance for k ≤ 25 (Thm 1, Lemma 1).
- **Single-penalty REML is not unimodal in practice.** An exact census of random P-spline fits found:
  - k=12: 2.5% bimodal;
  - k=25: 9.4% bimodal and 0.1% trimodal.
  - The competing minima differ by 6–7 log-likelihood units and by 10–20 edf, e.g. edf_pen 5.1 vs 15.7.
  - Any purely local certificate can therefore certify the wrong fit. The census includes the face-versus-interior case (trial 46: face V=370.10, interior V=377.11 at edf_pen 19.7).
- **Exact coordinate slices (Thm 2).** For the multi-penalty Gaussian problem with all λ_{i≠j} held fixed, V(λ_j) is again log-rational, with every zero and pole at λ_j ≤ 0.
  - Simultaneous congruence gives it in closed form from one generalized eigenproblem.
  - An exact *global* one-dimensional minimization along any coordinate therefore costs O(p³).
  - This gives a monotone globalization move and a *coordinatewise-global* certificate, which is strictly stronger than first- and second-order stationarity.
- **Chart theory (Thm 3).**
  - When the null model is identified, V is analytic in μ_j at the face. ∂V/∂μ_j(0) = c_j is the `rail_face.rs` closed form (agreement 1e-15 against 50-digit arithmetic).
  - In ρ, Newton moves one unit per step toward the face and never arrives.
  - In μ the face is an ordinary bound constraint with multiplier c_j, so strict complementarity holds iff c_j > 0.
  - In θ = √μ the face is an ordinary critical point with Hessian 2c_j and cubic Newton convergence, but the multiplier is zero, so strict complementarity is lost.
- **Numerically stable gradient near the face.** The float64 ρ-gradient must be computed in the **a-form**, a_k = λ_k tr(H⁻¹ L S_λ⁺ S_k) with L = H − S_λ.
  - The naive form ½(λβᵀSβ + λtr(H⁻¹S) − r) has the wrong sign at μ = 1e-17.
  - The a-form stays accurate to full relative precision (Prop 3.3).
- **Overlapping penalties at a corner (Prop 3.5).** When the penalties overlap, as in te() tensor smooths, V is **not differentiable** at the corner.
  - The directional derivative is ½tr(M(u)C), where M(u) = (Σ S_k/u_k)⁻¹ is the Anderson–Duffin parallel sum. It is homogeneous of degree 1 and nonlinear.
  - Demo: the axis derivatives are 0 and 0, while the diagonal derivative is −6.80.
  - Per-axis face tests are therefore unsound for overlapping penalties. The correct test is min over the simplex of ½tr(M(u)C) > 0.
- **Fellner–Schall and EM (Thm 4).**
  - Jacobians at a fixed point: J_FS = I − 2B⁻¹H with B = diag(b), and J_EM = I − 2 diag(r)⁻¹ H (checked to 8.8e-8 over 62 fixed points).
  - Gaussian REML satisfies H ⪯ B − M1 at every fixed point (proof via Kadison–Schwarz). Therefore **stable FS fixed points are exactly the strict local minima**, and undamped FS is locally contractive at each one.
  - At the null face, FS in ρ jumps log κ0 per iteration: λ→∞ is attracting iff κ0 > 1 iff the face is KKT. EM is sublinear there.
  - `EFS_MAX_STEP = 5` is a cap that only hides this. Delete it.
- **Closed-form Kantorovich constant for Gaussian single-penalty REML (Thm 5).** It comes from the universal softplus bounds |s''|, |s'''|, |s''''| ≤ s'.
  - The Newton–Kantorovich certificate h ≤ ½ fires at Newton iteration 2 with no tuned constants.
- **Refuted or demoted ideas.**
  - Certified homotopy (Beltrán–Leykin) is correct but costs many path-tracks, and each track fails at folds. Keep it as a test oracle only.
  - Morse theory on the compactified orthant gives only the Euler identity Σ(−1)^index = 1 over the critical points, faces included. That is a consistency check, not a bound.
  - The Fisher metric is **singular at the null face in θ** (use the observed Hessian there) and **regular in μ**. The face is at finite Fisher distance and λ = 0 is at infinite distance. This fixes the natural trust-region norm.
- **Recommended design (§6).** The components are:
  - a projected exact-Hessian trust-region Newton method in the μ chart, with the Fisher-metric trust-region norm (chart-invariant);
  - the a-form gradient;
  - exact face constants c_j, and a simplex test at overlapping corners;
  - a Kantorovich/strong-regularity certificate built from rigorous error enclosures in place of every gradient tolerance;
  - for Gaussian models, exact slice minimization for globalization and a global root-isolation certificate when m = 1.

  Delete the box (`rho_domain.rs:144,190`), the tail-extrapolation certificate (`asymptote_certificate.rs:81–240`), the rail margin (`rail.rs:87`), the EFS cap (`efs.rs:5,367,539`), the block-alternation constants (`gaussian_reml.rs:23,27,31`) and the brute-force B&B (`gaussian_reml.rs:5387`).

---

## 2. Setup and notation

**Model.**
- y ∈ ℝⁿ and X ∈ ℝ^{n×p}.
- The penalties are S_k ⪰ 0 for k = 1..m, with S_λ = Σ λ_k S_k.
- r_k = rank S_k, and r = rank S_λ at generic λ.
- The chart coordinates are ρ_k = log λ_k, μ_k = 1/λ_k = e^{−ρ_k}, and θ_k = e^{−ρ_k/2} = √μ_k.

**Gaussian REML with profiled scale** (ν = n − p + r, the residual d.o.f. of REML with the null space unpenalized):

  V(λ) = (ν/2) log q(λ) + ½ log|XᵀX + S_λ| − ½ log|S_λ|₊ + const,  where q(λ) = min_β ‖y − Xβ‖² + βᵀS_λβ.

**Single penalty (m = 1).** Reduce via the generalized eigenproblem of (S, XᵀX) restricted to range(S):
- d_j > 0 are the eigenvalues of XᵀX relative to S on the penalized range;
- w_j ≥ 0 are the squared projections of y onto the corresponding columns;
- r0 = ‖y − ŷ_{unpen}‖² > 0 when the data are not interpolated.

Let d_1..d_s be the distinct values with multiplicities m_k (Σ m_k = r) and summed weights. Then

  q(μ) = r0 + Σ_j w_j t_j, where t_j = d_j/(d_j + μ) = λd_j/(1+λd_j) ∈ (0,1),
  V(μ) = (ν/2) log q(μ) + ½ Σ_k m_k log(d_k + μ) + const.

In ρ, with softplus s(x) = log(1+eˣ) and x_j = ρ + log d_j: t_j = s'(x_j), and V(ρ) = (ν/2) log q + ½Σ s(x_j) − rρ/2 + const.

**LAML (non-Gaussian).** V(λ) = −ℓ(β̂) + ½β̂ᵀS_λβ̂ + ½log|H| − ½log|S_λ|₊, where H = XᵀWX + S_λ.

**Null face F_A.** The face for an index set A is {μ_j = 0, j ∈ A}, i.e. λ_j = ∞. The limit model restricts β to ∩_{j∈A} null(S_j).

**Identifiability of the null model** means the limit inner problem has a unique minimizer with a PD Hessian on that subspace.

**Face constant.** C is the first-order form of the limit fit (boundary-asymptotics lane, `rail_face.rs:956,1113`). Q_j is an orthonormal basis of the directions released by coordinate j. The per-coordinate face constant is

  c_j = ½ tr((Q_jᵀS_jQ_j)⁻¹ Q_jᵀ C Q_j).

**FS quantities.**
- a_k = tr(S_λ⁺ λ_kS_k) − tr(H⁻¹λ_kS_k) (the trace/edf term) and b_k = λ_k β̂ᵀS_kβ̂/φ.
- The REML gradient is g_k = ½(b_k − a_k).
- The FS map is ρ_k ← ρ_k + log(a_k/b_k). B = diag(b).

---

## 3. Results with proofs

### 3.1 Exact structure, single penalty (idea 1)

**Theorem 1 (stationarity polynomial).** Define:
- Π(μ) = Π_k (d_k + μ);
- Q(μ) = q(μ)Π(μ) = r0 Π + Σ_k w_k d_k Π_{i≠k}(d_i+μ);
- N(μ) = Σ_k m_k Π_{i≠k}(d_i + μ).

Then 2V'(μ) = P(μ)/(Q(μ)Π(μ)), with P = ν(Q'Π − QΠ') + QN. P has exact degree 2s−1 and leading coefficient r·r0 > 0. On μ ≥ 0 the denominator QΠ > 0, so sign V' = sign P.

*Proof.*
- 2V' = ν q'/q + Σ m_k/(d_k+μ).
- q'/q = (Q'Π − QΠ')/(QΠ), and Σ m_k/(d_k+μ) = N/Π. Combining over the common denominator QΠ gives P.
- Q has degree s with leading coefficient r0. So the μ^{2s−1} coefficient of Q'Π − QΠ' is s·r0 − r0·s = 0, and that term has degree ≤ 2s−2.
- N has degree s−1 with leading coefficient Σm_k = r, so QN has degree 2s−1 with leading coefficient r·r0.
- All roots of Q and Π are negative, since q > 0 and every d_k > 0. ∎

**Corollary 1.1 (null-face score test).** Write q(0) = r0 + Σw_j and σ̂0² = q(0)/ν, the scale estimate of the limit model λ = ∞. Then:
- P(0)/Π(0)² = −ν Σ_j w_j/d_j + q(0) Σ_j 1/d_j.
- So V'(0) > 0 ⇔ T := [Σ_j (w_j/d_j)/σ̂0²] / Σ_j 1/d_j < 1.

*Proof.* Use q'(0) = −Σ w_j/d_j, Π'(0)/Π(0) = Σ 1/d_k and N(0)/Π(0) = Σ m_k/d_k. ∎

T is the score statistic for "no smooth", in the sense of Crainiceanu & Ruppert (2004). It is the exact, finite-sample, closed-form version of the face test. The face constant is exactly c = V'(0) = ½ (Σ_j 1/d_j)(1 − T).

**Remark (λ → 0 barrier).** As μ → ∞, q → r0 > 0 and ½Σm_k log(d_k + μ) ~ (r/2) log μ → +∞. The λ = 0 end is a natural barrier and needs no bound.

For LAML the same holds whenever the unpenalized problem has a PD Hessian: −½log|S_λ|₊ = +(r/2)log μ + const, while ½log|H| stays bounded. When the unpenalized fit does not exist (separation), log|H| partially cancels it. That case belongs to the glm-laml-landscape lane.

**Lemma 1 (exact Morse count on [0, ∞]).** Suppose P is squarefree, P(0) ≠ 0, and N is the number of roots of P in (0, ∞). Then V restricted to [0, ∞) has exactly ⌊N/2⌋+1 local minima, null face included. The face is a local minimum iff N is even.

*Proof.*
- sign V' = sign P. Squarefreeness means every positive root is a sign change.
- P(+∞) > 0, since the leading coefficient is positive. So the sign sequence runs from sign P(0) to + through N changes, and N is even iff P(0) > 0.
- Each −→+ change is an interior minimum and each +→− change is an interior maximum.
- The face μ = 0 of the half-line is a local minimum iff V'(0) > 0.
- Counting: if P(0) > 0 there are N/2 interior minima plus the face. If P(0) < 0 there are (N+1)/2 interior minima.
- Non-squarefree P is handled by first dividing by gcd(P, P'). Even-multiplicity roots are then non-extremal inflections and are excluded. ∎

**Global certificate for m = 1.**
1. Isolate the roots of P on (0, ∞) in certified balls. Use Arb complex isolation on exact rational coefficients; a real root is reported with an exactly zero imaginary part and a real ball excluding 0.
2. Evaluate V over each minimizing ball, and at μ = 0 if P(0) > 0, in interval arithmetic.
3. The global minimizer is certified when its V-interval lies strictly below all the others.

This is a complete decision procedure. It is not a search, since there is no sampling, and no tolerance enters except the interval widths, which the procedure produces itself.

**Cost.**
- Forming P takes O(s²) rational operations. Root isolation of a degree-(2s−1) polynomial is polynomial in s and the bit size.
- Measured: 1000 instances at k=12 plus 1000 at k=25 took ≈ 30 s in python-flint, including the P-spline construction.
- For large s the rational bit size grows. The equivalent in float with rigorous enclosures uses interval Newton/Krawczyk subdivision on the compact circle chart φ ∈ [0, π/2] (§3.3). It evaluates V' as the rational function ½[ν q'/q + Σ m/(d+μ)] in O(s) per interval, with no polynomial expansion.
- Exclusion boxes are those whose interval V' excludes 0; inclusion is by the Krawczyk test. Subdivision stops when every box is decided. That is a finite, proven decision and not a grid: no box width is prescribed.

**SPEC.** This is compliant: exact derivatives, no bounds, no tolerances beyond derived interval enclosures, and the null face treated as a legal limit model.

**Remark (multi-penalty algebra).**
- For m > 1 the critical equations are a polynomial system after clearing denominators. Its number of complex solutions is the REML degree (Gross, Drton & Petrović 2012), which grows combinatorially.
- V is a DC function of λ:
  - q(λ) is a minimum of functions affine in λ, so it is concave; log q is then concave.
  - log|XᵀX + S_λ| is concave (log-det of an affine PD map).
  - −log|S_λ|₊ is convex.

  So V = (concave) + (convex).
- An SOS/moment global certificate on the degree-(2s−1)^m system is out of reach for p in the hundreds. Global certification for m > 1 is left open (§7). The exact coordinate slice below is the strongest tractable replacement.

### 3.2 Exact one-dimensional minimization (idea 4)

**Theorem 2 (exact coordinate slice).** Fix λ_i for i ≠ j and write:
- A = XᵀX + Σ_{i≠j} λ_iS_i, which is PD when the model with coordinate j at λ_j = 0 is identified;
- B = Σ_{i≠j} λ_iS_i.

Then, with (α_i, z_i) from the generalized eigenproblem of (S_j, A), and s_i, b_i from a simultaneous congruence of (B, S_j) on W = range(B) + range(S_j):
- log|A + λS_j| = log|A| + Σ_i log(1 + λα_i);
- q(λ) = yᵀy − Σ_i z_i²/(1 + λα_i) − (terms independent of λ);
- log|B + λS_j|₊ = const + Σ_i log(b_i + λ s_i), with b_i, s_i ≥ 0 and b_i + s_i > 0.

Hence dV/dλ is a rational function whose zeros and poles all lie in λ ≤ 0 except for the critical points. Its numerator is a polynomial P_j(λ), and Lemma 1 applies verbatim on [0, ∞] with both faces, λ = 0 and λ = ∞.

*Proof.*
- The first two identities follow from the generalized eigendecomposition A^{-1/2}S_jA^{-1/2} = UΛUᵀ.
- For the pseudo-determinant: B and S_j are PSD. Restricted to W, B + S_j is PD, so there is a congruence T with Tᵀ(B+S_j)T = I and TᵀBT = diag(b_i/(b_i+s_i)) simultaneously diagonal. The rank of B + λS_j on W is constant for λ > 0, so the pseudo-determinant factors as stated.
- Poles and zeros of each factor sit at λ = −1/α_i or λ = −b_i/s_i ≤ 0. ∎

**Consequences.**
1. *Exact global coordinate minimization.* One generalized eigenproblem, O(p³), plus root isolation of P_j gives the global minimizer of V along coordinate j, including the faces.
2. *Coordinatewise-global certificate.* A point where no single-coordinate change decreases V is certified by m such isolations. It is strictly stronger than second-order stationarity, and it catches the multimodality of §3.1 along any axis.
3. *Monotone globalization.* Block coordinate descent with exact global 1-D steps is monotone. By Tseng (2001), when each coordinate's minimizer is unique, every limit point is coordinatewise minimal. Powell's (1973) cycling examples need non-unique coordinate minimizers, which Lemma 1 lets us detect exactly: two minima with equal V.

   This is a valid *globalization* phase, but its rate is linear, set by the off-diagonal Hessian coupling. It must hand over to Newton (§6), not replace it.

**Demo (M2).** Two overlapping penalties with λ2 = 3/2 fixed:
- deg P_1 = 7, with a unique positive root λ1 ∈ (148, 149) where V'' > 0 and V = 75.758;
- face limits: V(λ1 → 0) = 79.84 and V(λ1 → ∞) = 75.87;
- so the interior point is the certified global minimum of the slice.

**SPEC.** Compliant: exact, and no search.

### 3.3 Charts, faces and the Morse picture (idea 3)

**Theorem 3 (face analyticity; boundary-asymptotics lane for the general proof).** If the null model on face F_A is identified, V extends analytically in (μ_j)_{j∈A} to a neighbourhood of μ_A = 0. In one released coordinate:

  V = V_∞ + c_j μ_j + O(μ_j²),

with c_j the `rail_face.rs` constant.

*Numerical check (C1–C3, binomial logit, 50 digits).*
- (V − V_∞)/μ → c1 = −3.940871155646771.
- The closed form from the `rail_face.rs` formula is −3.9408711556467617.
- Ṽ(θ) = V(θ²) is exactly even, with Ṽ''(0) = 2c1.

**Proposition 3.1 (chart dynamics).**
- *ρ chart.* V = V_∞ + cμ + O(μ²) gives V_ρ = −cμ and V_ρρ = cμ, so the Newton step is Δρ = +1 exactly to leading order, and the gradient shrinks by e per step. Observed: ρ went 3 → 10.78 in 7 steps, with g 0.13 → 8.2e-5 and ratio ≈ e^{−1}.
  - When c < 0 this is ascent toward a face *maximum*: the ρ-Newton step is a saddle-attracted step.
  - This is the mechanism behind the binomial "railed" failures (§6.4).
- *μ chart.* The face is a bound μ ≥ 0, with multiplier c. KKT at the face is c ≥ 0, and strict complementarity is c > 0. Projected Newton identifies the active set in finitely many steps (Bertsekas 1982) and then converges quadratically.
- *θ chart.* The face is an interior critical point with Ṽ' = 0 and Ṽ'' = 2c. Since Ṽ''' = 0, Newton converges cubically when c > 0. Observed (chart_face_min, c = 3.122): θ went 2.8e-3 → 1.4e-7 → 1.9e-20, with h → 6.244 = 2c.
  - When c < 0, θ-Newton goes straight to the interior minimum at θ = −0.6724 (C4: g 1.1e-3 → 3.8e-7 → 4.6e-14). Negative θ is legal by evenness.
  - The price: the multiplier is zero, so a face minimum and a degenerate interior minimum look alike to first order, and the test becomes the second-order test 2c > δ.
- *Circle chart.* λ = λ̄ tan²φ, φ ∈ [0, π/2], is θ-like at both ends:
  - dρ/dφ = 4/sin 2φ and d²ρ/dφ² = −8cos 2φ/sin²2φ;
  - at the face H_φφ → 2c/λ̄.

  λ̄ is a data scale, not a bound. For example λ̄ = (γ_min γ_max)^{-1/2}, taken from the penalty-range spectrum that `rho_domain.rs:37,47` already computes.

  The compactification lane's u = edf-fraction chart is a μ-type chart with strict complementarity. Its remark that "the θ chart loses strict complementarity" is the same trade-off stated above. μ and u are recommended over θ for the production active-set logic. θ-Newton is recommended only as a local polish at a certified face minimum, where it is cubic. Both charts share the genuine degeneracy c = 0 (§7).

**Proposition 3.2 (Fisher geometry).** For Gaussian REML with known scale, I_ρρ = E[∂²V/∂ρ²] = ½ Σ_j (1 − t_j)², using E[w_j] = φ/t_j. It follows that:
- as ρ → ∞, √I_ρρ ~ μ(½Σ1/d_j²)^{1/2}, so the null face is at finite Fisher distance;
- as ρ → −∞, I_ρρ → r/2, so λ = 0 is at infinite Fisher distance, which is the barrier again;
- I_μμ = I_ρρ/μ² → ½Σ1/d_j² > 0, so μ is Fisher-regular at the face;
- in θ, I_θθ = 4θ² I_μμ → 0, so Fisher scoring stalls in θ at the face and the observed Hessian must be used there.

*Consequence.* A trust region measured in the Fisher metric ‖Δ‖_I is chart-invariant to first order:
- in the interior it equals the ρ-norm;
- at the face it equals a fixed multiple of the μ-norm.

This is the principled replacement for every hand-set ρ step bound and trust-radius constant.

**Proposition 3.3 (a-form gradient).** With L = H − S_λ (= XᵀWX), the trace term in the ρ-gradient satisfies a_k = λ_k tr(H⁻¹ L S_λ⁺ S_k).

*Proof.*
- Let P_0 = I − S_λS_λ⁺ be the projector onto null(S_λ).
- H⁻¹LS_λ⁺ = H⁻¹(H − S_λ)S_λ⁺ = S_λ⁺ − H⁻¹(I − P_0). Hence S_λ⁺ − H⁻¹ = H⁻¹LS_λ⁺ − H⁻¹P_0.
- Multiply on the right by λ_kS_k and take the trace. Since P_0S_k = 0, the last term vanishes. ∎

As μ → 0 both tr(S_λ⁺λS) and tr(H⁻¹λS) tend to r, and their difference is O(μ). The naive form subtracts two O(1) numbers and loses everything below μ ≈ ε. The a-form computes the O(μ) quantity directly.

The implementation uses the μ-scaled Schur block H⁻¹_QQ = μ(S_QQ + μ·Schur)⁻¹ on the released directions Q. It never forms H⁻¹ at λ = 1e17.

*Numerical check (C5).* The naive float64 gradient is wrong below μ = 1e-15 and has the wrong sign at 1e-17. The a-form matches −cμ at every μ down to 1e-17.

**Proposition 3.4 (face Hessian).** If faces j ∈ A release pairwise non-overlapping subspaces, the μ-chart Hessian at a face point is block-diag over A plus an interior block. The mixed face–face entries vanish to first order.

The θ-chart Hessian is block-diag(2c_j)_{j∈A} ⊕ (interior block). Both come from `rail_face.rs` quantities; no extra solves are needed.

**Proposition 3.5 (overlapping corners are non-smooth).** At a corner μ_1 = μ_2 = 0 with overlapping ranges of S_1 and S_2, the one-sided directional derivative along u ∈ ℝ²₊ is

  D_uV = ½ tr(M(u) C), where M(u) = (S_1/u_1 + S_2/u_2)⁻¹ on the released space.

M is the parallel sum (Anderson & Duffin 1969). It is positively homogeneous of degree 1 and not linear, so V is not differentiable at the corner.

*Check (multi_exact M1).*
- Disjoint ranges: D_{e1} + D_{e2} = 4.839 − 25.354 = −20.515 = D_{(1,1)}, which is additive.
- Overlapping ranges: D_{e1} = D_{e2} = 0, yet D_{(1,1)} = −6.7988 = ½tr(M(1,1)C).

The per-axis test (both zero, so "certified") is unsound. The corner is a Morse–Bott exceptional divisor after the blow-up μ = s·u, u ∈ Δ. The correct corner certificate is min_{u∈Δ} f(u) > δ with f(u) = ½tr(M(u)C). For two overlapping penalties this is a one-dimensional rational minimization, solved exactly as in §3.1.

This is precisely the te(x, z) tensor-product situation. It also explains why `certify_rail_face`'s joint λ_min(C) test is overly *strict*: it demands positive definiteness of C on all released directions, where the exact requirement is positivity of f on the simplex. The per-axis test would be overly *lax*.

**Morse theory on the compactified orthant (idea 3, demoted).**
- Apply stratified Morse theory to [0, ∞]^m, with λ = 0 a barrier and the faces counted only where −∇V points into the domain.
- The result is Σ(−1)^{index} = χ = 1. In one dimension this reproduces Lemma 1's #min − #max = 1.
- It gives no upper bound on the number of minima, so it cannot certify uniqueness. It is a consistency check on critical-point enumerations and nothing more.

**Microergodic toric rays (Matérn).** The Newton polygon of 1 + λ(κ² + ω²)² shows that the iso-kappa identifiable direction is ρ + 4ψ = const. Charts adapted to this ray are the toric analogue of μ. This is deferred to the Matérn lane.

### 3.4 MM / EM / Fellner–Schall (idea 5)

**Theorem 4a (EM is MM).** For disjoint penalties, linearize the concave part of the DC split (§3.1) at λ⁰ and minimize the convex surrogate. This gives λ_k ← r_k/(2∂_{λ_k}(concave part)), which is the EM/Harville update. V decreases monotonically (Dempster, Laird & Rubin 1977; Wu 1983).

FS (Wood & Fasiolo 2017) replaces r_k by the effective trace a_k. That is the same MM when the penalties are nested or disjoint and a heuristic otherwise.

**Theorem 4b (fixed-point Jacobians).** At a fixed point g = 0:
- J_FS = I − 2B⁻¹H with B = diag(b) = diag(a);
- J_EM = I − 2 diag(r)⁻¹ H.

*Proof.* Differentiate ρ + log(a/b) = ρ + log(1 − 2g/b) at g = 0: J = I − 2 diag(b)⁻¹ ∂g/∂ρ. EM is identical with b replaced by r. ∎

*Check (F1, F3).*
- max|J_num − (I − 2B⁻¹H)| = 8.8e-8 over 62 fixed points (overlapping penalties, known and profiled scale).
- One penalty: FS rate 1 − 2H/a = 0.555092 and EM rate 1 − 2H/r = 0.889100, numerical equal to theory. FS is faster because a < r.

**Theorem 4c (FS stability = strict local minimality, Gaussian REML).** At every FS fixed point:
- known scale: 2H ⪯ 2B − 2M1 − G;
- profiled scale: the right-hand side has an additional −bbᵀ/ν.

Here:
- M1_{kl} = λ_kλ_l β̂ᵀS_kH⁻¹S_lβ̂/φ ⪰ 0;
- G ⪰ 0 is the "gap" matrix G_{kl} = tr(P_kP_l) − 2tr(P_kP_lR) + tr(P_kRP_lR);
- P_k = λ_k S^{+1/2}S_kS^{+1/2};
- R = S^{1/2}H⁻¹S^{1/2} ⪯ I.

Hence every eigenvalue κ of B^{-1/2}HB^{-1/2} satisfies κ ≤ 1 − vᵀ(M1 + G/2)v / vᵀBv ≤ 1, where v is its eigenvector. The inequality is strict whenever M1 + G/2 ≻ 0, which holds generically: β̂ has a nonzero component in each penalized range.

At a nondegenerate local minimum, where H ≻ 0, κ ∈ (0, 1). So the eigenvalues of J_FS, which are 1 − 2κ, lie in (−1, 1), and undamped FS is locally contractive. At a saddle or maximum H has a negative eigenvalue, and J_FS has an eigenvalue > 1, so the point is repelling.

*Proof (known scale; profiled adds the rank-one −bbᵀ/ν from ∂log q).*
- Write λ_kS_k = S^{1/2}P_kS^{1/2}, with S = S_λ. Then tr(S⁺λ_kS_k) = tr P_k, tr(H⁻¹λ_kS_k) = tr(RP_k), and a_k = tr(P_k(I−R)).
- Using ∂β̂/∂ρ_l = −H⁻¹λ_lS_lβ̂:
  - ∂b_k/∂ρ_l = δ_kl b_k − 2M1_kl;
  - ∂a_k/∂ρ_l = δ_kl a_k − tr(P_kP_l) + tr(RP_kRP_l).
- At a fixed point (a = b), 2H = T − 2M1 with T_kl = tr(P_kP_l) − tr(RP_kRP_l).
- T + G = 2[tr(P_kP_l(I−R))]. So the claim 2H ⪯ 2B − 2M1 − G is equivalent to tr(A²(I−R)) ≤ tr((Σ_k v_k²P_k)(I−R)) for all v, where A = Σv_kP_k.
  - Since I − R ⪰ 0, this follows from A² ⪯ Σv_k²P_k.
  - Proof of that: stack G_s = [P_1^{1/2} … P_m^{1/2}] and D = blockdiag(v_kI), so that A = G_sDG_sᵀ.
  - G_sG_sᵀ = ΣP_k is the projector onto range(S), so G_sᵀG_s ⪯ I and A² = G_sD(G_sᵀG_s)DG_sᵀ ⪯ G_sD²G_sᵀ = Σv_k²P_k.
  - This is the Kadison–Schwarz inequality for the unital positive map X ↦ ΣP_k^{1/2}XP_k^{1/2} (Kadison 1952; Choi 1974).
- Finally, G ⪰ 0 because vᵀGv = tr(A²) − 2tr(A²R) + tr(ARAR) = ‖(I−R)^{1/2}A(I−R)^{1/2}‖_F², and M1 ⪰ 0 as a Gram matrix in the H⁻¹ inner product. ∎

*Check (F2).* spec(B^{-1/2}HB^{-1/2}) over all fixed points lies in [0.034, 0.4935]. The bound 1 is not sharp, and constructions reach 0.75, so there is no general ½ bound.

*Consequence.* An FS limit with numerically verified ρ(J) < 1 is a strict local minimum. For the per-atom frontier lane (`reml/per_atom_efs.rs`), where exact outer Hessians are unavailable, this is the certificate.

**Proposition 4d (face behaviour).**
- Let κ0 = lim_{λ→∞} a/b along the released direction. FS in ρ satisfies ρ_{t+1} − ρ_t → log κ0 (F4: 6.26 observed, with κ0 = 522).
- So λ → ∞ is FS-attracting iff κ0 > 1, which holds iff c > 0, i.e. iff the face is KKT.
- FS converges linearly in μ with rate 1/κ0 but never reaches the face in finite iterations. Eventually b underflows and log(a/0) = NaN.
- EM is sublinear there: ρ went 5.005 → 5.058 in 60 iterations.
- `EFS_MAX_STEP` (`efs.rs:5`, applied at `:367,:539`) truncates the correct log κ0 jump to 5 and adds a cap. The right treatment is to switch that coordinate to the μ chart and test c_j directly.

### 3.5 Closed-form Kantorovich certificate (Gaussian, m = 1)

**Theorem 5.** For softplus s, 0 < s' < 1 and |s''|, |s'''|, |s''''| ≤ s'. On an interval I in ρ, let:
- Q_m = Σ w_j max_I s'(x_j);
- q_lo = q(inf I), since q is increasing in ρ;
- u = Q_m/q_lo.

Then V''' is Lipschitz-bounded on I by

  L_I = (ν/2)(u + 3u² + 2u³) + ½ Σ_j max_I s'(x_j).

If the Newton quantities at ρ0 satisfy η = |V'/V''|, h = L_I η/V'' ≤ ½ and V'' − 2ηL_I > 0 with I = [ρ0 − 2η, ρ0 + 2η], then I contains exactly one critical point, and it is a strict local minimum.

*Proof.*
- q^{(k)} = Σ w_j s^{(k+1)}(x_j), so |q^{(k)}| ≤ Q_m.
- V''' = (ν/2)[q'''/q − 3q'q''/q² + 2(q'/q)³] + ½Σ s'''(x_j) gives the bound.
- The conclusion is the Kantorovich theorem (Ortega & Rheinboldt 12.6.2) applied to V', with radius (1 − √(1−2h))η/h ≤ 2η. Positivity of V'' on I follows from V''(ρ) ≥ V''(ρ0) − 2ηL_I. ∎

*Check (kanto.py).* The certificate fired at Newton iteration 2 with h = 0.125, V'' = 2.98 and η = 0.083, and the true minimum ρ = −1.542 lies inside.

With rigorous floating-point enclosures δ_g and δ_H from the fp-error-analysis lane, replace V' by |V'| + δ_g and V'' by V'' − δ_H. That is the whole derivation of the stopping rule; nothing is tuned.

For m > 1 and for LAML, the H-norm Kantorovich / Smale-α version is owned by the certificate-theory lane. Theorem 5 is its closed-form specialization and a test oracle for it.

### 3.6 Certified homotopy (idea 2, demoted)

- Beltrán–Leykin certified tracking (2012, 2013) uses Smale α-theory along a linear homotopy from a start system of total degree. It is rigorous for regular paths.
- For m = 1 it is dominated by Theorem 1: exact isolation of one univariate polynomial.
- For m > 1:
  - the number of paths is the Bézout/BKK number of the cleared system, which is at least the REML degree and grows combinatorially in m and s;
  - each path needs complex evaluations of the rational slice structure, which exists only via Theorem 2's per-coordinate factorization and not jointly;
  - certified tracking fails at folds, which occur exactly at the bimodality transitions that make the problem interesting.
- Verdict: not a production optimizer. Use it as an *offline test oracle* for small m, s via HomotopyContinuation.jl with certify() (Breiding & Timme 2018; Breiding, Rose & Timme 2020, Krawczyk).

---

## 4. Numerical checks

All scripts are in `SP/theory/novel-theory/`.

**Exact structure and multimodality**

| id | script | claim | result |
|---|---|---|---|
| G1 | `g1_exact.py` | deg P = 2s−1; lc = r·r0; P(0) sign = score test | verified symbolically (sympy) |
| G2 | `g1_mc_flint.py` | multimodality census, exact rational P plus Arb isolation | k=12: {1 min: 975, 2 min: 25} of 1000, face local min in 109. k=25: {1: 890, 2: 93, 3: 1}, face local min in 111. Synthetic two-scale instance: N=2, so face plus interior. About 30 s total |
| G3 | `g1_mc_verify.py` | dense 30-digit sign evaluation of V' agrees with the census, and gives the depths | trial 6: minima at μ = 0.0728 / 11.9, edf_pen 5.13 / 15.73, V = 185.72 / 191.55. trial 39: V = 85.88 / 92.92. trial 46: face (μ=0, edf_pen 0) V = 370.10 vs μ = 40.4 (edf_pen 19.7) V = 377.11. trial 87: V = 345.39 / 351.01 |

In G3 the printed script column "edf" is Σt = r − edf_pen with r = 23. The full 1000-trial dense recount hit the shell time limit after confirming the four multimodal instances shown, so the exact census (G2) is the authoritative count.

**Coordinate slices and overlapping corners**

| id | script | claim | result |
|---|---|---|---|
| M1 | `multi_exact.py` | overlapping corner is non-differentiable; D_uV = ½tr(M(u)C) | disjoint: 4.839 + (−25.354) = −20.515 (additive). Overlapping: 0, 0, but −6.7988 on the diagonal, equal to the closed form |
| M2 | `multi_exact.py skip` | exact slice, Thm 2 | deg P = 7; unique root λ1 ∈ (148, 149); V = 75.758 < face limits 79.84 and 75.87 |

**Charts and the a-form gradient (binomial, 50-digit arithmetic)**

| id | script | claim | result |
|---|---|---|---|
| C1 | `chart_laml.py` | V analytic in μ at the face | (V−V∞)/μ → −3.94087115564677 |
| C2 | `chart_laml.py` | Ṽ(θ) even, Ṽ''(0) = 2c | V(θ) − V(−θ) = 0 exactly; ratio → c1 |
| C3 | `chart_laml.py` | c1 equals the `rail_face` closed form | −3.9408711556467617 vs −3.9408711556467706 |
| C4 | `chart_laml.py` | ρ-Newton gives +1 per step; θ-Newton is quadratic | ρ: 3 → 10.78 in 7 steps, g ratio e^{-1}. θ: g 1.1e-3 → 3.8e-7 → 4.6e-14 |
| C4' | `chart_face_min.py` | face minimum: θ-Newton cubic, h → 2c | θ: 2.8e-3 → 1.4e-7 → 1.9e-20, h → 6.2442 = 2·3.1221 |
| C5 | `chart_face_min.py` | naive vs a-form gradient in float64 | naive wrong for μ ≤ 1e-15, wrong sign at 1e-17; a-form matches −cμ down to 1e-17 |

**Fellner–Schall and EM**

| id | script | claim | result |
|---|---|---|---|
| F1 | `fs_em.py` | J_FS = I − 2B⁻¹H | max error 8.8e-8 over 62 fixed points |
| F2 | `fs_em.py` | spec(B^{-½}HB^{-½}) < 1 | range [0.034, 0.4935] |
| F3 | `fs_em.py` | one-penalty rates | FS 0.555092 (theory 0.555092); EM 0.889100 (theory 0.889100) |
| F4 | `fs_em.py` | face: FS jumps log κ0; EM sublinear | Δρ/iteration → 6.26 = log 522; EM 5.005 → 5.058 in 60 iterations |

**Kantorovich certificate**

| id | script | claim | result |
|---|---|---|---|
| K1 | `kanto.py` | Thm 5 certificate | fires at iteration 2: h = 0.125, V'' = 2.98, η = 0.083; the minimum −1.542 is inside |

---

## 5. Literature

**Smoothing parameters, REML and variance components**
- Wood, S. N. & Fasiolo, M. (2017). A generalized Fellner–Schall method for smoothing parameter optimization. *Biometrics* 73(4):1071–1081.
- Harville, D. A. (1977). Maximum likelihood approaches to variance component estimation. *JASA* 72:320–340.
- Laird, N. M. & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics* 38:963–974.
- Gilmour, A. R., Thompson, R. & Cullis, B. R. (1995). Average information REML. *Biometrics* 51:1440–1450.
- Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear mixed models with one variance component. *JRSS-B* 66(1):165–185.

**Algebraic structure**
- Gross, E., Drton, M. & Petrović, S. (2012). Maximum likelihood degree of variance component models. *Electron. J. Statist.* 6:993–1016.

**EM, MM and coordinate methods**
- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). *JRSS-B* 39:1–38.
- Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *Ann. Statist.* 11:95–103.
- Tseng, P. (2001). Convergence of a block coordinate descent method for nondifferentiable minimization. *JOTA* 109(3):475–494.
- Powell, M. J. D. (1973). On search directions for minimization algorithms. *Math. Prog.* 4:193–201.

**Bound-constrained and trust-region methods**
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM J. Control Optim.* 20(2):221–246.
- Burke, J. V. & Moré, J. J. (1988). On the identification of active constraints. *SIAM J. Numer. Anal.* 25:1197–1211.
- Lin, C.-J. & Moré, J. J. (1999). Newton's method for large bound-constrained optimization problems. *SIAM J. Optim.* 9(4):1100–1127.
- Conn, A. R., Gould, N. I. M. & Toint, Ph. L. (2000). *Trust-Region Methods*. SIAM.
- Cartis, C., Gould, N. I. M. & Toint, Ph. L. (2011). Adaptive cubic regularisation methods, Part I. *Math. Prog.* 127:245–295.
- Robinson, S. M. (1980). Strongly regular generalized equations. *Math. Oper. Res.* 5:43–62.

**Operator inequalities**
- Kadison, R. V. (1952). A generalized Schwarz inequality and algebraic invariants for operator algebras. *Ann. Math.* 56:494–503.
- Choi, M.-D. (1974). A Schwarz inequality for positive linear maps on C*-algebras. *Illinois J. Math.* 18:565–574.
- Anderson, W. N. & Duffin, R. J. (1969). Series and parallel addition of matrices. *J. Math. Anal. Appl.* 26:576–594.

**Certification: Kantorovich, α-theory and homotopy**
- Ortega, J. M. & Rheinboldt, W. C. (1970). *Iterative Solution of Nonlinear Equations in Several Variables*, §12.6 (Kantorovich).
- Blum, L., Cucker, F., Shub, M. & Smale, S. (1998). *Complexity and Real Computation*, Ch. 8 (α-theory).
- Beltrán, C. & Leykin, A. (2012). Certified numerical homotopy tracking. *Exp. Math.* 21(1):69–83.
- Beltrán, C. & Leykin, A. (2013). Robust certified numerical homotopy tracking. *Found. Comput. Math.* 13:253–295.
- Breiding, P. & Timme, S. (2018). HomotopyContinuation.jl. *ICMS 2018*, LNCS 10931:458–465.
- Breiding, P., Rose, K. & Timme, S. (2020). Certifying zeros of polynomial systems using interval arithmetic. *ACM TOMS* 49 (2023).
- Johansson, F. (2017). Arb: efficient arbitrary-precision midpoint-radius interval arithmetic. *IEEE Trans. Comput.* 66(8):1281–1292.

**Numerical linear algebra**
- Cox, A. J. & Higham, N. J. (1998). Stability of Householder QR factorization for weighted least squares problems. In *Numerical Analysis 1997*, Pitman Res. Notes 380.

**Convexity, log-det and information geometry**
- Vandenberghe, L., Boyd, S. & Wu, S.-P. (1998). Determinant maximization with linear matrix inequality constraints. *SIAM J. Matrix Anal. Appl.* 19:499–533.
- Nesterov, Yu. & Nemirovskii, A. (1994). *Interior-Point Polynomial Algorithms in Convex Programming*. SIAM.
- Wainwright, M. J. & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. *FnT ML* 1:1–305, Thm 3.3.

**Toric geometry**
- Fulton, W. (1993). *Introduction to Toric Varieties*. Princeton.
- Sottile, F. (2003). Toric ideals, real toric varieties, and the moment map. *Contemp. Math.* 334.

**Microergodic parameters**
- Zhang, H. (2004). Inconsistent estimation and asymptotically equal interpolations in model-based geostatistics. *JASA* 99:250–261.

---

## 6. Consequences for gamfit

All paths are relative to `crates/gam-solve/src/` of the reviewed snapshot unless stated otherwise.

### 6.1 Recommended design

**Outer variable and domain.**
- Per coordinate, use μ_j = λ̄_j/λ_j ∈ [0, ∞). The only constraint is μ_j ≥ 0, which is a domain constraint of the mathematics (λ = ∞ is the limit model), not a box.
- λ = 0 is not a boundary. It is a barrier under identification (Remark after Cor 1.1), so no upper bound on μ is needed.
- λ̄_j = (γ_min γ_max)^{-1/2} from `penalty_range_gammas_*` (`estimate/rho_domain.rs:37,47`) is a *scale*. It conditions the chart but constrains nothing.

**Oracle.** At each iterate the oracle returns, from one inner solve:
- V;
- the a-form gradient, computed through the μ-scaled Schur block on released directions (Prop 3.3);
- the exact outer Hessian, already required for LAML by the optimizer-dynamics lane's ARC recommendation;
- rigorous enclosures δ_V, δ_g and δ_H (fp-error-analysis lane).

**Step.** Projected trust-region Newton (Lin & Moré 1999), or ARC (Cartis, Gould & Toint 2011), on the orthant μ ≥ 0.
- The trust-region norm is the Fisher metric ‖Δ‖²_I = ΔᵀI(μ)Δ. It is Fisher-regular at the face (Prop 3.2) and equals the ρ-norm in the interior, so no step caps are needed.
- Active-set identification is finite under strict complementarity c_j > 0 (Burke & Moré 1988).
- Once the face is identified, optionally switch that coordinate to θ_j for cubic local convergence (Prop 3.1). This is a polish, not a requirement.

**Overlapping-face corners.**
- When the released subspaces of two face coordinates intersect (`rail_face.rs` already computes the released ranks at `:300`), replace per-axis multipliers by the simplex test.
- Compute f(u) = ½tr(M(u)C) and minimize it exactly on Δ. For two coordinates this is a rational function of one variable, handled as in §3.1.
- If min f < 0, the descent direction is the minimizing u, and the trust-region step is taken along it.

**Gaussian specialization.**
- (a) m = 1: replace the whole optimizer by Theorem 1. Isolate the roots of P, evaluate V at the candidate minima and the face in interval arithmetic, and return the certified global minimum.
- (b) m > 1: before Newton, run exact global slice minimizations (Thm 2) in cyclic order until one sweep makes no coordinate's global slice minimizer change basin. Then hand over to Newton.

  The final certificate includes coordinatewise-global optimality: m slice isolations at the accepted point.
- This replaces the brute-force enumeration/B&B in `enumerate_and_select_rho_with_controls` (`gaussian_reml.rs:5387`, #2585), the block-alternation constants (`gaussian_reml.rs:23,27,31`), and the orthogonal-block special path (`gaussian_reml_blocks_orthogonal_shared_scale`, `gaussian_reml.rs:2197`), which is a special case of exact slices.

**Termination.** There is no gradient tolerance and no iteration cap. The iteration stops when one of two things happens:
- (i) the certificate below holds, and a fit is returned;
- (ii) the model's predicted decrease falls below δ_V while the certificate fails, and the problem is reported **not resolvable at working precision** and refused.

  Alternative (ii) is a proof that the error floor exceeds the certificate margin, not a timeout. Global convergence of the trust-region method to second-order critical points (Conn, Gould & Toint 2000, Thm 6.6.8) guarantees that one of the two occurs in finitely many steps whenever V is bounded below on the domain. V is bounded below because λ = 0 is a barrier and the faces are compactified.

**FS/EFS.**
- Remove it from the main path.
- Keep it only where no exact outer Hessian exists (the per-atom frontier lane, `reml/per_atom_efs.rs`).
- There, accept a limit only if the Jacobian test ρ(I − 2B⁻¹Ĥ) < 1 holds with an enclosure, which certifies a strict local minimum by Thm 4c. At faces, use κ0 > 1 together with the c_j test.

### 6.2 The certificate (theorem)

**Theorem 6 (certified local REML/LAML minimizer on the compactified orthant).** Let x = (μ_F, μ_A = 0) with free set F and face set A. Let g, H be the μ-chart gradient and Hessian with enclosures δ_g, δ_H. Let L be a Lipschitz bound for H on the ball B(x, 2η) ∩ ℝ^m₊ (certificate-theory lane; Theorem 5 in closed form when m = 1).

Assume:
1. (**free block**) σ := λ_min(H_FF) − δ_H > 0, η := ‖H_FF⁻¹‖·(‖g_F‖ + δ_g), and h := Lη/σ ≤ ½;
2. (**faces, non-overlapping**) for each j ∈ A, c_j − δ_{c_j} > 2ηL_j, where L_j bounds ∂c_j/∂x on the ball;
3. (**faces, overlapping groups G ⊂ A**) min_{u∈Δ_G} f_G(u) − δ_{f_G} > 2ηL_G.

Then there is exactly one KKT point of V on B(x, 2η) ∩ ℝ^m₊. It has active set exactly A, it is a strict local minimizer, and it depends Lipschitz-continuously on the data.

*Proof.*
- Condition 2/3 at radius 2η keeps every face multiplier (or simplex-directional derivative) strictly positive on the ball. So every KKT point in the ball has active set A, and the problem reduces to the free equation g_F(μ_F, 0) = 0.
- Condition 1 is the Kantorovich hypothesis for that equation. It gives a unique zero at radius ≤ 2η with H_FF ≻ 0 there.
- Strict complementarity plus a positive definite reduced Hessian is Robinson strong regularity (Robinson 1980), which gives strict local minimality and Lipschitz stability.
- In the overlapping case, the blow-up coordinates (s, u) make condition 3 the strict-complementarity condition on the exceptional divisor. ∎

**Global strengthening.**
- Gaussian m = 1: Thm 1 plus Lemma 1 gives global optimality.
- Gaussian m > 1: add the coordinatewise-global slice check (Thm 2).
- LAML: the certificate is local. Global statements belong to the glm-laml-landscape lane.

### 6.3 How each tolerance is derived

| quantity | derivation |
|---|---|
| δ_V, δ_g, δ_H | Rigorous forward error of the inner solve and trace/Schur computations (fp-error-analysis lane). No free constant. |
| h ≤ ½ | Kantorovich constant, a theorem not a tuning. |
| face margin δ_{c_j} | FP enclosure of c_j: the Schur-block solve error times cond, as in the existing `curvature_margin` at `rail_face.rs:374`, but applied to c_j, not to λ_min(C). |
| 2ηL_j term | Persistence of the multiplier over the certified ball, Thm 6. |
| trust-region radius | Standard ratio test of Conn, Gould & Toint. The initial radius is 1 in the Fisher metric, which is dimensionless because the metric is the information. |
| termination | Predicted decrease < δ_V together with the certificate state; there is no iteration or time budget. |
| λ̄_j | Data scale from the penalty-range spectrum. It is not a bound: the result is invariant to λ̄ except in conditioning. |
| subdivision in root isolation (Gaussian) | Decided by interval exclusion/Krawczyk inclusion; no width parameter. |

### 6.4 What to delete, change and build (file:line)

**Delete.**
- The box: `estimate/rho_domain.rs:144` `resolvability_interval` (as a box), `:190–193` `coordinate_domain` clamping to `LOG_STRENGTH_MIN/MAX`, and the box consumers `:297`, `:337`, `:363`, `gaussian_reml_multi_penalty.rs:~548,~1097`.
  - The resolvability *reasoning* survives as δ_g-driven refusal (§6.1 Termination), not as bounds.
  - `LOG_STRENGTH_MIN/MAX` in gam-problem should then be deleted as outer bounds.
- The tail-extrapolation certificate: `rho_optimizer/asymptote_certificate.rs:81` (`DEFAULT_ASYMPTOTE_WINDOW=12`), `:86` (`MIN_TAIL_SAMPLES=3`), `:117` `tail_constant`, `:221,:226,:230` (`EXP4_*` constants), and `:340` `assess_coordinate`. The face is decided exactly by c_j (Thm 3); no tail sampling is needed.
- The rail margin: `rho_optimizer/rail.rs:87,98` (`coordinate_rail_margin`, 0.5). There are no rails without a box.
- The EFS cap: `reml/reml_outer_engine/efs.rs:5` `EFS_MAX_STEP=5.0` and the clamps at `:367,:539`. Also delete the "q_eff ≤ 0 → zero step, rely on the outer fallback" branch documented at `:127–133`, which is a fallback.
- The Gaussian brute force: `gaussian_reml.rs:5387` `enumerate_and_select_rho_with_controls` (B&B), `:23` `BLOCK_ORTHOGONAL_SCORE_TOL=1e-7`, `:27` `MAX_OUTER_PASSES=200`, `:31` `BLOCK_UPDATES_PER_PASS=32`, and the orthogonal-block path at `:2197`.

**Change.**
- `rho_optimizer/rail_face.rs:332` `certify_rail_face`: replace the joint test λ_min(C) > margin at `:369–383` with:
  - per-coordinate c_j > δ_{c_j} + 2ηL_j for non-overlapping releases;
  - the simplex test for overlapping groups (released-rank logic at `:300`).

  The current joint test over-refuses. It demands C ≻ 0, which (per boundary-asymptotics) accepts only about P(χ²_r < 1) of true null faces, whereas Thm 1 shows the exact condition is the scalar c_j > 0 (T < 1 for Gaussian m = 1).
- Reuse `gaussian_rail_face_limit` (`:956`) and `laml_rail_face_limit` (`:1113`) as the source of c_j, of the face Hessian blocks (Prop 3.4) and of C for f_G(u).
- ρ-gradient everywhere: the a-form with the μ-scaled Schur block (Prop 3.3). The naive ½(b + τ − r) form must not be used for any coordinate with μ < √ε·(scale). It is simplest to delete it outright.

**Build.**
- In the **opt crate**, a generic projected trust-region/ARC Newton on the orthant with:
  - a user-supplied metric (Fisher);
  - oracle enclosures (δ's);
  - a Theorem 6 certificate object;
  - no bounds API.
- In **gam-solve**, the μ-chart oracle: a-form gradient, exact outer Hessian, Fisher information in μ, and face constants.
- In **gam-solve / gaussian**:
  - `exact_slice(j)`: one generalized eigenproblem, rational V'(λ_j), and certified root isolation (Thm 2);
  - `gaussian_single_penalty_global`: Thm 1 plus Lemma 1, returning the global minimizer with its certificate.

  Root isolation uses interval Newton/Krawczyk on the circle chart in f64 intervals, with an exact-rational (FLINT-style) path only in tests.
- **Test oracles** (test code only): the census of §4 G2, the certified homotopy of §3.6 for m ≤ 3, and the 50-digit face-constant check C3.

### 6.5 Which failing clusters each item addresses

| cluster (evidence) | mechanism (this lane) | fix |
|---|---|---|
| Binomial railed at θ = 22.73: \|Pg\| = 2.28e-5 > 7.3e-6, BFGS StepSizeTooSmall (q1561 `all-tests.log` 324/871/886) | Face optimum in the ρ chart: the gradient decays like e^{−ρ}, BFGS conditioning is about e^{ρ}, and the naive gradient loses sign below μ ≈ ε (Prop 3.1, 3.3; C4, C5). The box edge is where noise equals signal. | μ chart plus the a-form gradient plus the exact c_j face test (Thm 3, Thm 6); delete the box and the rail margin. |
| Prostate, no rail, ρ3 = 20.73, \|Pg\| = 9.6e-6 (log 786/822) | Same as above, but the multiplier is too small for the naive gradient to show, so the face is never declared. | Same; the face is decided by c_3's sign with an enclosure. |
| Face refusals ("not positive definite") | λ_min(C) > 0 is stronger than the exact condition c_j > 0; for overlapping faces the correct test is the simplex test (Prop 3.5). | Rewrite `certify_rail_face` as in §6.4. |
| Tensor te() / overlapping penalties at corners | V is non-differentiable at the corner; per-axis derivatives vanish while the diagonal descends (M1). | Simplex test plus the blow-up direction. |
| Gaussian B&B cost and timeouts (#2585), block-alternation constants | Enumeration where an exact algebraic answer exists. | Thm 1 (m = 1, global), Thm 2 slices (m > 1). |
| Wrong-minimum risk (silent) | 2.5–9.4% of single-penalty fits are multimodal, with 6–7 log-unit gaps (G2, G3). | Global certificate for m = 1; coordinatewise-global certificate for m > 1. |
| Outer stalls and timeouts | First-order and quasi-Newton methods in badly scaled charts, plus caps. | Exact-Hessian trust region in the Fisher metric; termination by certificate or proved non-resolvability. |
| EFS instability, NaN at faces | Log κ0 jumps capped at 5; b underflows (F4). | Delete EFS from the main path; in the per-atom lane, certify by the Jacobian test (Thm 4c) and κ0/c_j. |
| Iso-kappa Matérn tail-snap | Microergodic toric ray ρ + 4ψ = const is not a coordinate face. | Toric chart; owned by the Matérn lane. |

---

## 7. Open problems

1. **Global certification for m > 1.** The REML degree grows combinatorially. Is there an SOS or moment relaxation that is tight on the DC structure (concave log q + log|H|, convex −log|S_λ|₊)? A candidate is a convex–concave outer approximation with branch-and-bound on log λ intervals using the rational slice bounds of Thm 2.

   A B&B with exact interval bounds is a decision procedure, not a search. Its worst-case cost, however, is exponential in m.
2. **Degenerate faces c_j = 0.** Both μ and θ charts lose the certificate. The next-order term c2 (C1 found c2 ≈ 17.28) decides it. A third-order Kantorovich variant is needed, and the Gaussian census shows c = 0 has probability zero. Is exact detection of c = 0 needed at all, or is refusal (δ_c > |c|) the correct outcome?
3. **FS theorem beyond Gaussian.** Theorem 4c uses the exact Gaussian Hessian identity. For LAML, the third-derivative (W') terms break H ⪯ B − M1. Is there a PQL-type bound that keeps "stable FS fixed point ⇒ local minimum" true?
4. **Minimizing f_G(u) on a simplex for |G| ≥ 3.** The parallel sum M(u) is jointly concave in u (Anderson–Duffin), so f is concave when C ⪰ 0 on the released space. In that case the minimum is at a vertex: the axis tests restricted to the null spaces of the other penalties.

   For indefinite C, is there a closed form or an exact procedure beyond |G| = 2?
5. **Multimodality census for LAML.** Only the Gaussian case is exact here. Does the ⌊N/2⌋+1 structure persist for binomial/Poisson with a small number of observations? This is joint with glm-laml-landscape.
6. **Toric charts for Matérn and anisotropic Duchon.** Classify the Newton polygons of the spectral penalties. Is the microergodic ray always a face of the moment polytope, so that a μ-type chart exists along it (Fulton 1993; Sottile 2003)?
7. **Rational bit growth in exact P for s ≥ 100.** The f64-interval Krawczyk route avoids expansion, but its worst-case subdivision depth near near-double roots, which are the bimodality transitions, is unbounded in principle. A separation bound in terms of the discriminant of P would make the cost provably finite.
