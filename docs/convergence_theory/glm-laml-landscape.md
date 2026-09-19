# The LAML landscape for non-Gaussian GAMs (binomial logit, Poisson)

Slug: `glm-laml-landscape`. Scripts: `SP/theory/glm-laml-landscape/{replica.py, face_stability.py, leak.py, sep_part_a.py, sep_part_a_hp.py, multistart_part_b.py}`, where SP is the session scratchpad. They run with `SP/theory/venv/bin/python`.
Failing clusters in scope:
- **B-rail**: the ebm, sklearn and pymc binomial-logit tests on the prostate data. Examples: `quality_vs_interpretml_ebm_binomial_logit.rs:191` and `quality_vs_sklearn_binomial_logit.rs`.
- **B-pymc**: `quality_vs_pymc_nuts_binomial_logit.rs` and `quality_vs_pymc_hmc_binomial_penalized_vs_unpenalized.rs`. These fail with |Pg| = 9.6e-6 against a bound of 7.3e-6, unrailed at ρ₂ = 20.73.
- **L-link**: mgcv CI `quality_vs_mgcv_confidence_interval_gaussian_logistic_link.rs:151`, which fails with `OuterHessianNotAnalytic` from the latched #784 correction.

---

## 1. Summary

- **V is real-analytic on all of ℝ^M, or β̂ exists nowhere.** For binomial or Poisson with λ > 0, β̂(ρ) exists and is unique for every ρ ∈ ℝ^M exactly when the data are not (quasi-)separable inside the joint null space ∩ₖ ker Sₖ, and there is no aliasing there (Albert & Anderson 1984 and Haberman 1974 applied to the unpenalized sub-model). Existence does not depend on ρ. When β̂ exists, V is real-analytic in ρ (Result 1). Separation that lies in a *penalized* direction never destroys existence. Along that direction V → +∞ like ½ log|ρ|, with slope O(1/|ρ|) (Result 5, verified numerically).
- **Every ρₖ → +∞ face is an analytic boundary point in t = e^{−ρₖ}.** The −rₖ log t terms of log|H| and log|S|₊ cancel identically. Therefore V(ρ) = V_face + aₖ e^{−ρₖ} + O(e^{−2ρₖ}), where V_face is the ordinary LAML of the reduced model with the range of Sₖ removed. The closed form for aₖ has four terms: a score term, a Schur-trace term, a **W′ (third-derivative) term** and a penalty-overlap term (Result 3). It is verified to 5 digits against −λ ∂V/∂ρ on the replica.
- **When aₖ > 0 the infimum is at the face and is not attained in ℝ^M.** ∂V/∂ρₖ = −aₖe^{−ρₖ} never vanishes, so no ρ-space stationarity test can certify the fit. The exact certificate is a KKT condition on the compactified domain: ∇_free V_face = 0, aₖ > 0 for every face coordinate, and ∇²_free V_face ≻ 0 (Result 4). The condition is the LAML analogue of the REML variance-component score test at zero (Crainiceanu & Ruppert 2004).
- **The prostate replica has the same face.** The replica is y ~ s(pc1,k=5)+s(pc2,k=5) with double penalties, n = 490 and p = 9. Exact Newton with the analytic ρ-Hessian drives the same coordinate as the failing tests, #2 (the s(pc2) wiggle penalty), to +∞, with a₂ = +0.04594 > 0. The other three coordinates converge.
- **The analytic Laplace gradient is complete for canonical links only with the W′ term.**
  - The gradient with the drift C[vₖ] = Xᵀdiag(c ⊙ Xvₖ)X, where c = dW/dη, matches 40-digit mpmath central differences to a relative error of 2.9e-12.
  - Dropping C[vₖ] gives a 5.6 % error.
  - The ρ-Hessian needs W″ as well. Implemented with it, it matches finite differences to 2.9e-10.
  - gamfit's Laplace gradient does contain C[vₖ] (`reml_outer_engine/objective.rs:1116-1125` and the gradient loop at ~1200-1330).
- **Root cause of the B-rail and B-pymc failures: λ-amplified penalty leakage, not arithmetic conditioning.** In floating point, fl(RₖᵀRₖ) is not exactly zero on the nominal null space of Sₖ: u_nᵀfl(Sₖ)u_n ≈ 1e-16 · ‖Sₖ‖. Multiplying by λₖ = e^{22.73} ≈ 7e9 creates a spurious ridge of 6e-7 to 3e-6 on the unpenalized linear pc2 term.
  - On the replica at ρ₂ = 22.73 this contaminates the *interior* gradient by 6e-6 to 9e-6, and the face coordinate by 5e-5. It shifts V by 1.9e-4 and makes V jagged at the 2e-5 level. That is the failing |Pg| = 2.28e-5 > 7.3e-6 and the `StepSizeTooSmall`.
  - With exact structural zeros, the same κ(H) = 1.3e12 computation is accurate to 1e-15, and the fused rank cancellation of #2331 is not the issue.
  - The +∞ face chart β_r = √t γ is exact and has κ = 1.2e3.
- **The #784 block-local quadrature correction must leave the outer criterion.** Δ_b is not continuous in ρ: it jumps when the top-m |γ| ranking of the eigen-directions changes (a codimension-1 set) and is undefined at eigenvalue crossings of H (codimension 2). Its gradient is a documented wrong contraction with 1e-4 to 1.3e-1 relative error (`block_quadrature_correction.rs:1002-1030`), and it has no ρ-Hessian. Once latched, it forces BFGS (`gradient_hessian.rs:177`, `objective.rs:2247`, `objective.rs:2372`, `eval.rs:993`) and kills the smoothing-corrected covariance. That is the entire L-link failure, and it is a second, independent way to produce `curvature_source=unavailable` plus `StepSizeTooSmall` in B-rail.
- **What to build.**
  - Delete the #784 splice.
  - Replace the hand ρ-box with the compactification t = e^{−ρ} on the +∞ side and λ = e^{ρ} on overlapping −∞ sides.
  - Evaluate near a face in the face chart, where λ never multiplies Sₖ in floating point.
  - Certify with the face KKT (aₖ > 0 plus the reduced Newton system).
  - If a beyond-Laplace correction is wanted, the only smooth candidate is the full tensor Tierney–Kadane / Shun–McCullagh O(n⁻¹) term, which is basis-invariant and has analytic derivatives.

---

## 2. Setup and notation

- Data (yᵢ, xᵢ), i = 1..n. Design X ∈ ℝ^{n×p}, η = Xβ, log-likelihood ℓ(β) = Σᵢ ℓᵢ(ηᵢ). Each ℓᵢ is real-analytic in η; this holds for the logit, probit, cloglog and log links.
- Working weight and its derivatives: W = diag(wᵢ), wᵢ = −∂²ℓᵢ/∂ηᵢ² (observed information), cᵢ = dwᵢ/dηᵢ, dᵢ = d²wᵢ/dηᵢ².
  - Binomial logit: w = μ(1−μ), c = w(1−2μ), d = w(1−6w).
  - Poisson log: w = c = d = μ.
- Penalties: Sₖ = RₖᵀRₖ ⪰ 0 with rank rₖ, k = 1..M; λₖ = e^{ρₖ}; S_λ = Σₖ λₖSₖ; Aₖ = λₖSₖ.
- Joint null space N_S = ∩ₖ ker Sₖ. It is the same for every λ ∈ (0,∞)^M, because range(Σ λₖSₖ) = Σ range(Sₖ) for PSD summands.
- Penalized objective f(β, ρ) = −ℓ(β) + ½βᵀS_λβ, with β̂(ρ) = argmin_β f and H(ρ) = XᵀW(β̂)X + S_λ.
- LAML (Wood 2011, eq. for 𝒱_r with the scale fixed at 1 for binomial and Poisson):

  V(ρ) = f(β̂, ρ) + ½ log|H| − ½ log|S_λ|₊ + const.

- Derivatives used throughout (Wood 2011 §3 and Appendix; Wood, Pya & Säfken 2016 §3):
  - Mode response: vₖ := H⁻¹Aₖβ̂, so ∂β̂/∂ρₖ = −vₖ.
  - Total Hessian drift: Ḣₖ = Aₖ + C[−vₖ], where C[u] := Xᵀdiag(c ⊙ Xu)X.
  - Gradient: ∂V/∂ρₖ = ½β̂ᵀAₖβ̂ + ½tr(H⁻¹Ḣₖ) − ½∂ₖlog|S_λ|₊.
  - Second-order mode response: β̂ₖₗ = −H⁻¹(Ḣₗ(−vₖ)… ) is written in the scripts as `bkl = -H⁻¹(Ḣₗ β̂ₖ + δₖₗ Aₖβ̂ + Aₖβ̂ₗ)`, with β̂ₖ = −vₖ.
  - Second-order drift: Ḧₖₗ = δₖₗAₖ + Xᵀdiag(d ⊙ Xβ̂ₖ ⊙ Xβ̂ₗ + c ⊙ Xβ̂ₖₗ)X.
  - Hessian:

    ∂²V/∂ρₖ∂ρₗ = δₖₗ½β̂ᵀAₖβ̂ + β̂ᵀAₖβ̂ₗ + ½[tr(H⁻¹Ḧₖₗ) − tr(H⁻¹ḢₗH⁻¹Ḣₖ)] − ½∂²ₖₗlog|S_λ|₊.

  The **W′ term** is the C[·] part of Ḣₖ. The **W″ term** is the d part of Ḧₖₗ.
- Face chart for coordinate k. Take the orthonormal eigenchart U = [U_n U_r] of Sₖ, with Sₖ = U_r D U_rᵀ and D ≻ 0 (r×r). Let S_rest = Σ_{j≠k} λⱼSⱼ, and write blocks as M_nn = U_nᵀMU_n and so on. Set t = e^{−ρₖ} = 1/λₖ.

## 3. Results with proofs

### Result 1 (existence and analyticity)

Take binomial (any link with log-concave inverse link) or Poisson (log link), and λ ∈ (0,∞)^M.

(i) β̂(ρ) exists and is unique iff there is no u ∈ N_S \ {0} with Xu = 0, and no u ∈ N_S with (2yᵢ−1)xᵢᵀu ≥ 0 for all i and Xu ≠ 0 (binomial: Albert & Anderson 1984, Thm 1–3, applied to the sub-model X|_{N_S}). For Poisson the condition is: no u ∈ N_S with xᵢᵀu ≤ 0 for all i, xᵢᵀu = 0 whenever yᵢ > 0, and Xu ≠ 0 (Haberman 1974, Thm 2.2).

(ii) The condition does not involve ρ. Hence β̂ exists either on all of ℝ^M or nowhere.

(iii) When it exists, β̂, H and V are real-analytic on ℝ^M.

*Proof.* Split β = β_N + β_⊥ with β_N ∈ N_S. On N_S^⊥, S_λ ⪰ λ_min·σ_min(S) I ≻ 0, so f is coercive in β_⊥ for every λ > 0. −ℓ is convex. Along directions in N_S, f equals −ℓ plus a coercive function of β_⊥, so coercivity reduces to that of −ℓ on N_S. For a convex function this is the recession-cone condition, which the cited theorems characterize. Strict convexity comes from H = XᵀWX + S_λ ≻ 0. Here W ≻ 0 because 0 < μ < 1 or μ > 0 at a finite η, and ker(XᵀWX) ∩ ker S_λ = ker X ∩ N_S = {0}.

For analyticity, ∇_βf(β, ρ) = 0 is an analytic system with Jacobian H ≻ 0. The analytic implicit function theorem (Krantz & Parks 2002, Ch. 2) makes β̂ analytic. log|H| is analytic on the PD cone. log|S_λ|₊ = log det(QᵀS_λQ) for a fixed orthonormal basis Q of the λ-independent range, so it is analytic too. ∎

*Consequence.* Firth rescue is only needed when separation lies in N_S, i.e. in unpenalized directions (see `gam-models/src/fit_orchestration/fit.rs:533-600`). Separation in a penalized direction never makes the LAML ill-posed for λ > 0 (Result 5).

### Result 2 (non-convexity)

V is not convex in ρ in general, even for canonical links.

*Witness.* On the prostate replica, the analytic ρ-Hessian at ρ = (1, −1, 3, −2) has three negative diagonal entries. At ρ = 0 its spectrum contains −3.66 (§4).

*Structural reason.* Near a +∞ face with aₖ < 0, V ≈ V_face + aₖe^{−ρₖ} is concave in ρₖ. In general ½log|H| − ½log|S|₊ is a difference of functions of ρ with no convexity relation. Reiss & Ogden (2009, §4 and §5) show that REML for P-spline-type models can have multiple local optima (less often than GCV). Wood (2011, §1–2) motivates Newton with a Hessian-modification step because the LAML ρ-Hessian can be indefinite. The multistart count on the replica is in §4.

### Result 3 (the +∞ face is an analytic boundary point in t)

Fix ρ_{−k}, write t = e^{−ρₖ} and Ṽ(t) := V(ρ). Let β̂_n⁰ be the minimizer of the face-reduced objective f₀(β_n) = −ℓ(U_nβ_n) + ½β_nᵀS_rest,nnβ_n. It exists under Result 1's condition, because the reduced model has the same null space intersected with ker Sₖ. Let H_nn⁰ = X_nᵀWX_n + S_rest,nn ≻ 0.

Then Ṽ extends to a real-analytic function on a neighbourhood of t = 0, with

- Ṽ(0) = V_face := f₀(β̂_n⁰) + ½log|H_nn⁰| − ½log|S_rest,nn|₊ + const, which is the LAML of the reduced model with the columns XU_r removed;
- Ṽ′(0) = aₖ, where

**aₖ = −½ g_rᵀD⁻¹g_r + ½ tr(D⁻¹ Schur_rr(H)) + ½ tr((H_nn⁰)⁻¹ DH_nn[d]) − ½ tr(D⁻¹ Schur_rr(S_rest)).**

The quantities in this formula, all evaluated at (β̂_n⁰, 0):

- g_r = U_rᵀ∇_βf|_{(β̂_n⁰, 0)} = −X_rᵀ(y − μ) + S_rest,rn β̂_n⁰, the face score in the removed directions;
- Schur_rr(H) = H_rr − H_rn(H_nn)⁻¹H_nr, using H without the λₖSₖ term;
- Schur_rr(S_rest) = S_rest,rr − S_rest,rn S_rest,nn⁺ S_rest,nr;
- the face tangent d = (d_n, d_r) with d_r = −D⁻¹g_r and d_n = −H_nn⁻¹H_nr d_r;
- DH_nn[d] = X_nᵀdiag(c ⊙ Xd)X_n, the **W′ term**.

Consequently:

- V(ρ) = V_face + aₖe^{−ρₖ} + O(e^{−2ρₖ});
- ∂V/∂ρₖ = −aₖe^{−ρₖ} + O(e^{−2ρₖ});
- ∂V/∂ρⱼ = ∂V_face/∂ρⱼ + O(e^{−ρₖ}) for j ≠ k.

*Proof.*

(a) **Mode.** Substitute β_r = tγ. The stationarity equations become:

- n-block: −U_nᵀ∇ℓ(U_nβ_n + tU_rγ) + S_rest,nnβ_n + tS_rest,nrγ = 0;
- r-block, multiplied through by t: −U_rᵀ∇ℓ + Dγ + S_rest,rnβ_n + tS_rest,rrγ = 0.

Both are analytic in (β_n, γ, t) including t = 0. At t = 0 the Jacobian in (β_n, γ) is [[H_nn⁰, 0], [H_rn, D]], which is block-triangular with invertible diagonal blocks. By the analytic IFT, (β̂_n(t), γ(t)) is analytic at 0, with γ(0) = −D⁻¹g_r = d_r. Differentiating the n-block equation gives β̂_n′(0) = −H_nn⁻¹H_nr d_r = d_n.

(b) **f-term.** By the envelope theorem, dF/dλ = ½β̂_rᵀDβ̂_r = ½t²γᵀDγ. So dF/dt = −½γᵀDγ, which at 0 equals −½g_rᵀD⁻¹g_r.

(c) **log|H|.** In the chart,

log|H| = log|λD + H_rr| + log|H_nn − H_nr(λD + H_rr)⁻¹H_rn|
       = r log λ + log|D| + log|I + tD⁻¹H_rr| + log|H_nn(t) − tH_nrD⁻¹H_rn + O(t²)|.

H_nn(t) depends on t only through β̂(t), whose t-derivative is d. So (d/dt) log|H_nn(t)| = tr(H_nn⁻¹DH_nn[d]). Collecting terms,

(d/dt)[log|H| − r log λ]₀ = tr(D⁻¹H_rr) − tr(H_nn⁻¹H_nrD⁻¹H_rn) + tr(H_nn⁻¹DH_nn[d]) = tr(D⁻¹Schur_rr(H)) + tr(H_nn⁻¹DH_nn[d]).

(d) **log|S_λ|₊.** ker(λSₖ + S_rest) = U_n·ker(S_rest,nn) × {0}, because S_rest ⪰ 0 forces S_rest,rn·ker(S_rest,nn) = 0. So rank(S_λ) = r + rank(S_rest,nn) for every λ > 0, and restricting to the range gives the same Schur expansion:

log|S_λ|₊ = r log λ + log|D| + log|S_rest,nn|₊ + t·tr(D⁻¹Schur_rr(S_rest)) + O(t²).

No rank hypothesis is needed.

(e) **Cancellation.** The ½(r log λ + log|D|) of (c) and (d) cancel identically, so every remaining term is analytic in t. ∎

*Interpretation.* The first two terms of aₖ are "expected minus observed" score curvature for the removed component. aₖ < 0 exactly when the face score g_r is larger, in the D⁻¹ metric, than its Laplace-expected size. This is the LAML analogue of the REML score test for a variance component at zero. That the point mass at the boundary is large is classical (Crainiceanu & Ruppert 2004, §2–3; Self & Liang 1987 for boundary asymptotics). For non-Gaussian families the W′ term adds a third-derivative correction. On the replica it is −2.1e-5 for coordinate 2 but −0.128 for coordinate 3, so it is not negligible in general.

### Result 4 (when is the infimum attained at a face; the face-reduced problem)

Compactify each coordinate as ρₖ ∈ (−∞, +∞].

- The chart at +∞ is tₖ = e^{−ρₖ} ≥ 0 (Result 3).
- The chart at −∞ is λₖ ≥ 0, used only when range(Sₖ) ⊆ Σ_{j≠k}range(Sⱼ) ("overlapping"). Otherwise V → +∞ as ρₖ → −∞; see Result 5(a).

Let F be a set of face coordinates that is **range-compatible**: the U_r-projectors of the coordinates in F commute with each other and with those of the remaining penalties. Disjoint blocks, and the wiggle/null pair of a double penalty, are compatible. Then V is jointly analytic in (t_F, ρ_{−F}), by applying Result 3 coordinate by coordinate: the Schur steps commute.

At a point P = (t_F = 0, ρ_{−F}*):

(i) **First-order (KKT).** P is a KKT point of min V on the compactified domain iff ∇_{−F}V_face(ρ_{−F}*) = 0 and aₖ(P) ≥ 0 for all k ∈ F.

(ii) **Second-order sufficient.** If moreover aₖ > 0 for all k ∈ F (strict complementarity) and ∇²_{−F}V_face ≻ 0, then P is a strict local minimizer. There is no minimizer of V in ℝ^M in a neighbourhood of P, the infimum of V there equals V_face(ρ_{−F}*), and it is attained only on the compactified domain.

*Proof of (ii).* Taylor-expand V = V_face* + Σ_{k∈F}aₖtₖ + ½δᵀ∇²δ + O(|t||δ| + |t|² + |δ|³). This is strictly larger than V_face* for small (t ≥ 0, δ) ≠ 0.

(iii) **Face strictly inferior.** If some aₖ < 0, moving into the interior along tₖ strictly decreases V, so P is not a local minimizer.

(iv) **Degenerate.** If aₖ = 0, the second t-derivative decides. This case is non-generic; see Open problems.

(v) **Global statement.** Under Result 1's condition, and assuming every −∞ end is either coercive (Result 5a) or compactified by λₖ, V attains its minimum on the compact closure. The global minimizer lies on a +∞ face, i.e. LAML "attains its infimum at a face" and has no minimizer in ℝ^M, iff min over faces of V_face ≤ inf over ℝ^M of V. Locally, the exact test is (ii).

**The face-reduced problem** is the same LAML problem for the model with the columns X U_r (for every k ∈ F) deleted and penalties U_nᵀSⱼU_n (j ∉ F). It is the problem the fit should be certified on and reported for: the smooth s(pc2) keeps only its null-space (linear) part. For a double-penalty smooth whose wiggle coordinate is at +∞, the remaining null coordinate is an ordinary interior variable.

### Result 5 (the −∞ side, separation, unsupported directions)

(a) **Identifiable, non-overlapping** (range(Sₖ) ⊄ Σ_{j≠k}range(Sⱼ), and X has full rank on range(Sₖ) with no separation there). As λₖ → 0 the limit problem exists, so log|H| stays bounded while log|S_λ|₊ contains rₖ'ρₖ with rₖ' ≥ 1. Hence V = −½rₖ'ρₖ + O(1) → +∞ linearly. It is coercive, and there is no face.

(b) **Overlapping.** range(Sₖ) ⊆ Σ_{j≠k}range(Sⱼ). Then log|S_λ|₊ is analytic in λₖ at λₖ = 0, and so are β̂ and H. V is therefore analytic in λₖ ∈ [0, ε). The −∞ face is a KKT minimum iff bₖ := ∂V/∂λₖ|₀ ≥ 0, with

bₖ = ½β̂ᵀSₖβ̂ + ½tr(H⁻¹(Sₖ + C[−H⁻¹Sₖβ̂])) − ½tr(S_{λ,−k}⁺Sₖ),

and it is strict if bₖ > 0.

(c) **Separation in a penalized direction.** Suppose u ∈ range(Sₖ) separates. In the 1-D reduction, −ℓ(βu) ≈ C e^{−mβ} with margin m. Stationarity gives Cm e^{−mβ̂} = λβ̂, so β̂ = m⁻¹log(1/λ) + O(log log). Then H = λ(1 + mβ̂) and f(β̂) → 0, so

V = ½log(1 + mβ̂) + O(1) = ½log|ρ| + O(1) → +∞, with ∂V/∂ρ ≈ ½ρ⁻¹.

V is coercive but only logarithmically. A ρ-space search sees slopes O(1/|ρ|) and a PIRLS whose W → 0 on the separated rows. The quasi-complete case, where tied observations sit on the separating hyperplane, has the same leading order. The tied rows have xᵀu = 0, so they add only a constant log 2 per row. This is verified in 40- and 120-digit arithmetic in §4, which also shows that the O(1) term converges slowly, at rate O(log|ρ| / |ρ|), since mβ̂ = |ρ| − log|ρ| + O(1).

(d) **Unsupported direction** (X U_r = 0 and S_rest,rn = 0). β̂_r ≡ 0, and log|H| − log|S_λ|₊ does not depend on ρₖ, so V is exactly constant in ρₖ. Both ∂V/∂ρₖ and ∂²V/∂ρₖ² vanish identically: the ρ-Hessian is singular and aₖ = 0. The coordinate is non-identified and must be quotiented out. It is not a failure of convergence. When X U_r is small but nonzero, aₖ ≈ 0⁺ and the face is nearly degenerate.

### Result 6 (the #784 block-local correction is not a C¹ criterion)

Δ_b in `crates/gam-solve/src/reml/block_quadrature_correction.rs` works in three steps:

1. It selects the top-m eigen-directions of sym(H) by |γ_r|, the standardized third-derivative skewness along eigenvector e_r (≈ lines 340-400).
2. It integrates each axis by 1-D Gauss–Hermite.
3. It adds a mixed-axis Laplace term Φ (`mixed_axis_laplace_term`, line 1255).

Three defects follow.

(i) **Discontinuous.** On the set {|γ_{(m)}| = |γ_{(m+1)}|}, which is codimension 1 in ρ, the selected block changes. Δ_b jumps by the difference of the two axes' 1-D corrections, and that difference is generically nonzero.

(ii) **Undefined at eigenvalue crossings.** Where eigenvalues of H coincide (codimension 2 for real symmetric families; von Neumann & Wigner 1929), the eigen-axes are not unique. An axis-by-axis quadrature is not invariant under rotations within the degenerate eigenspace, so Δ_b has no limit there. The code refuses these points (`EigenframeNearDegeneracy`, line 931).

(iii) **Gradient known to be wrong.** The gradient channels are documented as a *wrong contraction*: "MEASURED (#2623) … NEITHER SIGN … 1e-4 to 1.3e-1 relative … INVERTS the search" (lines 1002-1030). The ρ-Hessian is `None` (≈ line 1060).

Consequently the latched criterion LAML − Δ_b violates the hypotheses of every line-search convergence theorem: C¹ with a Lipschitz gradient and gradient–value consistency (Nocedal & Wright 2006, Thm 3.2 Zoutendijk, and §6.1 for BFGS). `StepSizeTooSmall` is the predicted outcome, not an accident.

The comment at `gradient_hessian.rs:173-176`, which says Δ_b comes "with its exact gradient", contradicts the measurement at `block_quadrature_correction.rs:1002-1030`.

By contrast, the full-tensor second-order Laplace correction (Tierney & Kadane 1986; Tierney, Kass & Kadane 1989; Shun & McCullagh 1995, eq. for the O(n⁻¹) term),

δ₁ = −⅛ f_{ijkl}H^{ij}H^{kl} + ⅛ f_{ijk}f_{lmn}H^{ij}H^{kl}H^{mn} + (1/12) f_{ijk}f_{lmn}H^{il}H^{jm}H^{kn},

is a full contraction and therefore invariant under any linear reparametrization of β. It is real-analytic wherever Result 1 holds, and its ρ-derivatives follow from the same drifts as §2. For GLMs, f_{ijk} = Σᵢcᵢxᵢ⊗xᵢ⊗xᵢ, so the contractions reduce to row-pair sums, which is exactly the existing `TkRowPairRoute`.

---

## 4. Numerical checks

The replica uses the prostate train split (rows i%4 ≠ 0, n = 490). Each smooth is a cubic B-spline with k = 5, sum-to-zero centred via QR. The wiggle penalty is a second-difference P-spline penalty of rank 3; the null-space penalty is the projector onto the linear direction, of rank 1. With the intercept, p = 9 and M = 4. The ρ order is (pc1 wiggle, pc1 null, pc2 wiggle, pc2 null).

| check | result |
|---|---|
| gradient with W′ vs 40-digit mpmath central FD (h = 1e-12) at ρ = (1, −1, 3, −2) | max rel. err **2.9e-12** |
| gradient without the C[vₖ] drift | max rel. err **5.6e-2** (entries 0.5149 vs 0.5456, 3.582 vs 3.663) |
| analytic ρ-Hessian (with W″) vs Richardson FD of the analytic gradient | max abs err **2.9e-10** (‖H‖ = 2.85) |
| ρ-Hessian eigenvalues at ρ = 0 | (−3.66, −0.139, 0.055, 0.891): **indefinite** |
| exact Newton from ρ = 0 | ρ₀,₁,₃ → (−2.186, −3.084, −4.667) with |g| ≤ 1e-9; **ρ₂ → +∞** with V → 309.48475718 |
| face coefficients at the end point, (score, trace, W′, overlap) | a₀ = −0.1009, a₁ = −0.0105, **a₂ = +0.04594** (−0.0322, +0.0782, −2.1e-5, 0), a₃ = −29.5 |
| −λ₂∂V/∂ρ₂ at ρ₂ = 6, 10 (exact-zero chart) | +0.045938, +0.045931 → a₂ = 0.045937 ✓ |
| same quantity at ρ₂ = 14, 18, 22.73 in the as-built (leaky) chart | 0.0172, −30.3, −9.8e5: **garbage** (sign flips) |

**Leakage experiment** (`face_stability.py`, ρ₂ = 22.73, other coordinates at the replica optimum):

| representation | ∂V/∂ρ (0, 1, 2, 3) | abs err vs truth | V jaggedness |
|---|---|---|---|
| as-built Sₖ = ZᵀPZ (leaky), naive H | (−1.28e-5, −9.74e-6, 1.81e-4, −1.06e-5) | 6e-6, 9e-6, 5e-5, 6e-6 | **2.2e-5** |
| exact-zero chart, naive H (κ = 1.3e12) | (−1.257e-9, 4.003e-9, −6.16e-12, 2.686e-9) | ≤ 1.4e-15 | 1.5e-11 |
| face chart β_r = √t γ (κ = 1.2e3) | same | ≤ 2.2e-15 | 1.7e-13 |
| truth (exact-zero chart, 40 digits) | (−1.257e-9, 4.003e-9, −6.159e-12, 2.686e-9); −a₂t = −6.159e-12 ✓ | — | — |

The as-built value V differs from the true V by 1.9e-4. `leak.py` measures u_nᵀ(ZᵀPZ)u_n = 3.7e-16 and u_nᵀfl(RᵀR)u_n = 8.1e-17 against ‖S‖ = 11. Multiplied by λ = e^{22.73}, these give spurious ridges of 2.8e-6 and 6.1e-7. The rule λₖ·u·‖Sₖ‖ ≥ τ = 1e-8 is first met at ρₖ ≈ 15.2. Past that point the ρ-chart gradient of *every* coordinate is dominated by the leak. The failing tests sit at 22.73 (B-rail) and 20.73 (B-pymc), 1.3e3× and 1.8e2× past it.

SEPARATION_AND_MULTISTART_PLACEHOLDER

---

## 5. Literature

- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B* 73(1), 3–36.
  - §2–3 define LAML and its Newton optimization, including the third- and fourth-derivative terms of the ρ-gradient and ρ-Hessian for non-canonical and canonical links.
  - §3.1 and Appendix B give a stable log|S|₊ via a similarity transform that isolates dominant penalties. This is the published version of the "exact-zero chart" of §6.
- Wood, S. N., Pya, N. & Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *JASA* 111(516), 1548–1563. §3 gives the general derivative formulae for LAML beyond exponential families.
- Reiss, P. T. & Ogden, R. T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *JRSS-B* 71(2), 505–523. REML versus GCV, including multiple local optima and REML's lower propensity for them.
- Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear mixed models with one variance component. *JRSS-B* 66(1), 165–185. The large point mass at the boundary of the (RE)ML variance-component estimate. Result 3 is its LAML analogue.
- Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA* 82(398), 605–610.
- Albert, A. & Anderson, J. A. (1984). On the existence of maximum likelihood estimates in logistic regression models. *Biometrika* 71(1), 1–10.
- Haberman, S. J. (1974). *The Analysis of Frequency Data*. University of Chicago Press, Ch. 2: existence of the MLE in log-linear (Poisson) models.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80(1), 27–38.
- Tierney, L. & Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. *JASA* 81(393), 82–86.
- Tierney, L., Kass, R. E. & Kadane, J. B. (1989). Fully exponential Laplace approximations to expectations and variances of nonpositive functions. *JASA* 84(407), 710–716.
- Shun, Z. & McCullagh, P. (1995). Laplace approximation of high dimensional integrals. *JRSS-B* 57(4), 749–760.
- von Neumann, J. & Wigner, E. (1929). Über das Verhalten von Eigenwerten bei adiabatischen Prozessen. *Phys. Z.* 30, 467–470: eigenvalue crossings of real symmetric families have codimension 2.
- Kato, T. (1995). *Perturbation Theory for Linear Operators*, 2nd ed. Springer, Ch. II: analytic eigenprojections, which are not analytic eigenvectors at crossings.
- Krantz, S. G. & Parks, H. R. (2002). *The Implicit Function Theorem*. Birkhäuser: the real-analytic IFT.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM J. Control Optim.* 20(2), 221–246. Newton on the orthant tₖ ≥ 0.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Thm 3.2 (Zoutendijk) and §6.1 (BFGS).
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3 (running error bounds) and §10 (Cholesky).

---

## 6. Consequences for gamfit

Paths are relative to the repository root.

### 6.1 Delete the #784 block-local correction from the outer criterion (addresses L-link, and B-rail/B-pymc when latched)

It is not C¹ (Result 6), its gradient is measured wrong, and it has no Hessian. A certified optimum of a discontinuous criterion is not a well-defined object. Remove:

- `crates/gam-solve/src/reml/objective.rs:2233-2250`: the `block_local_quadrature_correction` splice as `"sampled_block_marginal"` and the `HessianValue::Unavailable` override at 2247.
- `crates/gam-solve/src/reml/objective.rs:2359-2374`: the same in `assemble_and_evaluate_efs`.
- `crates/gam-solve/src/reml/eval.rs:993`: the latched-Hessian override.
- `crates/gam-solve/src/reml/gradient_hessian.rs:173-177`: the `!self.block_correction_latched()` gate, together with its incorrect "exact gradient" comment.
- `crates/gam-solve/src/estimate/optimizer.rs:1236`, `1343`, `1399`, `1455`, `1519`, `1773`, `2380-2383` and `2444-2447`: `defer_block_correction_admission`, `decide_block_correction_admission` and the `corrected_continuation` BFGS restart.
- `crates/gam-solve/src/reml/block_quadrature_correction.rs`: the whole outer-criterion use. The directional-cubic diagnostic in `gam-inference/src/hmc_io.rs` may remain as a *post-fit report* ("Laplace trustworthiness"). It must not feed back into ρ̂.

If a beyond-Laplace criterion is required, use δ₁ of Result 6 (the full-tensor TK). It is smooth and basis-invariant, and its gradient and Hessian are exact from the same drift quantities. `tierney_kadane_terms` (`gradient_hessian.rs:2210`) already implements a TK atom, currently gated on Firth (`state_caches.rs:1031`). Its exact Hessian is canonical-logit-only (`gradient_hessian.rs:170-173`); the non-canonical case needs the fourth η-derivative, i.e. W‴ in observed information. With Δ_b removed, `analytic_outer_hessian_enabled` returns true for every non-Firth fit, and mgcv CI logistic_link gets its smoothing-corrected covariance back.

### 6.2 Replace the hand ρ-box with the compactification (addresses B-rail and B-pymc)

The box `[−17.36, 22.73]` (`crates/gam-solve/src/rho_optimizer/rail.rs` and its producers) and the margin rail test are forbidden hand bounds. They are also harmful here: the upper face of the box lies far inside the region where the ρ-chart value is corrupted (§4).

- **Outer variables.** Each coordinate lives in (−∞, +∞] with the atlas {ρₖ} ∪ {tₖ = e^{−ρₖ} ∈ [0, ∞)}. The bound tₖ ≥ 0 is a genuine point of the compactified domain, not a hand bound. On overlapping −∞ ends, add the λₖ ∈ [0, ∞) chart.
- **Search.** Use projected Newton with the analytic Hessian (Bertsekas 1982) in (ρ_free, t_F). A coordinate enters F when its Newton step in t would cross 0, which is the exact active-set rule of projected Newton. No threshold constant is needed.
- **Evaluation near a face.** Use the face chart of `face_stability.py`: β = U T θ with T = diag(I_n, √t I_r). The scaled design X U_r√t and the fixed penalty D replace λSₖ, so **λₖ never multiplies a floating-point Sₖ**, and log|H| − rₖρₖ = log|H̃| holds exactly. The trace-minus-rank term becomes −t·tr((D + tK)⁻¹K) with K = Schur_rr(H − Aₖ): O(1) quantities times t, with no cancellation.
- **Chart choice.** Both charts are exact, so the choice affects only rounding. Use the chart with the smaller condition number of the assembled matrix: the face chart iff λₖ‖D‖ > ‖X_rᵀWX_r + S_rest,rr‖. This is a derived balance point, not a tuning constant.
- **Structural zeros.** The root `PenaltyCoordinate::{DenseRoot, BlockRoot}` (`crates/gam-problem/src/penalty_coordinate.rs:71-90`) is formed as fl(RᵀR)·λ in `scaled_block_local` (`penalty_coordinate.rs:537-556`). That product leaks u·‖Sₖ‖ onto ker Sₖ, and λₖ amplifies it: this is the measured failure mechanism. Store each coordinate in its **eigen-chart with exact zeros** (Wood 2011 §3.1 similarity transform): rotate the block basis once so that Sₖ = diag(0, D). The existing #2331 fused rank cancellation (`reml_outer_engine/objective.rs:1032-1198`) removes only *arithmetic* cancellation. It cannot remove this *representation* error, as §4 shows: the exact-zero chart with a naive κ = 1.3e12 H is already accurate to 1e-15.

### 6.3 Certificate (replaces the ρ-space |Pg| test at a rail)

At the returned point with face set F, all of the following must hold:

1. `‖∇_{−F} V_face‖ ≤ τ_g`, computed in the reduced model.
2. For every k ∈ F, `aₖ > τ_a,k`, using the closed form of Result 3. It includes the W′ term and costs one reduced PIRLS solve plus O(np²).
3. The Cholesky factorization of `∇²_{−F} V_face` succeeds. This is the analytic Hessian of the reduced LAML, with W″.
4. The reported model is the face-reduced model: the columns X U_r of each k ∈ F are dropped, and edf and covariance are computed without them.

Face coordinates are reported as λₖ = ∞ ("fully smoothed to the null space"), not as ρₖ = 22.73.

### 6.4 Derived tolerances (no magic constants)

- **τ_g.** In the face chart, bound the forward error of each gradient entry with a running error bound (Higham 2002 §3). With γ_q = qu/(1 − qu), q = O(np), and u the unit roundoff,

  τ_g = γ_{np}·(½|β̂ᵀAⱼβ̂| + ½Σ|tr terms| + ½rⱼ)·κ(H̃).

  On the replica this is ≈ 1e-13, while the measured error is 2e-15. The "solver-band" bound 7.3e-6 in the failing log is 7 orders of magnitude looser than necessary. It still fails, because the ρ-chart error at the rail, ≈ λₖu‖Sₖ‖·‖∂β̂‖, is larger than it.
- **τ_a,k.** Use the same running bound applied to the four terms of aₖ. If |aₖ| ≤ τ_a,k, the face is degenerate: go to the second-order test in t (Open problem 1). Never round it to either side.
- **Sanity identity (tests only).** −λₖ∂V/∂ρₖ → aₖ, checked at two moderate λₖ in the exact-zero chart, as §4 does.

### 6.5 Separation and Firth (`gam-models/src/fit_orchestration/fit.rs:533-600`)

By Result 1, Firth is needed only for separation in N_S (unpenalized directions). Separation in penalized directions gives V ~ ½log|ρ|: coercive but flat. A ρ-space BFGS can drift to ρ ≈ −10² while β̂ ≈ |ρ|/m. That regime calls for an exact Newton with the analytic Hessian. It does not call for Firth.

### 6.6 Mapping to the failing clusters

| cluster | mechanism | fix |
|---|---|---|
| B-rail (ebm, sklearn k=5; railed #2 at 22.73; |Pg| 2.28e-5; StepSizeTooSmall after 7 iterations; BFGS; `curvature_source=unavailable`) | a₂ > 0 face, so V is flat along ρ₂ and BFGS runs to the box. At the box, λ-amplified leak noise in g (~1e-5) and V (~2e-5) stops the line search. If the #784 latch fired, it also removed the Hessian and made the criterion discontinuous. | 6.1 + 6.2 + 6.3 |
| B-pymc (nuts/hmc; default k; ρ₂ = 20.73 unrailed; |Pg| = 9.6e-6) | Same face; the leak noise at ρ₂ = 20.73 (λu‖S‖ ≈ 1e-6 × O(1–10)) exceeds the bound. | 6.2 + 6.3 |
| L-link (Gaussian, logit link) | Latched #784: no analytic ρ-Hessian, so no smoothing-corrected covariance. | 6.1 |

---

## 7. Open problems

1. **Degenerate faces (aₖ = 0 within τ_a).** The second t-derivative Ṽ″(0) needs the O(t²) Schur terms and a W″ term. The closed form has not been derived here. It is non-generic, but "unsupported direction" data (Result 5d) sit exactly on it.
2. **Non-compatible face sets.** When the U_r projectors of two face coordinates do not commute (e.g. tensor-product marginal penalties with overlapping ranges both at +∞), joint analyticity in (t_k, t_l) is not proved. The limit may depend on the direction of approach t_k/t_l.
3. **Non-canonical links.** Result 3 holds with W replaced by the observed information, which includes (y − μ)-dependent terms, and c, d replaced by its η-derivatives. This report did not verify that the gamfit inner solve uses observed rather than expected information for non-canonical links. If it uses Fisher weights, the IFT drift ∂β̂/∂ρ = −H⁻¹Aβ̂ is inconsistent, and so is the whole gradient. This needs a check like `replica.py` on a probit or Gaussian-logit replica.
4. **Multiple minima across faces.** Result 4(v) reduces the global problem to comparing finitely many face-reduced LAMLs, one per compatible face set. No bound on how many faces must be examined, other than 2^M, is known without extra structure.
5. **Separation plus face.** Separation in range(Sₖ) combined with Sₖ at a +∞ face gives g_r of order the separation margin, which makes aₖ very negative. Whether this interacts badly with Firth rescue (which changes f) is unexamined.
