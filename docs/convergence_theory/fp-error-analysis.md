# Floating-point error analysis of the LAML/REML cost, gradient and Hessian

Lane: `fp-error-analysis` (convergence theory team). The scripts are in
`SP/theory/fp-error-analysis/`, where
`SP = /tmp/claude-0/-home-user-gam/02aeec89-32a7-52a0-8d71-90f383516996/scratchpad`.
Each result is labelled with its status: **[proved]**, **[derived to first order]**, **[checked numerically]** or **[conjecture]**.

---

## 1. Summary

1. **The rails failure is noise from the eigendecomposition route.** In gamfit's binomial-logit failure, |Pg| = 2.28e-5 at ρ₂ = 22.7. This is below the forward-error band of the route that produced it. The route takes a dense eigendecomposition of the assembled H (`dense_spectral.rs`, `laml_logdet.rs` header: ‖H‖ = 6.2e11). For that route the predicted band on the free coordinates is 3.5e-5 to 4.2e-4. In a replica, BFGS+Armijo on the same route stalls in the same way (StepSizeTooSmall, |g| = 1.3e-4). The certification bound 7.30e-6 = n·√ε (`run.rs:8702`) has no relation to either the true error or the achievable accuracy. **[checked numerically]**
2. **Error laws for the eigendecomposition route.** On the assembled H, a normwise backward error ‖E‖ ≤ c_p u‖H‖ ∝ λ_max gives the following. **[proved]**
   - Value: |δ log|H|| ≤ ‖E‖ tr(H⁻¹).
   - Gradient trace: |δ tr(H⁻¹λ_jS_j)| ≤ ‖E‖·λ_j‖R_jH⁻¹‖²_F.
   - The stiff coordinate's own gradient is λ-free, but every *soft* coordinate is polluted by u·λ_max‖S_max‖. Errors therefore grow like e^{ρ_max}: 1.4e-10 at ρ₂ = 12, 1.6e-6 at 22.7, and 6.9e-4 at 30.
3. **Cholesky in an aligned frame is λ-free.** Cholesky has a *componentwise* backward error (Higham Thm 10.3). With van der Sluis scaling this gives |δ log|H|| ≤ p γ_{p+1} tr(H̃⁻¹)/(1−η), where H̃ = D^{-1/2}HD^{-1/2} and D = diag(H). In an aligned frame κ(H̃) stays bounded as λ → ∞: it is 2.1 at the checkpoint, where κ(H) = 3.2e11. Measured: V error ≤ 1e-13 and gradient error ≤ 2e-15 for all ρ₂ ∈ [0, 30] and for spans up to e^{±30}. The same BFGS converges to |g| = 1.9e-14. **[proved + checked]**
4. **Representation leakage.** Any frame in which a penalty's null space is only approximately representable adds a spurious penalty of about u‖S_j‖λ_j on the "null" direction, because it forms QᵀSQ or reassembles EᵀE numerically. The resulting error scales like e^{ρ_j}. Cholesky and QR suffer it equally, so it is a representation error, not a factorization error: 1e-4 gradient error at ρ₂ = 22.7 in the raw frame, against 1e-16 in the aligned frame. **Structural zeros**, meaning exact block structure in the reparameterized frame, are the remedy. **[proved to first order + checked]**
5. **The penalty log-det is exactly affine when the penalty ranges are independent.** Then log|S_λ|_+ = Σ_k r_kρ_k + log det(R̄R̄ᵀ), so ∂/∂ρ_k = r_k *exactly* and the Hessian is 0. This covers single penalties, double (null-space) penalties and disjoint smooths. **[proved]**
6. **Tensor-product (Kronecker-sum) penalties have a cancellation-free closed form.** log|S_λ|_+ = Σ_{non-null multi-index} log(Σ_kλ_k a^{(k)}_{i_k}). This is exact to 2e-13 at spans of e^{±30}. **[proved + checked]**
7. **General overlapping penalties.**
   - Stacked-root SVD (`penalty_logdet.rs:770`) obeys a **√κ law**: error ≈ c·u·r·√(μ_max/μ_min). It gives 5.6e-3 at e^{±30}.
   - Assembled eigendecomposition with the 100pε threshold (`pseudo_logdet.rs:26`) misclassifies the rank beyond about e^{27}, giving log-det errors of 10 to 289.
   - Row-sorted, column-pivoted Householder QR followed by one-sided Jacobi, with the exact identity Σ_k tr(S⁺λ_kS_k) = r used for the largest trace, is accurate to 1e-13 at e^{±30}. **[checked; relative-accuracy theory from Demmel et al. 1999; the λ-free claim for general overlap is a conjecture]**
8. **Certificate.** With a rigorous per-coordinate band |ĝ_j − g_j| ≤ ε_{g,j}, the test |Pĝ|_j ≤ max(ε_{g,j}, τ_j − ε_{g,j}) is the tightest test that always passes at an exact stationary point. Passing guarantees |Pg|_j ≤ max(2ε_{g,j}, τ_j), where τ_j is the statistical tolerance from the statistical-resolution lane. A multiplier c = 1 on ε is *derived*: it is the minimal c that is sound against adversarial rounding. **[proved]**
9. **Line-search stall law.** Once gᵀ𝓗⁻¹g < 4ε_V/(1−c₁)², Armijo may fail for every step (Hager–Zhang 2005, §4). This is exactly the StepSizeTooSmall signature. The cure is an accurate route; the band itself then derives the approximate-Wolfe slack. **[proved]**
10. **`decrement_bands.rs:86` omits the factorization term.** Its gradient band has no factorization term. It charges the whole IFT correction |kkt| as error instead of the O(‖r‖²) remainder, and its Hessian band γ_m‖𝓗‖_F has no factorization content. Section 6 lists every magic tolerance that the derived formulas replace.

---

## 2. Setup and notation

- u = 2⁻⁵³ is the unit roundoff and γ_k = ku/(1−ku) (Higham 2002, Lemma 3.1). ρ ∈ ℝ^K and λ_k = e^{ρ_k}.
- The penalties are S_k = R_kᵀR_k ⪰ 0, with rank r_k. S_λ = Σ_kλ_kS_k, A_k = λ_kS_k, and r = rank S_λ.
- For the binomial logit: η = Xβ, μ = σ(η), w = μ(1−μ), w′ = w(1−2μ), and H = XᵀWX + S_λ at the inner mode β̂.
- LAML:

  V(ρ) = Σ_i ℓ_i(β̂) + ½β̂ᵀS_λβ̂ + ½log|H| − ½log|S_λ|_+,  ℓ_i = −y_iη_i + log(1+e^{η_i}).

- Exact gradient, via the IFT dβ̂_j = −H⁻¹A_jβ̂:

  g_j = ½β̂ᵀA_jβ̂ + ½tr(H⁻¹A_j) + ½tr(H⁻¹Xᵀdiag(w′⊙Xdβ̂_j)X) − ½tr(S_λ⁺A_j)

  The four terms are q_j, th_j, tift_j and ts_j.
- 𝓗 is the outer Hessian ∂²V/∂ρ².
- **Routes** for log|H| and H⁻¹:
  - **E**: symmetric eigendecomposition of the assembled H. This is the dense_spectral kernel and the laml_logdet assembled spectrum.
  - **C**: Cholesky of the assembled H.
  - **Q**: Householder QR of the stacked root [W^{1/2}X; √λ_1R_1; …].
- **Frame.** A frame is an orthogonal coefficient basis Q; everything is expressed in Q.
  - A frame is **aligned** if, for every k, the stored in-frame penalty Ŝ_k has *exact* zeros in the null-space rows and columns of S_k.
  - More generally, the stored matrices must have the same range/null structure as the exact S_k. This is structural zeros, not small numbers.
- D = diag(H) and H̃ = D^{-1/2}HD^{-1/2}, the unit-diagonal (van der Sluis) scaling.
- **Replica** (`replica.py`):
  - Data: prostate train rows (n = 490), binomial logit.
  - Model: intercept plus two cubic B-spline smooths, 5 basis functions each (4 after centering), p = 9.
  - Four penalties per smooth pair, as [wiggle, null] × 2 with ranks 3, 1, 3, 1. This is gamfit's double-penalty layout.
  - Checkpoint: ρ* = (2.807, −2.341, 22.733, −3.030), i.e. gamfit's failing `rho_checkpoint` shape, with coordinate 2 railed.
  - The mpmath reference uses 60 digits (`ref_checkpoint.py`): V = 307.8196890435107 and g = (1.3603e-2, 0.68435, 1.46e-12, 1.45703).

---

## 3. Results

### 3.1 Value: log|H|

**Theorem 1 (eigendecomposition route E). [proved]** Let the computed eigenvalues μ̂ be the exact eigenvalues of H+E with ‖E‖₂ ≤ c_p u‖H‖₂. This holds for backward-stable symmetric eigensolvers (Golub & Van Loan 2013, §8.3). Let η_E = ‖E‖₂‖H⁻¹‖₂ < 1. Then

  |Σ log μ̂_i − log|H|| ≤ ‖E‖₂ tr(H⁻¹)/(1−η_E).

*Proof.* log|H+E| − log|H| = Σ log(1+ν_i), where ν_i are the eigenvalues of M = H^{-1/2}EH^{-1/2}. Each |ν_i| ≤ η_E, and |log(1+ν)| ≤ |ν|/(1−η_E). Also Σ|ν_i| = ‖M‖_* ≤ ‖E‖₂ tr(H⁻¹), because |tr(PB)| ≤ ‖B‖₂ tr P for P ⪰ 0 (apply it to the polar factor). ∎

**Consequences.**
- ‖H‖₂ ≈ λ_max‖S_max‖, so the bound grows like e^{ρ_max}.
- It becomes void (η_E ≥ 1) once κ(H) ≳ 1/(c_p u). Beyond that point computed eigenvalues can be negative.
- `dense_spectral.rs:1249` `logdet_forward_error` implements exactly p·ε·max|μ|·Σ1/|μ_i|, i.e. c_p = p. So it is **the correct band for route E**. The trouble is that route E is intrinsically inaccurate at large λ.

**Theorem 2 (Cholesky route C). [proved]** Suppose Cholesky runs to completion on H, giving L̂L̂ᵀ = H + ΔH with |ΔH| ≤ γ_{p+1}|L̂||L̂ᵀ| (Higham 2002, Thm 10.3). Let the formation error of H be componentwise, |δH_form| ≤ φ|X|ᵀW|X| + ψ|Ŝ_λ|, with φ = γ_n and ψ = γ_K (plus rounding of the in-frame penalty entries). Define

  δ̃ = p(γ_{p+1}(1+O(u)) + γ_n + γ_K),  η = δ̃‖H̃⁻¹‖₂ < 1,  T̃ = tr(H̃⁻¹) = Σ_i h_ii (H⁻¹)_ii.

Then

  |2Σ log l̂_ii − log|H|| ≤ δ̃ T̃/(1−η) + 2γ_p Σ_i|log l̂_ii| + 2p u.

The last two terms are the log evaluation and the summation.

*Proof.*
1. Σ_j |l̂_ij|² = h_ii + ΔH_ii, so by Cauchy–Schwarz (|L̂||L̂ᵀ|)_ij ≤ √(h_ii h_jj)(1+O(u)).
2. Hence every entry of D^{-1/2}ΔHD^{-1/2} is at most γ_{p+1}(1+O(u)), and the spectral norm of that p×p matrix is at most p·γ_{p+1}(1+O(u)).
3. The same Cauchy–Schwarz step bounds the entries of D^{-1/2}|X|ᵀW|X|D^{-1/2} and D^{-1/2}|Ŝ_λ|D^{-1/2} by 1. Here S_λ ⪰ 0 and the diagonal of XᵀWX is at most h_ii.
4. Apply the argument of Theorem 1 to H̃ with perturbation Δ̃ = D^{-1/2}(ΔH+δH_form)D^{-1/2}. The ratio of determinants is scale-invariant. ∎

**Proposition 3 (the scaled condition number is λ-free in an aligned frame). [proved]** In an aligned frame, write

  H = [[B + λΣ, C], [Cᵀ, E]],

where Σ ≻ 0 is the stiff penalty's range block and λ → ∞. Then

  H̃ → blockdiag(corr(Σ), corr(E)) + O(λ^{-1/2}),

so κ(H̃) stays bounded. The off-diagonal scaled block is D_s^{-1/2}CD_e^{-1/2} = O(λ^{-1/2}). By van der Sluis (1969), κ(H̃) ≤ p·min_{diagonal Δ} κ(ΔHΔ).

Numerically at ρ* (`exp_routes.py`): κ(H) = 3.2e11, κ(H̃) = 2.07, T̃ = 9.27 and tr(H⁻¹) = 5.23. The predicted |δlog|H|| is 2.15e-4 for route E and 4.6e-14 for route C.

**Route Q. [derived]** A plain Householder QR of the stacked root is only normwise stable. The R factor then carries an absolute error of u·‖stack‖ = u√‖H‖, which gives a √κ(H) law. With **row sorting by decreasing norm plus column pivoting**, Householder QR is row-wise backward stable (Cox & Higham 1998; Higham 2002 §19.4). The rows √λ_kR_k are then perturbed *relatively*, which is a λ-free perturbation of each R_k, and the log-det error is λ-free as for C. In the aligned replica frame even unsorted QR is λ-free, because the structure is block.

### 3.2 Value: the other channels

**Deviance, [proved to first order].**

  |δΣℓ_i| ≤ (γ_n + c_ℓu)Σ|ℓ_i| + Σ_i|μ_i−y_i|·γ_p(|x_i|ᵀ|β̂|).

Here c_ℓ ≈ 3 is the ulp count of a correctly implemented softplus. The second term is the rounding of η = Xβ̂ propagated through ∂ℓ/∂η = μ−y.

**Penalty quadratic, [proved].** Computed as Σ_kλ_k‖R_kβ̂‖², its error is at most

  ½γ_{p+1}Σ_kλ_k‖|R_k||β̂|‖²

Both factors are λ-benign: in the stiff directions β̂ = O(1/λ).

**Penalty log-det: Theorem 4 (affine penalty log-determinant). [proved]** Suppose range(S_1), …, range(S_K) are linearly independent, i.e. their sum is direct. Stack the full-row-rank roots R̄ = [R_1; …; R_K] (r×p, rank r = Σr_k) and let Λ = blockdiag(λ_kI_{r_k}). Then

  log|S_λ|_+ = Σ_k r_kρ_k + log det(R̄R̄ᵀ),  ∂_{ρ_k} = r_k,  ∂²_{ρ} = 0.

*Proof.* S_λ = R̄ᵀΛR̄. The nonzero eigenvalues of R̄ᵀ(ΛR̄) equal those of ΛR̄R̄ᵀ, which is r×r and nonsingular because R̄ has full row rank. So |S_λ|_+ = det(Λ)det(R̄R̄ᵀ). ∎

**Consequences of Theorem 4.**
- tr(S_λ⁺λ_kS_k) = r_k is an *integer identity*. No trace, eigendecomposition or threshold is needed. The only rounding in V is in log det(R̄R̄ᵀ), a λ-independent constant computed once (ρ = 0).
- It applies to a single penalty; to the double-penalty pair (wiggle ⊕ null-space shrinkage), whose ranges are complementary; and to any set of smooths on disjoint coefficient blocks.
- This removes every λ-dependent source of error in log|S|. The ρ₂ sweep (§4.3) shows the eigendecomposition-plus-threshold route producing V errors of 1.5–2.7 at ρ₂ ≥ 26, while the affine form is exact.

**Theorem 5 (tensor-product Kronecker sum). [proved]** Let S_k = I⊗…⊗A_k⊗…⊗I, with margin eigen-decompositions A_k = U_k diag(a^{(k)}) U_kᵀ. Then S_λ = (⊗U_k) diag(Σ_kλ_k a^{(k)}_{i_k}) (⊗U_k)ᵀ, whose eigenvectors are λ-independent, and

  log|S_λ|_+ = Σ_{i∉N} log(Σ_kλ_k a^{(k)}_{i_k}),  ∂_{ρ_j} = Σ_{i∉N} λ_j a^{(j)}_{i_j}/(Σ_kλ_k a^{(k)}_{i_k}).

Here N is the set of multi-indices with every i_k in the margin null. Each gradient summand lies in [0, 1], so there is no cancellation.

The error is λ-free: log of a positive sum has relative error at most γ_K. The margin eigenvalues a^{(k)} carry the λ-free, fixed-size margin conditioning; compute them to high relative accuracy by one-sided Jacobi on the margin root (Demmel & Veselić 1992).

The zero pattern N is **structural**. It is the dimension of the margin penalty's polynomial null space, which is a mathematical fact and not a tolerance.

*Caveat.* A sum-to-zero constraint Z applied to the whole tensor breaks the Kronecker structure. Centring each margin keeps it.

**Proposition 6 (√κ law of the stacked-root SVD). [derived + checked]** A backward-stable SVD of R_stack = [√λ_kR_k] has absolute singular-value errors at most c u σ_max. Therefore

  |δ log|S|_+| = |2Σ_i δσ_i/σ_i| ≲ 2c u Σ_i σ_max/σ_i ≤ 2c u r √(μ_max/μ_min),

where μ_i = σ_i² are the eigenvalues of S_λ on its range. Compare assembled eigendecomposition, whose error ~ c u r κ has the full κ rather than √κ.

A row-wise relative perturbation of size ϑ gives:
- η√(μ_max/μ_min) in the worst case;
- O(ϑ) + O(ϑ²μ_max/μ_min) when the penalties are structurally separated, as in a Kronecker sum.

The first two cases are proved; this last one is checked numerically.

**Algorithm R (relatively accurate log|S|_+ for general overlap).**
1. Sort the rows of R_stack by decreasing 2-norm.
2. Apply Householder QR with column pivoting (Cox–Higham: row-wise backward stable).
3. Apply one-sided (Hestenes) Jacobi to the triangular factor. Demmel et al. (1999) show that its singular values have relative accuracy governed by the condition number of the column-scaled factor, which pivoting keeps small.
4. log|S|_+ = 2Σlog σ_i over the structural rank r.
5. Compute the traces t_k = λ_k‖R_kVΣ⁻¹‖²_F directly for every k except the largest. Set the largest to r − Σ_{others}, using the exact identity Σ_k t_k = r.

Status: the identity is proved, and the accuracy (1e-13 at e^{±30}) is checked numerically. That it is λ-free for arbitrary overlapping penalties is a **conjecture**.

### 3.3 Gradient

**Theorem 7 (factorization error in the trace terms). [derived to first order; rigorous with the 1/(1−η) factor]** Let Ĥ = H + ΔH and let t_j = tr(H⁻¹A_j) be computed from Ĥ. Then δt_j = −tr(H⁻¹ΔH H⁻¹A_j) + O(‖H⁻¹ΔH‖²), and:

- **Route E:** |δt_j| ≤ ‖ΔH‖₂‖H⁻¹A_jH⁻¹‖_* = c_pu‖H‖₂ · λ_j‖R_jH⁻¹‖²_F.
- **Route C (aligned):** |δt_j| ≤ ‖Δ̃‖₂‖D^{1/2}H⁻¹A_jH⁻¹D^{1/2}‖_* = δ̃ · λ_j‖R_jH⁻¹D^{1/2}‖²_F.

Both use the fact that H⁻¹A_jH⁻¹ ⪰ 0, so its nuclear norm equals its trace.

**Push-through: the stiff coordinate's own trace error is λ-free.** Write H ⪰ λS + K. On range(S) the inverse behaves like (λS)⁻¹, so

  λ‖RH⁻¹‖²_F = λ tr(H⁻¹SH⁻¹) = tr(S⁺)/λ + O(λ⁻²).

The E-route error of the stiff coordinate is therefore about u‖S‖tr(S⁺), with no λ in it.

A *soft* coordinate k instead pays u·λ_max‖S_max‖·λ_k‖R_kH⁻¹‖²_F, which is proportional to e^{ρ_max}. This is how one rail pollutes every other coordinate.

At ρ*: λ_k‖R_kH⁻¹‖²_F = (0.19, 1.35, 4.35e-10, 0.115). The predicted E-route bounds are (7.8e-6, 5.6e-5, 1.8e-14, 4.7e-6), against observed errors of (−3.7e-7, −3.1e-6, 7.9e-16, −2.7e-7).

**IFT term.** tift_j = tr(H⁻¹M_j), where M_j = Xᵀdiag(w′⊙Xdβ̂_j)X is indefinite. Its factorization error has two parts:

  |δ| ≤ ‖Δ‖·‖H⁻¹M_jH⁻¹‖_* + |∂tift_j/∂dβ_j|·‖H⁻¹ΔH dβ̂_j‖.

The first uses Δ = ΔH for route E, or the D-scaled version for route C. For the nuclear norm use ‖H⁻¹M_jH⁻¹‖_* ≤ ‖H⁻¹Xᵀ|diag(w′⊙Xdβ̂_j)|^{1/2}‖²_F.

**Trace formation.** t_j = λ_j‖R_jL̂⁻ᵀ‖²_F is a sum of squares, so its relative error is γ_{p²}·t_j.

**The −½tr(S⁺A_j) term** is exactly −½r_j under Theorem 4 and cancellation-free under Theorem 5.

**Cancellation at large ρ_j.** th_j and ts_j both tend to r_j/2, and their difference is O(1/λ_j): true g₂ = 1.5e-12. The subtraction costs an absolute u·r_j and nothing relative to the certificate, which is absolute. So large |ρ| does **not** by itself destroy the gradient; only λ-growing factor or representation errors do. **[proved]**

If a relatively accurate tiny g_j is needed, for example by the asymptote/boundary lane, use the cancellation-free form. For a separated penalty in an aligned frame, let K be the Schur complement of the non-range block and M_j = K^{-1/2}Σ_jK^{-1/2}. Then

  ½tr(H⁻¹A_j) − ½r_j = −½tr((I + λ_jM_j)⁻¹),

a sum of positive O(1/λ_j) terms. *Proof:* (H⁻¹)_{RR} = (λΣ + K)⁻¹, and tr((λΣ+K)⁻¹λΣ) = r − tr((λΣ+K)⁻¹K). **[proved]**

**Proposition 8 (representation leakage). [first order, proved; checked]** Suppose the stored in-frame penalty Ŝ_j = fl(QᵀS_jQ), or a reassembled ÊᵀÊ, has |ẑᵀŜ_jẑ| = δ_j ≈ c u‖S_j‖ on a direction ẑ that should be exactly null. The fitted problem is then that of S_λ + λ_jδ_jẑẑᵀ, and to first order

  δV = ½λ_jδ_j[(ẑᵀβ̂)² + ẑᵀH⁻¹ẑ] − ½ log(1 + λ_jδ_j/(λ_n s_n)) + …,  δg_j ≈ δV.

Here λ_ns_n is the soft penalty's eigenvalue on ẑ. If ẑ lies in the joint null space, the pseudo-determinant *rank changes* instead, which is catastrophic under any threshold.

The error grows like e^{ρ_j}, the full range and not √. At ρ₂ = 22.7 with λ₂ = 7.4e9, the observed errors in the raw-block frame are dV = 3.6–5.1e-5 and dg₂ = 1.04e-4, for **both** Cholesky and QR. They are ≤ 1e-16 in the aligned frame.

Rounding in a *stiff–soft off-diagonal* perturbs the soft eigenvalue only at second order (δ²/d_max). A *diagonal* error on the soft direction perturbs it at first order.

**Remedy (structural zeros).** Build Ŝ_k in the frame with the null blocks *set* to exact zero, from the known range/null split. This is a backward perturbation of S_k of size u‖S_k‖ that **preserves the exact null space**. Its effect on V is λ_ju‖S_k‖‖P_Rβ̂‖² = O(u), because the range component of β̂ is O(1/λ_j). The unstructured perturbation of the same norm is harmful. **[proved to first order]**

**Inner-mode error. [derived to first order]** Let the computed mode have residual r̂ = ∇_β(penalized objective)(β̃), so δβ ≈ −H⁻¹r̂.
- V is stationary in β, so δV_pen = ½r̂ᵀH⁻¹r̂, which is second order.
- The trace terms are **not** stationary in β: δ(th_j + tift_j) = ∇_β(th_j + tift_j)ᵀH⁻¹r̂, which is first order.
- If the evaluation applies the exact IFT correction for β̃ (gamfit's `kkt` channel), then a correctly applied correction is not an error. Only its own rounding, the first-order trace term above, and the O(‖r̂‖²_{H⁻¹}) remainder are errors.
- The attainable residual floor is

  r_fl ≈ γ_n|X|ᵀ|μ−y| + γ_p|S_λ||β̂|,

which is λ-benign in an aligned frame, since |S_λ||β̂| = O(1) in the stiff block. This r_fl is the derived inner stopping tolerance.

**Evaluation-point rounding. [proved]** λ = fl(e^ρ) equals e^{ρ(1+θ)} with |θ|·|ρ| ≤ u, and ρ itself sits on a grid of spacing u|ρ|. The evaluation is therefore exact at a point within u(1+|ρ_k|) of ρ in coordinate k, which gives

  ε^ρ_{g,j} = u Σ_k (1+|ρ_k|)|𝓗_{jk}|.

### 3.4 The explicit bands (route C, aligned frame)

**Value band.**

  ε_V = ε_dev + ε_q + ½[δ̃T̃/(1−η) + 2γ_pΣ|log l̂_ii| + 2pu] + ½γ_K(Σr_k|ρ_k| + |c_0|) + ½r̂ᵀH⁻¹r̂ + γ_4·(Σ of channel magnitudes)

The pieces are:
- ε_dev and ε_q are the deviance and penalty-quadratic bounds of §3.2;
- the bracketed term is ½ of the Theorem 2 bound for log|H|;
- ½γ_K(Σr_k|ρ_k| + |c_0|) is the affine log|S| term, where c_0 = log det(R̄R̄ᵀ);
- ½r̂ᵀH⁻¹r̂ is the inner residual;
- the last term is the final 4-channel summation.

**Gradient band.**

  ε_{g,j} = ½[δ̃λ_j‖R_jH⁻¹D^{1/2}‖²_F + ‖Δ̃‖‖D^{1/2}H⁻¹M_jH⁻¹D^{1/2}‖_*]/(1−η) + ½γ_{p²}(th_j + |tift_j|) + ½γ_{p+1}λ_j‖|R_j||β̂|‖² + |∇_β(th_j+tift_j)ᵀH⁻¹r̂| + O(r̂ᵀH⁻¹r̂) + u Σ_k(1+|ρ_k|)|𝓗_{jk}|

**Route E.** Replace δ̃, D and T̃ by c_pu‖H‖₂, I and tr(H⁻¹).

All quantities are by-products of the evaluation:
- T̃ = Σ h_ii(H⁻¹)_ii;
- λ_j‖R_jH⁻¹D^{1/2}‖²_F uses the same R_jL̂⁻ᵀ products as th_j, with columns weighted by D^{1/2};
- ‖H̃⁻¹‖₂ ≤ T̃.

**Hessian band. [derived to first order]** 𝓗_{jk} contains terms tr(H⁻¹A_jH⁻¹A_k), tr(H⁻¹A_j) δ_{jk} and the IFT third- and fourth-derivative traces. The same argument gives

  |δ𝓗_{jk}| ≤ 2δ̃·‖D^{1/2}H⁻¹A_jH⁻¹A_kH⁻¹D^{1/2}‖_* + (the analogous IFT terms)

For route E, replace δ̃ and D by c_pu‖H‖ and I. `decrement_bands.rs:141` charges γ_m‖𝓗‖_F instead, a formation-only term with no factorization content.

### 3.5 The certificate

**Theorem 9. [proved]** Let P be the (coordinate) face projector: I in the interior, with coordinates zeroed on a structural face. Assume |ĝ_j − g_j| ≤ ε_{g,j} rigorously, and let τ_j ≥ 0 be the statistical tolerance from the statistical-resolution lane. Certify iff

  |(Pĝ)_j| ≤ max(ε_{g,j}, τ_j − ε_{g,j}) for all j.

Then:
- (a) **Completeness.** The test passes at any exact stationary point, Pg = 0.
- (b) **Soundness.** A pass implies |(Pg)_j| ≤ max(2ε_{g,j}, τ_j).
- (c) **Minimality.** Replace max(·) by c·ε_{g,j} with c < 1. Then an admissible rounding error, attaining the bound, makes the test fail at an exact stationary point. So c = 1 is minimal, and it is *derived*, not tuned.
- When 2ε_{g,j} > τ_j, the point is *arithmetic-limited*. The correct action is a more accurate route (§3.1–3.3), not widening the band.

*Proof.* (a) holds because |ĝ_j| ≤ |g_j| + ε_j. (b) is the triangle inequality. (c) follows by taking ĝ_j = g_j + ε_j. ∎

The bounds are worst case; the observed errors are 10–100× smaller (§4). A probabilistic γ̃_k = O(√k·u) (Higham & Mary 2019) is admissible only if the certificate is allowed to be probabilistic. It is *not* recommended for the SPEC's "certified".

**Decrement form.** A value-based stop is resolvable only if the predicted decrease ½ĝᵀ𝓗⁻¹ĝ exceeds 2ε_V.

### 3.6 Line-search stall law

**Proposition 10. [proved]** Let each computed value carry an error of at most ε_V. Along a direction d, write a = |gᵀd| and b = dᵀ𝓗d > 0. The Armijo test with constant c₁ can be guaranteed to accept some step only if the model satisfies

  max_α [(1−c₁)αa − ½α²b] = (1−c₁)²a²/(2b) ≥ 2ε_V.

For the quasi-Newton direction this means gᵀ𝓗⁻¹g ≥ 4ε_V/(1−c₁)². Below that threshold, an adversarial ±ε_V error can defeat every α. This is the StepSizeTooSmall/MaxAttempts signature.

**Remedy, in two parts.**
1. Make ε_V small: route C in an aligned frame gives ε_V ≈ 1e-13.
2. Where the stall law still binds, use the Hager–Zhang approximate Wolfe conditions (Hager & Zhang 2005, eq. (4.1) and the ε_k of §4). Take ε_k = 2ε_V, the *derived* band, rather than their ε|f_k|.

---

## 4. Numerical checks

All float64 results are compared against mpmath references.
- `exp_sweep.py` and `ref_checkpoint.py` use 60 digits.
- `exp_frames.py`, `exp_routes.py` and `exp_bfgs.py` compare against the 60-digit reference pickled by `ref_checkpoint.py` or against themselves. Where they report "0", the error is below the resolution of an internal 15-digit mp step. The sweep independently confirms that route C has |dV| ≤ 1e-13.
- `exp_kron.py` and `exp_kron2.py` use 50 digits.

### 4.1 Frames at the checkpoint (`exp_frames.py`)

| frame | route | \|dV\| | max\|dg\| |
|---|---|---|---|
| aligned (structural zeros) | C, Q | ulp | ~1e-16 |
| raw block (numerical Qᵀ S Q) | C, Q | 3.6–5.1e-5 | 1.04e-4 (coord 2), 1e-5 (coord 3) |
| random full rotation | C, Q | 3–5e-5 | 1.1e-4 |

Cholesky and QR agree to within the table, so the error comes from representation, not factorization (Prop. 8).

### 4.2 Bounds against observation at the checkpoint (`exp_routes.py`)

| | κ(H) | κ(H̃) | pred \|dV\| E | pred \|dV\| C | obs \|dV\| E |
|---|---|---|---|---|---|
| ρ* | 3.2e11 | 2.07 | 2.15e-4 | 4.6e-14 | 6.4e-6 |

Per-coordinate E-route gradient errors:

| | coordinate 0 | coordinate 1 | coordinate 2 | coordinate 3 |
|---|---|---|---|---|
| predicted bound | 7.8e-6 | 5.6e-5 | 1.8e-14 | 4.7e-6 |
| observed | 3.7e-7 | 3.1e-6 | 7.9e-16 | 2.7e-7 |
| row-permutation scatter (max) | 7.7e-7 | 4.6e-6 | 1.5e-15 | 4.1e-7 |

- Route E scatters under row permutation: sd(dV) = 3.7e-6, max 8.9e-6.
- Route C's scatter is 0 or about 1e-16.
- The bounds hold everywhere, and they exceed the observed errors by about 10–100×.

### 4.3 ρ₂ sweep (`exp_sweep.py`, aligned frame, 60-digit reference)

| ρ₂ | true g₂ | E \|dV\| | E bound | C \|dV\| | E max\|dg\| | C max\|dg\| |
|---|---|---|---|---|---|---|
| 0 | 1.1e-4 | 9e-15 | – | 5e-14 | – | ~1e-16 |
| 8 | 3.7e-6 | 2e-12 | – | 9e-14 | – | ~1e-16 |
| 12 | 6.7e-8 | 1.4e-10 | – | 9e-14 | – | ~1e-16 |
| 16 | 1.2e-9 | 6.5e-9 | – | 4e-14 | – | ~1e-16 |
| 20 | 2.2e-11 | 1.1e-7 | – | 6e-14 | 7.4e-8 | ≤2e-15 |
| 22.73 | 1.5e-12 | 1.6e-6 | 2.2e-4 | 1.3e-14 | 9e-7 | ≤2e-15 |
| 26 | 5.6e-14 | 6.6e-5 | 5.7e-3 | 4e-15 | 3.4e-5 | ≤2e-15 |
| 30 | 1e-15 | 6.9e-4 | 0.31 | 6e-14 | 3.9e-4 | ≤2e-15 |

**The rail is an asymptote.** True g₂ ∝ e^{−ρ₂}, so there is no interior stationary point in ρ₂.

**Penalty log-det variants with route C:**
- Assembled eigendecomposition + 100pε threshold: dV = 1.5–2.7 and dg = 0.5 at ρ₂ ≥ 26. At spans of ±30 the errors reach dV = 87 and dg = 1.5 (rank misclassification).
- Affine (Thm 4) and stacked-root SVD: accurate everywhere in this separated case.

At spans of ±30 (κ(H) up to 6e17), route E has dV = 0.09–0.16 and route C has dV = 2e-14 to 6e-14.

### 4.4 Optimizer (`exp_bfgs.py`, ρ₂ fixed at 22.73, free coordinates {0, 1, 3})

| route | result | iterations | final \|ĝ\| | true \|g\| |
|---|---|---|---|---|
| E | StepSizeTooSmall | 28 | 1.27e-4 | 9.9e-5 |
| C | converged | 29 | 1.9e-14 | ~1e-14 |

- Route C converges to z* = (−2.18602, −3.08437, −4.66683).
- Value jitter about a quadratic fit along a line: sd 2.5e-5 (max 6.6e-5) for route E, against 7.5e-14 for route C.
- The outer Hessian of the free block has eigenvalues (0.017, 0.266, 0.485).
- **Stall check.** At the E stall, ĝᵀ𝓗⁻¹ĝ ≤ (1.27e-4)²/0.017 ≈ 9.5e-7. This is inside the Prop. 10 region 4ε_V ≈ 1e-4. The replica reproduces gamfit's failure mode exactly.

### 4.5 Overlapping tensor-product penalty (`exp_kron.py`, `exp_kron2.py`)

Setup: S = λ₁A⊗I + λ₂I⊗B, where A and B are 7×7 second-difference penalties, p = 49, structural rank 45, raw frame. Each cell shows the error in log|S|_+ / the error in ∂_{ρ₁}log|S|_+.

| (ρ₁, ρ₂) | eigh + 100pε | stacked-root SVD | row-sorted QRCP + Jacobi | closed form (Thm 5) |
|---|---|---|---|---|
| (0, 0) | 1e-14 / 1e-14 | 3e-14 / 2e-14 | 3e-13 / 2e-13 | 3e-14 / 0 |
| (0, 10) | 1.4e-10 / 1.4e-10 | 1.1e-13 / 1e-13 | 4e-13 / 7e-14 | 0 / 0 |
| (0, 20) | 7.6e-6 / 7.6e-6 | 1.1e-11 / 1.1e-11 | 2e-13 / 9e-14 | 1e-13 / 2e-15 |
| (0, 30) | **10.6 / 10.0 (rank 35)** | 5.8e-10 / 5.8e-10 | 2e-13 / 4e-14 | 0 / 0 |
| (−20, 20) | **189 / 10 (rank 35)** | 3.0e-8 / 3.0e-8 | 1e-13 / 1e-14 | 0 / 0 |
| (−30, 30) | **289 / 10 (rank 35)** | 5.6e-3 / 5.6e-3 | 1e-13 / 1e-14 | 2e-13 / 0 |
| (30, −30) | **289 / 3e-14 (rank 35)** | 2.8e-3 / 2.3e-3 | 0 / 5.6e-3 → **0** with the complement identity | 1e-13 / 0 |

- The assembled eigendecomposition follows the κ law and misclassifies the rank beyond about e^{27}.
- The stacked-root SVD follows the √κ law: u·e^{30} ≈ 2e-3 at ±30.
- QRCP+Jacobi is λ-free once the complement identity is used for the stiff trace.

---

## 5. Literature

- **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM.
  - Lemma 3.1 (γ_k);
  - **Theorem 10.3** (Cholesky componentwise backward error |ΔA| ≤ γ_{n+1}|R̂ᵀ||R̂|);
  - Theorem 10.7 (success condition);
  - §7.3 and §10.1 on van der Sluis scaling;
  - **§19.4** (row-wise backward stability of Householder QR with column pivoting and row sorting).
- **van der Sluis, A. (1969).** "Condition numbers and equilibration of matrices." *Numer. Math.* 14:14–23. Diagonal scaling to a unit diagonal is within a factor p of optimal.
- **Cox, A. J. & Higham, N. J. (1998).** "Stability of Householder QR factorization for weighted least squares problems." In *Numerical Analysis 1997* (Proc. 17th Dundee Biennial Conf.), Pitman Res. Notes Math. 380, 57–73. Sorting rows by decreasing norm plus column pivoting gives row-wise backward stability; one sign choice in the Householder vector is required. The result goes back to Powell & Reid (1969).
- **Demmel, J. & Veselić, K. (1992).** "Jacobi's method is more accurate than QR." *SIAM J. Matrix Anal. Appl.* 13(4):1204–1245.
- **Demmel, J., Gu, M., Eisenstat, S., Slapničar, I., Veselić, K. & Drmač, Z. (1999).** "Computing the singular value decomposition with high relative accuracy." *Linear Algebra Appl.* 299:21–80. Rank-revealing decomposition followed by one-sided Jacobi.
- **Drmač, Z. & Veselić, K. (2008).** "New fast and accurate Jacobi SVD algorithm I/II." *SIAM J. Matrix Anal. Appl.* 29(4):1322–1342 and 1343–1362. QR preconditioning plus one-sided Jacobi.
- **Eisenstat, S. C. & Ipsen, I. C. F. (1995).** "Relative perturbation techniques for singular value problems." *SIAM J. Numer. Anal.* 32(6):1972–1988. Used for the row-wise relative perturbation bounds in Prop. 6.
- **Wood, S. N. (2011).** "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models." *J. R. Statist. Soc. B* 73(1):3–36.
  - §3.1 and Appendix B give the λ-dependent similarity transform. It splits penalties into dominant and subdominant sets by the magnitude of λ_k‖S_k‖, eigendecomposes the dominant sum, and recurses on its null space.
  - It is a dominance-ordered elimination, in spirit the "row sorting" of Cox–Higham applied blockwise. That is why it is accurate across λ ranges.
  - It relies on fixed rank and dominance thresholds, which are magic constants. It also recomputes the basis at every ρ, so the route changes discontinuously when the dominance set changes.
  - Theorems 4 and 5 and Algorithm R give the same accuracy without thresholds, where they apply.
- **Hager, W. W. & Zhang, H. (2005).** "A new conjugate gradient method with guaranteed descent and an efficient line search." *SIAM J. Optim.* 16(1):170–192. The approximate Wolfe conditions and the error-tolerant ε_k are in §4.
- **Higham, N. J. & Mary, T. (2019).** "A new approach to probabilistic rounding error analysis." *SIAM J. Sci. Comput.* 41(5):A2815–A2835. Gives the probabilistic √k·u constants; see §3.5.
- **Golub, G. H. & Van Loan, C. F. (2013).** *Matrix Computations*, 4th ed., §8.3–8.4. Backward stability of the symmetric QR algorithm.

---

## 6. Consequences for gamfit

Paths are relative to `SP/main_src/crates/`. Items are ordered by impact on the failing clusters.

### 6.1 Make the evaluation route accurate (primary fix)

**Build.** Form H *in an aligned frame with structural zeros* and factor it by Cholesky, or by row-sorted, column-pivoted QR of the stacked root. Take log|H| = 2Σlog l_ii. Derive th_j from ‖R_jL⁻ᵀ‖²_F and tift_j from the same factor. Specifically:

- **`gam-terms/src/construction.rs:2560–2900` (reparameterization).**
  - Stop reassembling `s_truncated = EᵀE` by matmul.
  - Stop asserting that null leakage ≤ 1e-10 (`:2893`). That assert tests a symptom with a magic number.
  - Instead, *construct* each in-frame S′_k with its null rows and columns set to exact 0 from the known range/null partition (Prop. 8 remedy).
  - `eigenvalue_floor = max·1e-12` (`:2766`) and `BALANCED_PENALTY_RANK_RELATIVE_TOL = 1e-12` (`:1956`) are magic constants. Replace them with the *structural* rank: each penalty's known null dimension, e.g. polynomial degree < m for an m-th-derivative or difference penalty, and the tensor null set N of Thm 5.
- **`gam-solve/src/reml/reparameterized_inner.rs`** (survival and custom-family lanes). This lane forms H′ = QsᵀHQs from a raw-frame H, which carries the leakage of Prop. 8. Assemble H′ = (XQs)ᵀW(XQs) + Σλ_kS′_k directly in the frame, with S′_k structural.
- **`gam-solve/src/pirls/loop_driver.rs:830–900`.** The sparse-native path uses identity (raw) coordinates. Sparse Cholesky keeps the componentwise Theorem 10.3 bound under any symmetric fill-reducing permutation, so it is λ-free *provided* the penalties are structurally exact in those coordinates. A raw difference or B-spline penalty with an exact polynomial null space is fine; a numerically rotated one is not.
- **`gam-solve/src/reml/reml_outer_engine/dense_spectral.rs:1249`.** Keep `logdet_forward_error` as the correct Theorem 1 band *for route E*, but stop using route E for log|H| and the traces at large ρ. It should remain only where an eigendecomposition is intrinsically needed, such as a genuinely rank-deficient H, and not as the default.
- **`gam-solve/src/reml/laml_logdet.rs`.**
  - `assembled_logdet_is_resolved` (`:184–185`, the √ε·(1+|logdet|) gate) and the 16·max(bound, √ε·…) at `:460` are magic. Replace them with Theorem 2's bound for the route actually used.
  - The root-scale operators (`:231`, `:270`, `:286`) become route Q. Add row sorting and column pivoting to make them row-wise stable.

### 6.2 Penalty log-determinant

- **`gam-solve/src/reml/reml_outer_engine/pseudo_logdet.rs:26–35`.** Delete `SAFETY_FACTOR = 100` and the eigenvalue threshold 100pε·max|ev|. The threshold misclassifies the rank beyond about e^{27} (§4.3, §4.5). Dispatch on structure instead:
  - **independent ranges** (single, double, disjoint): the affine form Σr_kρ_k + c₀ (Thm 4), with det term r_k exactly;
  - **Kronecker sums**: the closed form (Thm 5);
  - **other overlap**: Algorithm R with the complement identity.

  The callers to migrate are `objective.rs:3298` (`positive_penalty_rank_and_logdet`), `sparse_cholesky_backends.rs:1304` and `:1360`, and in gam-models `bms/block_specs.rs:754`, `gradient_paths.rs:567`, `deviation_runtime.rs:298` and `:1136`, `survival/marginal_slope/row_math.rs:414`, `penalized_vector_glm.rs:636` and `:955`, `inference/full_conformal.rs:938`, plus the gam-sae manifold code.
- **`gam-solve/src/reml/penalty_logdet.rs:770–859` (`eigensystem_from_scaled_roots`).** The plain SVD of the stacked scaled roots obeys the √κ law (Prop. 6). Replace it with Algorithm R, or with Thm 4/5 when the structure allows.
- **Consistency requirement.** V, g and 𝓗 must be derived from the *same* factorization and the same log|S| route. The three log-det modules are equivalent in exact arithmetic but not in floating point: they differ by up to O(1) at spans of e^{±30}. Mixing them makes g inconsistent with differences of V, which by itself stalls line searches.

### 6.3 Bands and certificate

- **`gam-solve/src/rho_optimizer/outer_measurement.rs:15` (`OuterFirstOrderMeasurement`).** Add fields `value_band: f64` (ε_V) and `gradient_band: Array1<f64>` (ε_{g,j}), computed from the §3.4 formulas by the evaluation that produced value and gradient. Every certificate reads the band from the measurement it certifies.
- **`gam-solve/src/rho_optimizer/decrement_bands.rs`.**
  - `:86`: the gradient band `growth·channels + |total − envelope|` must become ε_{g,j} of §3.4. That means:
    - add the factorization term ½δ̃λ_j‖R_jH⁻¹D^{1/2}‖²_F/(1−η), or c_pu‖H‖λ_j‖R_jH⁻¹‖²_F for route E;
    - add the IFT factorization term;
    - replace the full |kkt| charge with |∇_β(th_j+tift_j)ᵀH⁻¹r̂| + ½r̂ᵀH⁻¹r̂;
    - add u Σ_k(1+|ρ_k|)|𝓗_{jk}|.
  - `:111–119`: in the objective band, |kkt| again should be the remainder, not the whole correction.
  - `:141`: the Hessian band γ_m‖𝓗‖_F becomes the Theorem 7 analogue of §3.4.
  - The `ObjectiveNotResolvable` test against `rel_cost_floor·(1+|V|)` then compares a derived band with a derived statistical resolution.
- **`gam-solve/src/rho_optimizer/run.rs:8702` (`outer_arithmetic_gradient_floor`).** **Delete** `scale·√ε`, i.e. n√ε = 7.30e-6 for n = 490. It is not an error bound for any route: it is 1e8× the route-C error and 5–60× smaller than the route-E error at the failing point.
- **`run.rs:8774` (`outer_engine_gradient_band`) and `:8823` (`outer_stationarity_band_and_rung_at`).**
  - The `SolverBand` rung becomes the per-coordinate derived ε_{g,j} from the measurement.
  - `CertificateScoreRelative` (`rel_cost_tol·(1+|V|)`) becomes the statistical τ_j from the statistical-resolution lane, combined per Theorem 9 as max(ε_j, τ_j − ε_j).
  - Add a rung `ArithmeticLimited` for 2ε_j > τ_j. It is a *diagnostic* demanding a better route, never a pass.
  - The solver's `GradientTolerance` (`:8743`) should receive the same per-coordinate band. `opt` needs a vector tolerance, or test on the scaled vector ĝ_j/max(ε_j, τ_j − ε_j) ≤ 1 in ∞-norm.
  - `bridges.rs:3775` (`projected_gradient_norm`): certify with that scaled ∞-norm, not a raw 2-norm against a scalar.
- **Line search (`opt` crate).** When the stall law (Prop. 10) binds, pass the derived ε_V so that Armijo, or Hager–Zhang approximate Wolfe, uses slack 2ε_V. This is a derived constant, not a retry or jitter.

### 6.4 Tests

`gam-solve/src/rho_optimizer/logdet_forward_error_1b_tests.rs` currently checks only who charges which bound. Add three tests.

1. **λ-sweep.** On the §2 replica, or any separated double penalty, run ρ_stiff ∈ {0, 8, …, 30}. Assert:
   - the route-C log|H| error against a high-precision or closed-form reference is ≤ Theorem 2's bound, and that the bound is λ-free (≤ 1e-12 across the sweep);
   - the route-E bound grows ∝ e^{ρ} and contains the observed error.
2. **Leakage.** Assert that a numerically rotated penalty shows an error ∝ λ_j, and that the structural construction shows none.
3. **Kronecker.** Assert that the closed form, Algorithm R and a 50-digit reference agree to ≤ 1e-12 at (±30, ∓30), and that the threshold route is rejected.

These are tests, so high-precision references are allowed.

### 6.5 Failing clusters addressed

- **Binomial rails** (all-tests.log:324, |Pg| = 2.28e-5 vs 7.30e-6, StepSizeTooSmall) **and prostate** (:786, :822, 9.6e-6). Fixed directly by §6.1, §6.2 and §6.3 (the run.rs:8702 deletion). The railed ρ₂ is a genuine asymptote with g₂ ∝ e^{−ρ₂}, which the boundary lane handles. With route C the free coordinates converge to 1e-14 (§4.4).
- **cc/ARC decrement** (:1009, band_f = 3.95e-11 not resolvable). The derived bands of §6.3 replace `growth·(…+|kkt|)`, which overcharges the kkt channel.
- **Multinomial "Newton decrement stopped contracting"** (:13475). Same band derivation. Whether the root cause is route error is not verified; the multinomial lane decides.
- **Iso-kappa** (:14600, :15518, :15831, :33534; |Pg| = 0.331 vs 0.0181). This is not an arithmetic problem at that size. The derived bands only replace the magic rungs; the identifiability lane owns it.
- **Survival** (:33697, :33712). The Prop. 8 leakage in `reparameterized_inner.rs` is a plausible contributor (unverified). The derived rungs replace `certificate-score-relative`.

---

## 7. Open problems

1. **General overlapping penalties.** The claim that Algorithm R is λ-free for overlapping penalties without Kronecker or dominance structure (for example adaptive or multiple te sharing margins, or Matérn precision plus a ridge) is a conjecture. A proof would need a bound on the column-scaled condition number of the pivoted R factor, uniform in λ.
2. **ψ/κ-dependent penalties (iso-kappa Matérn).** S(κ) changes the frame with κ, so structural zeros must survive κ-derivatives. The error analysis of ∂S/∂κ traces is not done here.
3. **Outer Hessian.** Only the first-order factorization term is derived. The IFT 3rd/4th-derivative traces (w″, w‴) need the same D-scaled treatment made explicit.
4. **Non-canonical links and families with negative w.** H may be indefinite away from the mode. Cholesky's success condition then needs H ≻ 0 to be certified by its own run, which is a clean pass/fail and not a fallback.
5. **Tightness.** The bounds exceed the observed errors by 10–100×. A rigorous but tighter a-posteriori bound would use the computed residual ‖L̂L̂ᵀ − H‖ evaluated in extended precision (compensated dot products), which is still a derived and not a probabilistic quantity. It is cheaper than a probabilistic certificate and would suit Theorem 9 when τ_j ≈ 2ε_j.
6. **Rank decisions.** Structural rank settles every construction gamfit ships (difference, derivative and tensor penalties). Penalties that are genuinely only numerically rank-deficient, such as data-dependent or learned ones, have no structural rank, and the right certified rank decision for them is open.
