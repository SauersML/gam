# Convergence theory for multi-predictor models: multinomial softmax and location-scale (GAMLSS)

Slug: `multinomial-ls`. Scope: `statsmodels_ordinal_mnlogit` (ARC Newton-decrement stall,
`rho_checkpoint=[4.2535, 12.2882, -1.7712]`), `vgam_multinomial_softmax` (METRIC_OFF on the
main arm and a refusal on the penguins arm), `gamlss_gaussian_survival_ls` (a METRIC_OFF on the
synthetic arm and a TIMEOUT on the real-data arm), and the Gaussian location-scale inner/outer
problem in general.

Status labels: **[P]** proven here (a full proof or a complete derivation), **[N]** checked
numerically (the script is named), **[C]** conjectured or supported by evidence only.

---

## 1. Summary

1. **The "multinomial" ARC stall is not in the multinomial solver [P+N].** The panic in
   `tests/quality/families/quality_vs_statsmodels_ordinal_mnlogit.rs:150` comes from a *Gaussian*
   standard-REML fit of `y ~ x1 + s(x2, bs="cyclic")`. The test runs that fit only to obtain the design
   matrix; the multinomial fit that follows uses a fixed λ. That Gaussian model has three
   penalties:
   - the x1 null-recovery ridge;
   - the cyclic **harmonic** roughness, with null space {1, sin ωx, cos ωx} (`gam-terms/src/basis/bspline_build.rs:168-200`);
   - the fundamental-harmonic ridge (`bspline_build.rs:250-275`).

   The DGP truth `0.6·x1 + sin(2πx2)` lies *exactly in the harmonic null space*, so the REML
   optimum for the roughness coordinate is the face **λ_bend = +∞** (ρ₁ = 12.29 is on its way there).
   On an exponential asymptote V = V∞ + A e^{−ρ}, a Newton step is ≈ +1 and the decrement contracts
   only by the factor e^{−1} per unit of ρ (linearly, not quadratically), and more slowly still
   under ARC regularization. The "Newton decrement stopped contracting" certificate is therefore
   the wrong certificate for this coordinate. The correct one is the face KKT condition in the
   compact coordinate u = e^{−ρ}: ∂V/∂u|_{u=0} = A ≥ 0, with a closed form for A (Thm 3.6). The
   replica reproduces λ̂² ≈ 1e-4 at ρ ≈ 15 with step ≈ 1 and A ≈ 1.23e3 > 0.
2. **The per-class ("equivariant") multinomial penalty has no repulsion at λ_c = 0 [P+N].** For
   the penalty Σ_c λ_c (C_cᵀC_c) ⊗ S_t (`multinomial_reml.rs:1440`), Cauchy–Binet gives
   det Σ_c λ_c C_cᵀC_c = e_{K−1}(λ)/K². Hence log|S_λ|₊ = Σ_t r_t log e_{K−1}(λ_{t,·}) + const, which
   stays finite when one λ_c → 0. The point λ_c = 0 is an *interior* point of the natural domain
   Λ_t = {λ : A_t(λ) = Σ_c λ_c C_cᵀC_c ≻ 0}. That domain is the hyperbolicity cone of e_{K−1} and
   contains points with λ_c < 0. V is analytic across λ_c = 0.

   In ρ = log λ coordinates the whole ray ρ_c → −∞ is therefore a spurious asymptotic critical set
   (∂V/∂ρ_c = λ_c ∂V/∂λ_c → 0) whenever the optimum over Λ_t has λ_c ≤ 0. This is the mechanism
   behind the penguins refusal, where 7 of 15 coordinates railed at ρ ≈ −15.3 and the ρ-Hessian had
   λ_min = −23.

   Fix: optimize the class metric on its natural domain (λ-coordinates restricted to the
   hyperbolicity cone, with an exact fraction-to-boundary rule from a (K−1)×(K−1) generalized
   eigenproblem), or parameterize A_t directly by a Cholesky factor. The shared metric M ⊗ S_t is the
   one-parameter alternative, and it is repelled from λ = 0 by the usual −½ r log λ term.
3. **Reference-class invariance [P+N].** A change of reference class is a unimodular linear map of
   the ALR coefficients, so log|H| and log|S_λ|₊ are *exactly* invariant (not just up to a constant).
   Both the shared M ⊗ S_t penalty and the equivariant per-class family are invariant; the family is
   permuted with the classes. The CLR/orthonormal-contrast chart changes the log-determinants only by
   the constant 2·p_t·log|det R|. V agrees to 1e-13 across references 0, 1 and 2
   (`mn_laml.py`).
4. **The Gaussian location-scale NLL in (μ, log σ) is indefinite for every row with a nonzero
   residual [P].** The per-row Hessian determinant is −2r²e^{−4η} < 0. It is also indefinite in
   (μ/σ, log σ), where the determinant is y(y−μ)/σ². It is **strictly convex** in
   (ν, τ) = (μ/σ, 1/σ) on τ > 0 (determinant τ^{−2}) and in the natural parameters (μ/σ², 1/σ²).
   With linear predictors in (ν, τ), the penalized inner problem is strictly convex and
   self-concordant on the polyhedral domain {Zc > 0}, which gives global Newton convergence with an
   exact stopping rule (Thm 3.9). The (μ, log σ) model, which is the one gamfit ships
   (`gamlss/gaussian/location_scale.rs:49-50`), has no such guarantee. For it, globally convergent
   inner Newton means: Fisher-scoring direction (always a descent direction) with Armijo
   backtracking, then exact observed Newton once H_obs + S_λ ≻ 0, then a negative-curvature step at
   saddles (Moré–Sorensen). The certificate is ‖g‖ within its rounding band plus a Cholesky of
   H_obs + S_λ.
5. **Fisher instead of observed Hessian in LAML costs exactness at order n^{−1/2} under correct
   specification, and at order 1 under misspecification or smoothing bias [P sketch + N].**
   - With a correctly specified model, log|H_obs| − log|F| = O_p(p/√n), which is larger than the
     O(n^{−1}) Laplace error.
   - With the mean misspecified (and smoothing bias is always a local misspecification), the
     difference tends to a nonzero constant: −0.177 for n from 400 to 25600 in `ls_convexity.py`.
   - The Fisher-trace gradient is also not the derivative of any criterion whose implicit-function
     Jacobian is H_obs, and that mismatch alone makes a Newton decrement stall.
   - gamfit's Gaussian LS already uses the observed joint Hessian for the outer problem
     (`location_scale.rs:917-929`), which is correct. Fisher is used only as the inner search
     direction (`location_scale.rs:1010-1019`), which is admissible under item 4.
6. **The LAML exists only at a strict local mode [P].** If H_obs + S_λ has a negative eigenvalue at
   β̂, then β̂ is not a local minimum and no Laplace approximation exists. Replacing negative
   eigenvalues by their absolute values or flooring them changes the objective, so it is not
   allowed. `PseudoLogdetMode::PositiveDefinite` (`multinomial_reml.rs:3128`) is the right
   contract. What it needs is a *derived* PD threshold: Cholesky success certifies
   λ_min(H + ΔH) > 0 with ‖ΔH‖ ≤ c_p ε‖H‖ (Higham 2002, Thm 10.3).
7. **Survival location-scale has an exact continuous gauge (h, η_t, σ) → c(h, η_t, σ)
   [P].** It is pinned in `survival/location_scale/prepare.rs:504-534`. Consequence: the recovered
   location is in h-units. A truth comparison in log-t units must divide by the log-t slope of h;
   the test at `tests/quality/survival/quality_vs_gamlss_gaussian_survival_ls.rs:~300` does not.
   This cannot, however, explain rmse = 5.09, which stays open (§7). The TIMEOUT on the real-data
   arm is undiagnosed. `survival/location_scale/constants.rs:57-112` lists seven magic constants on
   that path (cap 60, 1e-5, 1e-4, 1e-6, 1e-8, ×10, 1e8).
8. **Ad hoc items in the multinomial path, each with a replacement in §6:**
   - ~~`multinomial.rs:116` (inner tol 1e-5 taken from a measured plateau)~~ removed (#4053);
   - `:170` (exact outer Hessian gated at dimension ≤ 24);
   - `:183` (separation threshold |η| > 25);
   - ~~`:195` (outer tol 1e-7 as a floor)~~ removed (#4053);
   - `:129` (a penalty rescale that is harmless but unneeded);
   - `:145` and `:3537-3542` (references to "±10 box" and `effective_df_floor_rho_upper_bounds`
     boxes);
   - the vgam test comment at `quality_vs_vgam_multinomial_softmax.rs:826-828` still describes a
     "single tied λ per term", which contradicts the equivariant per-class specs actually used.

---

## 2. Setup and notation

**Multinomial.**
- Classes c ∈ {0, …, K−1}, rows i = 1…n, a common design X (n × p) split into terms t with
  penalties S_t (p × p, PSD, rank r_t, supported on the columns of term t).
- ALR (reference-coded) chart with reference class k: β = (β_a)_{a≠k} ∈ R^{(K−1)p}, with η_{ia} = x_iᵀβ_a
  and η_{ik} ≡ 0.
- Centred class coefficients γ_c = β_c − K^{−1} Σ_{c'} β_{c'} (with β_k = 0). Write γ_c = (C_c ⊗ I_p) β,
  where C_c ∈ R^{1×(K−1)} has entries (C_c)_a = δ_{ac} − 1/K for a ≠ k, and C_k = −1ᵀ/K.
- CLR metric M = Σ_c C_cᵀC_c = I_{K−1} − J/K, with eigenvalues 1 (multiplicity K−2) and 1/K.

**Penalties.**
- Shared (centred): S^{sh}_λ = Σ_t λ_t M ⊗ S_t (`multinomial_reml.rs:1343`).
- Equivariant per class: S^{eq}_λ = Σ_t A_t(λ_t) ⊗ S_t with A_t(λ) = Σ_c λ_{t,c} C_cᵀC_c
  (`multinomial_reml.rs:1440`). This is used for K > 2 (`joint_smoothing_dimension` at `:1432`).

**Hessian.**
Joint penalized Hessian H = H_ℓ + S_λ. The (a, b) block of H_ℓ is
Σ_i x_i x_iᵀ p_{ia}(δ_{ab} − p_{ib}). The link is canonical, so observed = Fisher.

**LAML** (Wood 2011; Wood, Pya & Säfken 2016, §3.1), to be minimized:

  V(ρ) = −ℓ(β̂) + ½ β̂ᵀS_λβ̂ + ½ log|H| − ½ log|S_λ|₊ + const,

with λ = e^ρ, and for Gaussian REML with profiled scale:
V = ½(n − M_p) log(D_p) + ½ log|XᵀX + S_λ| − ½ log|S_λ|₊.

**Location-scale (Gaussian).**
Per-row NLL in the shipped parameterization (μ, η = log σ):
ℓ_i = η + r²e^{−2η}/2, with r = y − μ. The predictors are μ = Xβ and η = Zγ, with penalties
S_μ and S_σ. Fisher information per row: diag(e^{−2η}, 2).

**Survival LS.**
u = (h(t) − η_t) e^{−η_σ}, S(t|x) = 1 − Φ(u), with h a monotone flexible warp
(`survival/location_scale/row_kernel.rs:303-309, 1067-1073`).

---

## 3. Results

### 3.1 Reference class, ALR vs CLR, and the log-determinants

**Proposition 3.1 [P].**
*Statement.* Let β^{(k)} and β^{(k')} be the ALR coefficients for references k and k'. Then
β^{(k')} = (R ⊗ I_p) β^{(k)} with R ∈ Z^{(K−1)×(K−1)} and |det R| = 1. Hence
- H^{(k')} = (R ⊗ I)^{−ᵀ} H^{(k)} (R ⊗ I)^{−1};
- log|H^{(k')}| = log|H^{(k)}|;
- log|S^{(k')}|₊ = log|S^{(k)}|₊ for any penalty defined through γ;
- V is exactly reference-invariant.

In a CLR/orthonormal-contrast chart β = (Q ⊗ I)θ, both log-determinants shift by the same
constant 2 p log|det Q|₊ (the penalty by rank-weighted constants), so V shifts by a
ρ-independent constant.

*Proof.* The reference change is β'_a = β_a − β_{k'} for a ≠ k', with β'_k = −β_{k'}. This is an
integer matrix whose inverse is of the same form, so it is unimodular. The likelihood is
chart-invariant and β̂ maps equivariantly. Congruence by a unimodular matrix preserves
determinants. The penalty is γᵀ(·)γ, and γ is chart-free, so S transforms by the same congruence.
Pseudo-determinants transform with |det R|² restricted to the range; here the range is mapped
bijectively and the factor is 1. ∎ Numerically, V agrees to 1e-13 for references 0, 1 and 2
(`mn_laml.py`).

*Consequence.* A penalty that is invariant to the choice of reference class is any penalty
written in terms of the centred γ_c that is permutation-symmetric in c. Both S^{sh} and S^{eq}
qualify.

### 3.2 Determinant of the equivariant class metric

**Theorem 3.2 [P+N].**
*Statement.* det A(λ) = det(Σ_{c=0}^{K−1} λ_c C_cᵀC_c) = e_{K−1}(λ_0, …, λ_{K−1}) / K², where
e_{K−1} is the elementary symmetric polynomial of degree K−1. Consequently
log|S^{eq}_λ|₊ = Σ_t [ r_t log e_{K−1}(λ_{t,·}) + (K−1) log|S_t|₊ − 2 r_t log K ].

*Proof.* Write A = CᵀΛC with C ∈ R^{K×(K−1)} (rows C_c) and Λ = diag(λ). By Cauchy–Binet,
det A = Σ_{|T|=K−1} Π_{c∈T} λ_c · det(C_T)².

Deleting row j of C leaves C_T. For j = k (the reference), C_T = I − J/K, whose determinant is
1 − (K−1)/K = 1/K. For j ≠ k, a column operation shows |det C_T| = 1/K as well; this follows from
invariance under the unimodular reference change of Prop 3.1. So every det(C_T)² equals 1/K².

The Kronecker rule |A ⊗ S|₊ = |A|^{r}|S|₊^{K−1} then gives the second statement. ∎

Checked symbolically for K = 3 ((λ0λ1 + λ0λ2 + λ1λ2)/9) and K = 4 (e_3/16) (`mn_laml.py`).

**Corollary 3.3 (no repulsion at λ_c = 0) [P+N].** For fixed λ_{c'} > 0 (c' ≠ c),
e_{K−1}(λ) → Π_{c'≠c} λ_{c'} > 0 as λ_c → 0. Every K−1 of the C_c span R^{K−1}, so H stays PD.
Therefore V(λ) → V_0 < ∞ and V = V_0 + λ_c ∂V/∂λ_c|_0 + O(λ_c²).

In ρ-coordinates ∂V/∂ρ_c = λ_c ∂_{λ_c}V → 0 and ∂²V/∂ρ_c² → 0 as ρ_c → −∞, *whatever the sign of
∂_{λ_c}V|_0*. Numerically, with λ = (e^{ρ_0}, 1, 1), V converges geometrically: 309.604224 at
ρ_0 = −12, −16 and −20. The shared penalty instead diverges linearly as −6ρ, since r = 12
(`mn_laml.py`).

**Theorem 3.4 (natural domain = hyperbolicity cone) [P].**
*Statement.* Λ = {λ ∈ R^K : A(λ) ≻ 0} is an open convex cone. It contains R^K_{≥0} minus the rays
with two or more zeros, and it equals the hyperbolicity cone of e_{K−1} with respect to 1. V is
real-analytic on Λ (where the inner mode exists), and −½ r_t log e_{K−1} → +∞ at ∂Λ is the natural
REML barrier.

*Proof.* A(λ + s·1) = A(λ) + sM with M ≻ 0. So det(A(λ) + sM) = det M · Π_j (s + μ_j), where the
μ_j are the (real) generalized eigenvalues of the symmetric pencil (A(λ), M). This shows e_{K−1} is
hyperbolic in direction 1 and that its hyperbolicity cone is {all μ_j > 0} = {A(λ) ≻ 0}
(Gårding 1959). Convexity follows from linearity of A in λ. Analyticity: A ≻ 0 means S^{eq}_λ is
PSD with a λ-independent range, H is PD, and β̂(λ) is analytic by the IFT. The log e_{K−1} term
diverges only at ∂Λ. ∎

*Interpretation.* The per-class parameterization is a linear slice of the class-metric cone. For
K = 3 it is a *complete* chart of Sym⁺(2): the three rank-one C_cᵀC_c are a basis of Sym(2). Its
REML optimum can lie at negative λ_c. In the ρ = log λ chart this appears as ρ_c → −∞ with a
vanishing gradient: a rail that no box, rail heuristic or asymptote certificate can certify,
because the true optimum is not on that ray at all.

**Exact line-domain rule [P].** For a step d from λ ∈ Λ, α_max = 1 / max(0, −μ_min), where μ_min is
the smallest eigenvalue of A(λ)^{−1/2} A(d) A(λ)^{−1/2}. This is a (K−1)×(K−1) eigenproblem and its
roots are real, so the fraction-to-boundary step is exact and needs no constant.

### 3.3 Quasi-separation per class

**Proposition 3.5 [P].**
*Statement.* Let N_λ = null(S_λ) restricted to the ALR coefficients. The penalized multinomial NLL
f(β) = −ℓ(β) + ½βᵀS_λβ is strictly convex and coercive (so it has a unique finite mode) iff there is
no nonzero δ ∈ N_λ that weakly separates the data. Weak separation means
x_iᵀ(δ_{y_i} − δ_c) ≥ 0 for all i and c, where δ_k = 0; this is the Albert & Anderson (1984) cone
condition restricted to N_λ.

*Proof.* On N_λ^⊥ the penalty is PD and −ℓ ≥ 0, so f is coercive there. On N_λ, f = −ℓ, whose
recession function is positive along δ iff δ does not weakly separate (Albert & Anderson 1984,
Thm 1, multinomial form). Strict convexity holds because H_ℓ is PD on the complement of the
direction 1 ⊗ (null of X), which the ALR chart removes. ∎

*Consequences.*
- With λ ∈ Λ, N_λ = ⊕_t (R^{K−1} ⊗ null S_t) ⊕ (unpenalized parametric columns). With the default
  null-recovery ridges (commit b7b874a2a1 for linear effects; the double penalty for smooths), N_λ
  reduces to the K−1 intercepts. Separation can then only come from an empty or pure class, which
  is a data condition checkable by LP (Konis 2007) *before* any fit.
- Inside Λ a finite mode therefore always exists. On ∂Λ (two λ_c → 0 in one term for K = 3, so A
  is singular), a separating direction in the lost class direction makes the mode run to infinity.
  There V stays finite: ½log|H| ≈ ½log λ cancels −½log λ, a genuine λ → 0 face with no finite fit.
  **[C]** This is the penguins signature ("softest curvature 3.096e-8 at/below rounding band").
  The certified outcome there is "no finite optimum" unless the model carries a proper prior such
  as the Jeffreys term (`joint_jeffreys_term_strength`, `multinomial_reml.rs:3119-3126`).
- `MULTINOMIAL_SEPARATION_ETA_THRESHOLD = 25` (`multinomial.rs:183`) is a proxy for this LP
  certificate and should be replaced by it.

### 3.4 Outer Hessian block structure

With Ṡ_{t,c} = λ_{t,c}(C_cᵀC_c ⊗ S_t) and B_j = dH/dρ_j = Ṡ_j + D_j, where
D_j = Σ_a ∂H_ℓ/∂β_a · (dβ̂/dρ_j)_a and dβ̂/dρ_j = −H^{−1}Ṡ_jβ̂:

  V_j = ½β̂ᵀṠ_jβ̂ + ½tr(H^{−1}B_j) − ½ ∂_{ρ_j} log|S_λ|₊,

  V_{jl} = δ_{jl}·½β̂ᵀṠ_jβ̂ − β̂ᵀṠ_jH^{−1}Ṡ_lβ̂ + ½tr(H^{−1}∂_lB_j) − ½tr(H^{−1}B_lH^{−1}B_j) − ½ ∂²_{ρ_jρ_l} log|S_λ|₊.

This is the standard formula (Wood 2011, §3; Wood, Pya & Säfken 2016, §3.1). The new ingredient is
the penalty term, which by Thm 3.2 is **block-diagonal across terms and dense within a term**.
For term t with ε = e_{K−1}(λ_t) and ε_c = ∂ε/∂λ_c = e_{K−2}(λ_{−c}):

  ∂_{ρ_c} log|S|₊ = r_t λ_c ε_c / ε,
  ∂²_{ρ_cρ_d} log|S|₊ = r_t [ δ_{cd} λ_c ε_c/ε + λ_cλ_d (ε_{cd}/ε − ε_cε_d/ε²) ],

where ε_{cd} = e_{K−3}(λ_{−c,−d}) for c ≠ d and ε_{cc} = 0.

The trace terms couple *all* coordinates through H^{−1}. Useful identity [P]:
Σ_c ∂_{ρ_{t,c}} = the derivative along the common scaling 1_t, which is the shared-λ coordinate.
So the 1_t direction of the equivariant problem reproduces the shared model's gradient, and the
contrast directions (orthogonal to 1_t) carry the class-specific information that goes flat as
any λ_c → 0.

### 3.5 Root cause of the Gaussian helper stall (mnlogit)

Model: `y ~ x1 + s(x2, bs="cyclic")`, Gaussian, with penalties
- S_x1: the rank-1 ridge on the x1 coefficient;
- S_bend: the harmonic roughness ∮(f'' + ω²(f − f̄))², null {1, sin, cos} (`bspline_build.rs:168-200`);
- S_fund: the rank-2 function-space ridge on the fundamental (`bspline_build.rs:250-275`).

The truth m = 0.6·x1 + sin(2πx2) has **zero harmonic roughness**. The profiled REML is monotone
decreasing in ρ_bend toward a finite limit (replica `gauss_harmonic.py`: V = 539.88 at ρ = 2,
527.9075 at 12, 527.89971 at 25), so the optimum is the face λ_bend = ∞. **[C, strongly supported]**
Coordinate 1 of the checkpoint (12.29) is ρ_bend. It is the only coordinate whose value is
consistent with an asymptote; ρ_2 = −1.77 is a strongly identified ridge (railing it costs 0.135,
as the log states).

**Theorem 3.6 (face certificate at λ = ∞) [P].**
*Statement.* Let penalty j have range R_j (basis U_j with U_jᵀU_j = I) and S_j = U_j Σ_j U_jᵀ.
Put u = 1/λ_j. As u → 0, V(u) = V_∞ + A_j u + O(u²), where

  A_j = ½ [ tr(Σ_j^{−1} G_j) − φ̂^{−1} g_jᵀ Σ_j^{−1} g_j ],

with
- g_j = U_jᵀ Xᵀ W (y − μ̂_∞): the range score at the λ_j = ∞ fit;
- G_j = U_jᵀ H_ℓ U_j − U_jᵀ H_ℓ N (Nᵀ H_∞ N)^{−1} Nᵀ H_ℓ U_j: the Schur complement of the range
  block, N a basis of R_j^⊥, and H_∞ the penalized Hessian with penalty j removed and β restricted
  to R_j^⊥;
- φ̂ the profiled scale (φ̂ = 1 for known-scale families).

The face λ_j = ∞ is a KKT point iff A_j ≥ 0 (together with first-order stationarity of the other
coordinates on the face).

*Proof.* Expand β̂(u) = β̂_∞ + u Δ + O(u²). From the penalized score, U_jᵀβ̂ = u Σ_j^{−1} g_j + O(u²),
so β̂ᵀS_jβ̂/u = u g_jᵀΣ_j^{−1}g_j + O(u²). The log-determinant term is
log|H| = log|Nᵀ H_∞ N| + log|Σ_j/u| + u·tr(Σ_j^{−1} G_j) + O(u²) (block determinant, Schur
complement expansion). The −½log|S|₊ part cancels the log|Σ_j/u| piece exactly (the ranks match).
In the profiled Gaussian case the ½(n − M_p)log D_p term contributes −(u/2φ̂) gᵀΣ^{−1}g, because
dD_p/du|_0 = −g_jᵀΣ_j^{−1}g_j and D_p/(n − M_p) = φ̂. The O(u) terms of the other coordinates vanish
by stationarity on the face (envelope theorem). ∎

For a rank-1 ridge this reduces to A ≥ 0 ⇔ g²/(φ̂G) ≤ 1, i.e. |t| ≤ 1 for the score statistic. This
is the Crainiceanu & Ruppert (2004) boundary condition for REML. The replica gives A ≈ 1.23e3 > 0
by the test-only difference quotient.

**Proposition 3.7 (Newton on an asymptote) [P+N].** If V = V_∞ + A e^{−ρ} + O(e^{−2ρ}), the exact
Newton step is 1 + O(e^{−ρ}) and λ̂² = A e^{−ρ}(1 + O(e^{−ρ})). So λ̂² contracts by e^{−1} per step
(linear) and never meets a quadratic-convergence certificate. With ARC regularization σ the step is
1/(1 + σ/A e^{ρ}) < 1, so contraction is slower still. The observed "1.452e-4 → 9.426e-5 over two
steps, then stalled" is consistent.

Replica (`gauss_harmonic.py`), pure Newton from ρ_bend = 8: steps 1.55, 1.29, 0.95, 0.94, 0.97,
0.98, 1.00, 1.01, with λ̂² = 0.37, 0.12, 2.6e-2, 9.5e-3, 3.6e-3, 1.4e-3, 5.1e-4, 1.9e-4. That is
ratio ≈ e^{−1} per step, and λ̂² ≈ 1e-4 at ρ ≈ 15.7, the same order as production's 1.45e-4 at 12.3.

*Correct algorithm.* Classify the coordinate by the sign of the compact-coordinate derivative
(u = e^{−ρ}, ∂_u V = −e^{ρ}∂_ρV). Once A_j (computed in closed form, not extrapolated) is
certified positive above its rounding band δ_A (§6.4), move the coordinate to the face u = 0.
Evaluate the reduced problem there (penalty j becomes a hard null constraint U_jᵀβ = 0) and certify
the remaining coordinates with the ordinary Newton decrement on the face.

### 3.6 Gaussian location-scale: convexity

**Proposition 3.8 [P, sympy in `ls_convexity.py`].** Per-row Hessians of the NLL:

(a) In (μ, η = log σ): H = e^{−2η}[[1, 2r], [2r, 2r²]] with det = −2r²e^{−4η} < 0 for all r ≠ 0.
So every row is indefinite, and the NLL is jointly convex in (β, γ) only on a measure-zero set of
data configurations. At a mode H can still be PD, because the sum of indefinite rank-2 row
matrices is PD. Example: for the intercept-only model at the MLE, Σr = 0 kills the cross term and
Σ2r²/σ² = 2n > 0.

(b) In (ν = μ/σ, s = log σ): det = y(y − μ)/σ², indefinite whenever y and y − μ have opposite signs.

(c) In (ν, τ) = (μ/σ, 1/σ), τ > 0: ℓ = −log τ + ½(τy − ν)², H = [[1, −y], [−y, y² + τ^{−2}]],
det = τ^{−2} > 0. **Strictly convex.**

(d) In the natural parameters (θ₁, θ₂) = (μ/σ², 1/σ²), θ₂ > 0: det = 1/(2θ₂³) > 0. Strictly convex
(the log-partition function of an exponential family).

**Theorem 3.9 (convex LS inner problem) [P].**
*Statement.* Let ν = Xb and τ = Zc, with the penalty ½bᵀS_νb + ½cᵀS_τc and weights w_i. The
objective f(b, c) = Σ_i w_i[−log(z_iᵀc) + ½(y_i z_iᵀc − x_iᵀb)²] + penalties is strictly convex on the
open polyhedron D = {c : Zc > 0}. If w_i ≥ 1 (or after rescaling by min_i w_i), f is standard
self-concordant: −log is self-concordant, quadratics are, and sums of self-concordant functions
are. So damped Newton with step 1/(1 + λ_N) converges from any point in D with at most
(f_0 − f*)/0.0888 + log₂log₂(1/ε) iterations. It stops with the *exact* suboptimality bound
f − f* ≤ λ_N² once λ_N ≤ 0.68 (Nesterov 2004, Thm 4.1.12 and §4.1.5; Boyd & Vandenberghe 2004,
§9.6). f is coercive iff no nonzero direction in null(S_ν) × null(S_τ) is a recession direction,
which is a finite LP.

This is a *different model* (the scale predictor is additive in 1/σ, not in log σ). It is the one
location-scale chart in which "inner convergence certified" is a theorem rather than a hope. It is
a legitimate modelling choice (a link choice) and could be offered as `sigma_link = "inverse"`. It
does not by itself fix the log-link model.

**Proposition 3.10 (existence for the log-σ model) [P].**
*Statement.* f(β, γ) = Σ w_i[η_i + ½r_i²e^{−2η_i}] + ½λ_μβᵀS_μβ + ½λ_σγᵀS_σγ is bounded below and
coercive iff, for every nonzero δ ∈ null(S_σ) with Σ_i w_i z_iᵀδ ≤ 0, the rows
I_δ = {i : z_iᵀδ < 0} cannot all be interpolated (r_i = 0) by some μ = Xβ with finite penalty.

*Proof.* Along penalized γ-directions the quadratic penalty dominates the linear gain. Along
δ ∈ null(S_σ), f(γ + sδ) = s Σ w_i z_iᵀδ + Σ ½w_i r_i² e^{−2(η_i + s z_iᵀδ)} + const, which tends to
−∞ iff Σ w_i z_iᵀδ < 0 and every row with z_iᵀδ < 0 has r_i = 0. ∎

With null(S_σ) = {1} this says "μ cannot interpolate all n rows". That holds generically for
p_μ < n and fails for saturated μ bases, a degenerate-likelihood case that must be refused, not
fitted.

**Theorem 3.11 (globally convergent inner Newton for the log-σ model) [P, standard].**
*Statement.* Let F(β, γ) = blockdiag(XᵀW_μX, 2ZᵀW_wZ) + S_λ ≻ 0, the Fisher information plus penalty.
Iterate as follows.
1. d = −F^{−1}∇f, with Armijo backtracking. This is a descent direction because F ≻ 0 and
   κ(F) is bounded on the compact level set from Prop 3.10. By Zoutendijk's theorem
   (Nocedal & Wright 2006, Thm 3.2), ∇f → 0.
2. Once the Cholesky of H_obs + S_λ succeeds, switch to d = −(H_obs + S_λ)^{−1}∇f (quadratic local
   convergence).
3. If ∇f is at its rounding band but the Cholesky fails, take the negative-curvature direction
   (the eigenvector of the smallest eigenvalue) with a curvilinear Armijo search
   (Moré & Sorensen 1979). Limit points then satisfy the second-order necessary conditions.

Only step 3 makes a saddle a non-terminating state. WPS 2016 (§3.1.2) instead perturb
H + εI with ε increasing until the Cholesky succeeds. Their note that the perturbation "does not
change the converged state" holds for first-order stationarity but not for second-order: it can
converge to a saddle. **The certificate is the Cholesky of H_obs + S_λ, not of F.**

### 3.7 LAML with observed vs Fisher Hessian

**Proposition 3.12 [P sketch + N].** Let Δ = F − H_obs at β̂. For the Gaussian LS:
- Δ_{μμ} = 0 (identity link);
- Δ_{μσ} = −Σ 2 r_i e^{−2η_i} x_i z_iᵀ;
- Δ_{σσ} = Σ 2(1 − r_i²e^{−2η_i}) z_i z_iᵀ.

Then log|F + S| − log|H_obs + S| = tr((H_obs + S)^{−1}Δ) + O(‖(H+S)^{−1}Δ‖²).

- Under correct specification each entry of Δ is a sum of mean-zero terms: O_p(√n) against
  H = O(n), so the log-determinant error is O_p(p/√n). This is **larger than the O(n^{−1}) Laplace
  error** (Tierney & Kadane 1986), so Fisher-LAML is not the Laplace approximation to that order.
- Under misspecification (and smoothing bias at finite λ is a local misspecification) the means are
  nonzero and the error is O(1).

Numerically (`ls_convexity.py`, 3+3 quadratic predictors, 40 replicates):
- well specified: the mean difference falls from −0.105 (n = 100) to about −1e-3 (n = 25600), with
  sd·√n ≈ 1.2–1.4;
- misspecified mean (sin(2x) fitted by a quadratic): mean −0.177 at every n from 400 to 25600.

Beyond the value error, a Fisher-trace gradient paired with IFT sensitivities that use H_obs is not
the gradient of any function. The outer Newton then sees H ≠ ∇g, which on its own produces a
non-contracting decrement.

*Consequence.* The outer LAML must use H_obs + S at a certified strict mode, and the sensitivity
dβ̂/dρ = −(H_obs + S)^{−1}Ṡβ̂ must use the same matrix. Fisher is admissible only as an inner search
direction (Thm 3.11).

**Proposition 3.13 (indefinite H at the returned β) [P].** A C² local minimum requires
H_obs + S ⪰ 0, so a negative eigenvalue proves β is not a mode, and the LAML (a Laplace
approximation *around a mode*) is undefined. A zero eigenvalue at a stationary point means a flat
direction, i.e. non-identifiability, gauge or separation. There log|H| is −∞ and V is not a valid
criterion. The only certified responses are:
- continue the inner iteration (step 3 of Thm 3.11), or
- remove the gauge structurally (Prop 3.14).

Absolute-value eigenvalues, pseudo-spectral floors and jitter all change the objective and violate
SPEC.

### 3.8 Survival location-scale gauges

**Proposition 3.14 [P].** u = (h(t) − η_t)e^{−η_σ} is invariant under:
- (G1) (h, η_t) → (h + a, η_t + a);
- (G2) (h, η_t, η_σ) → (c·h, c·η_t, η_σ + log c) for c > 0.

With free intercepts in h, η_t and η_σ, the penalized Hessian has exactly two gauge null
directions at every β. The penalties scale as c² under G2, so the penalized problem has no finite
minimizer along G2 unless it is pinned. That is issue #2106, which the code pins by fixing the log-σ
constant column at `prepare.rs:504-534`; this is correct.

Consequence for scoring: after pinning, η_t is measured in h-units. The log-t-unit location is
μ(x) = (η_t(x) + b)/ā, where ā = the n-weighted mean of dh/d log t, and log σ_true = η_σ − log ā.
The test compares centred η_t to the log-t truth directly, which is gauge-dependent. **[C]** This
explains a scale factor of order ā (≈ 1.2 for Weibull shape 1.5) but not rmse = 5.09. The large
error needs a converged-inner certificate check (Cholesky of H_obs + S on the gauge-fixed chart),
listed in §7.

---

## 4. Numerical checks

All scripts are in `SP/theory/multinomial-ls/` (SP = the session scratchpad), run with
`SP/theory/venv/bin/python`.

| script | claim | result |
|---|---|---|
| `mn_laml.py` (sympy) | Thm 3.2 | K=3: (λ0λ1+λ0λ2+λ1λ2)/9; K=4: e₃/16 |
| `mn_laml.py` | Cor 3.3: finite limit at λ_c→0 | V = 301.67 (ρ₀=2), 309.497 (−4), 309.6041874 (−12), 309.6042233 (−16), 309.6042239 (−20); V(λ₀=0) = 309.6042239; ∂V/∂λ₀\|₀ = −5.95 |
| `mn_laml.py` | shared penalty is repelled | V = 306.36, 330.25, 354.25, 378.25 at ρ = 0, −4, −8, −12 (slope −6 = −r/2, r = 12) |
| `mn_laml.py` | Prop 3.1 | V(ref=2) = V(ref=0) = V(ref=1) = 302.284614023387 |
| `mn_cone.py` | Thm 3.4: V analytic across λ_c = 0; optimum over Λ vs over ρ | see §4.1 |
| `gauss_harmonic.py` | §3.5 replica | profile V(ρ_bend) monotone to a finite limit; Newton steps ≈ 1; λ̂² ratio ≈ e^{−1}; A ≈ 1.23e3 > 0 |
| `ls_convexity.py` (sympy) | Prop 3.8 | determinants: (μ, log σ): −2r²e^{−4η}; (ν, τ): τ^{−2}; (ν, log σ): y(y−μ)e^{−2s}; natural: 1/(2θ₂³) |
| `ls_convexity.py` | Prop 3.12 misspecified | mean log\|H_obs\| − log\|F\| ≈ −0.176 ± 0.02 for n ≥ 400 (O(1)) |
| `ls_fisher_wellspec.py` | Prop 3.12 well specified | mean → 0 (−0.105, −0.020, −0.0048, −0.0002, −0.0013); sd·√n ≈ 1.0–1.4 (O_p(n^{−1/2})) |
| `gauss_cc.py` | installed 0.1.267 has one ρ for cc | yes (λ ≈ 36–176, seeds 0–39). The pre-harmonic build, so it cannot reproduce the 3-ρ stall; hence the replica |

### 4.1 Cone optimum

Script: `SP/theory/multinomial-ls/mn_cone_cont.py`. It uses the K=3 equivariant penalty of §3.2 on the `mn_laml.py` DGP and slices along λ = (λ0, 1, 1). The PD-cone condition is e₂(λ) = 2λ0 + 1 > 0, so the slice leaves the positive orthant at λ0 = 0 but stays inside the cone Λ down to λ0 = −0.5.

| λ0 | e₂(λ) | V(λ) |
|---|---|---|
| +0.20 | 1.400 | 308.60445746 |
| +0.10 | 1.200 | 309.06208775 |
|  0.00 | 1.000 | 309.60422393 |
| −0.10 | 0.800 | 310.26881931 |
| −0.20 | 0.600 | 311.12702347 |
| −0.40 | 0.200 | 314.41315572 |

- **V is continuous and smooth across λ0 = 0.** The value at 0 equals the λ0 → 0⁺ limit from `mn_laml.py` (309.6042239). The one-sided difference quotients are −5.42 on (0, 0.1) and −6.65 on (−0.1, 0), and they bracket the analytic ∂V/∂λ0(0) = −5.95. This verifies Cor 3.3 and Thm 3.4: the face λ_c = 0 is not a boundary of the criterion, only of the ρ = log λ chart.
- **V diverges only at ∂Λ.** V rises steeply as e₂ → 0⁺ (from −0.2 to −0.4 V gains 3.3, versus 0.7 per 0.1 step near 0). This matches the log e₂(λ) barrier in the penalty log-determinant, so the λ-chart iteration of §6.2(A) stays inside Λ by the exact α_max step rule. No box is needed.
- On this DGP, V increases toward negative λ0 along the slice, so the optimum has λ0 > 0. Negative-λ_c optima inside Λ are possible in principle, since Λ is strictly larger than the orthant, but this example does not show one. The joint Nelder-Mead search over the full cone (the test-only part of `mn_cone.py`) did not finish within the time budget and was stopped. That search is listed in §7 as open.

Finite differences appear only in these test scripts. Production derivatives are the analytic
formulas of §3.4 and Thm 3.6.

---

## 5. Literature

- Wood, S. N., Pya, N. & Säfken, B. (2016). Smoothing parameter and model selection for general
  smooth models. *JASA* 111(516), 1548–1563.
  - §3.1 defines the LAML with the *observed* negative Hessian H at "a positive definite maximum".
  - The outer algorithm steps (c)–(f) perturb the ρ-Hessian and drop "indefinite" coordinates from
    the step set I, an ad hoc device that Thm 3.6 replaces with a KKT face test.
  - §3.1.2 describes the inner Newton with pivoted Cholesky of H + εI, ε increased until PD, step
    halving, diagonal preconditioning, and a rank test with the balanced penalty Σ S_j/‖S_j‖_F.
  - Location-scale and multivariate additive models are covered as examples.
- Rigby, R. A. & Stasinopoulos, D. M. (2005). Generalized additive models for location, scale and
  shape. *JRSS-C* 54(3), 507–554. The RS and CG algorithms (§3, App. B) are backfitting / Fisher
  scoring with no global convergence theorem. They motivate Thm 3.11 but should not be copied.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation
  of semiparametric GLMs. *JRSS-B* 73(1), 3–36. §3 gives the REML/LAML gradient and Hessian used in
  §3.4.
- Yee, T. W. & Wild, C. J. (1996). Vector generalized additive models. *JRSS-B* 58(3), 481–493.
  VGAM uses reference-coded multinomial with per-linear-predictor smoothing, i.e. a reference-class
  dependent penalty; §3.1 explains why gamfit should not.
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall. §4 covers
  the ALR and CLR charts and the centred metric M.
- Albert, A. & Anderson, J. A. (1984). On the existence of maximum likelihood estimates in logistic
  regression models. *Biometrika* 71(1), 1–10. The separation / overlap trichotomy, including the
  multinomial case (Thm 1); used in Prop 3.5. Konis, K. (2007). *Linear programming algorithms for
  detecting separated data in binary logistic regression models*. DPhil thesis, Oxford (the LP
  certificate).
- Gårding, L. (1959). An inequality for hyperbolic polynomials. *J. Math. Mech.* 8, 957–965.
  Renegar, J. (2006). Hyperbolic programs, and their derivative relaxations. *Found. Comput. Math.*
  6, 59–79. Hyperbolicity cones of e_k; used in Thm 3.4.
- Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear mixed models with one
  variance component. *JRSS-B* 66(1), 165–185. The probability mass of the REML estimate at the
  boundary and the score-type condition; the rank-1 case of Thm 3.6.
- Moré, J. J. & Sorensen, D. C. (1979). On the use of directions of negative curvature in a
  modified Newton method. *Math. Programming* 16, 1–20. Second-order convergence (Thm 3.11,
  step 3).
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer. Thm 3.2
  (Zoutendijk), §3.4 (modified Newton), §4 (trust regions).
- Nesterov, Yu. (2004). *Introductory Lectures on Convex Optimization*. Kluwer. §4.1,
  Thm 4.1.12–4.1.14 (self-concordant damped Newton). Boyd, S. & Vandenberghe, L. (2004). *Convex
  Optimization*, §9.6 (stopping bound f − f* ≤ λ²).
- Städler, N., Bühlmann, P. & van de Geer, S. (2010). ℓ1-penalization for mixture regression
  models. *TEST* 19, 209–256. The reparameterization (β/σ, 1/σ), which makes the Gaussian NLL
  convex; Prop 3.8(c).
- Tierney, L. & Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal
  densities. *JASA* 81, 82–86. The O(n^{−1}) Laplace error.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. Thm 10.3
  and 10.7 (Cholesky backward error and success criterion); used for δ_H and δ_A.
- Cartis, C., Gould, N. I. M. & Toint, Ph. L. (2011). Adaptive cubic regularisation methods for
  unconstrained optimization, Part I. *Math. Programming* 127, 245–295. ARC convergence assumes a
  finite minimizer, which fails on the asymptotes of §3.5.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80, 27–38.

---

## 6. Consequences for gamfit

### 6.1 mnlogit cluster (Gaussian helper stall)

- **Test.** `tests/quality/families/quality_vs_statsmodels_ordinal_mnlogit.rs:146-154` fits a
  Gaussian REML model just to obtain a design. Build the design with
  `build_term_collection_design` (as the LS tests do) instead. The failing REML fit is unrelated to
  the multinomial claim under test.
- **Solver (the real defect).** Asymptote coordinates in rho_optimizer (`newton_polish.rs`,
  `decrement_bands.rs`, `asymptote_certificate.rs`, `rail.rs`) must be certified by Thm 3.6, not by
  "the decrement contracts".

  Algorithm:
  1. For each coordinate with ∂_ρV < 0 (heading to +∞), compute A_j in closed form at the face
     fit β̂_∞. This takes one extra penalized solve with U_jᵀβ = 0 imposed, plus the Schur
     complement G_j.
  2. If A_j > δ_A, move ρ_j to the face. Otherwise ρ_j is interior and the ordinary Newton
     decrement applies.
  3. Certify the remaining coordinates by the Newton decrement of the face-restricted problem.

  Delete the requirement that a coordinate at ρ = 12.3 show quadratic decrement contraction.
- **Rail candidates.** Choose them by the sign of the compact derivative ∂_u V = −e^{ρ}∂_ρV. The
  log shows `railing [2]` (the strongly identified fundamental ridge, costing 0.135) being tried
  while coordinate 1 was never tested against its face.
- **Cross-reference.** The companion reports (boundary asymptotics, compactified coordinates)
  handle the general face calculus; A_j here is the explicit first-order face coefficient for
  quadratic penalties.

### 6.2 vgam / penguins cluster (equivariant multinomial penalties)

- **Where the specs are built.** `multinomial_reml.rs:1440-1530` (`equivariant_class_penalty_specs`)
  and `:3039` (`joint_penalty_specs`). Keep the specs, but **change the hyperparameter chart**. Two
  principled options:
  - **(A) λ-chart on the natural cone.** Hyperparameters λ_{t,·} ∈ Λ_t (Thm 3.4), not ρ = log λ.
    The domain constraint A_t(λ_t) ≻ 0 is mathematical (SPEC-allowed). The barrier is the
    criterion's own −½r_t log e_{K−1} term, so no extra barrier is needed. Steps are clipped by the
    exact α_max of §3.2 (a (K−1)×(K−1) eigenproblem), and "fraction to boundary" is not a constant:
    backtracking on V handles it, since V → +∞ at ∂Λ_t. Faces λ_c → ∞ are still handled by
    Thm 3.6 in u_c = 1/λ_c. Derivatives: ∂_λ = e^{−ρ}∂_ρ, from the same formulas as §3.4.
  - **(B) Shared metric** M ⊗ S_t (`centered_joint_penalty_specs`, `:1343`), with one ρ per term
    component. It is reference-invariant, repelled from λ = 0, has lower dimension, and matches
    the VGAM test comment `quality_vs_vgam_multinomial_softmax.rs:826-828`. Its ρ-chart is then the
    correct one.

  Recommendation: (B) by default, since it penalizes all classes symmetrically, recovers the null,
  and has a smaller D. (A) is an opt-in "class-specific smoothness" mode. For K = 3, option (A) is
  a full chart of Sym⁺(2).
- **Delete** `MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM = 24` (`multinomial.rs:170`) and
  `multinomial_formula_use_outer_hessian`. The exact outer Hessian is always available analytically
  (§3.4). Its cost is O(D²) trace products, each of which reuses H^{−1}Ṡ_j. Choosing it by a
  calibrated dimension is a magic constant.
- **Delete** `MULTINOMIAL_SEPARATION_ETA_THRESHOLD = 25` (`multinomial.rs:183`). Replace it with
  the Prop 3.5 LP pre-check on N_λ, which with the default ridges means intercepts only (an empty or
  pure class). On ∂Λ (option A) or at λ → 0, report "no finite optimum", or require the Jeffreys
  term.
- **Done (#4053):** `MULTINOMIAL_FORMULA_INNER_TOL = 1e-5` (justified by a measured plateau) is
  deleted. The inner joint-Newton solve now targets the caller's `tol` against its own derived KKT
  band, `max(tol·(1+max(‖∇L‖∞,‖Sβ‖∞)), band)`, where `band` is the f64 rounding band of the score
  and penalty-gradient evaluation (#2812) plus the measured gradient bands and settling
  certificate (#2976, #2977). The plateau it cited (KKT residual 2.8e-5 to 9.4e-5 with objective
  changes at 1e-11 relative) is the separation-regime ill-conditioning of Prop 3.5. If it recurs,
  it is to be cured by the chart (option B) or the LP refusal, not by a looser tolerance.
- **Done (#4053):** `MULTINOMIAL_OUTER_REML_TOL = 1e-7` is deleted, and so is the family-set
  relative continuation bar `outer_rel_cost_tol`. The outer ρ search uses the caller's `tol` for its
  per-coordinate bands and certifies against the Newton-decrement resolution τ_stat = 1/(2n)
  (#2954). The anchored-continuation ladder compares criterion values against the same τ_stat, in
  the criterion's own absolute units.
- **Remove** `multinomial_formula_penalty_scale` (`multinomial.rs:129`). It is not wrong, since a
  constant rescale of S only shifts ρ̂. But it exists to keep ρ inside boxes that SPEC forbids, and
  the "±10 box" / `effective_df_floor_rho_upper_bounds` comments (`multinomial.rs:145, 3537-3542`)
  document box-dependence that should go with the boxes.
- **Certificate for a multinomial fit:**
  1. inner: ‖∇f‖ ≤ δ_g, and the Cholesky of H succeeds with λ_min(H) > δ_H = c_p ε ‖H‖_2,
     c_p = p(p+1) (Higham Thm 10.7), on the identified ALR span;
  2. outer: every interior coordinate has Newton decrement ≤ the derived band; every face
     coordinate has A_j > δ_A;
  3. option (A): λ ∈ int Λ_t, i.e. the Cholesky of A_t(λ_t) succeeds.

### 6.3 Gaussian location-scale (`gamlss/gaussian/*`)

- **Keep:** the observed joint Hessian in the outer LAML (`location_scale.rs:917-929`), and
  `exact_newton_joint_hessian_beta_dependent = true`. This is correct per Prop 3.12.
- **Keep, with a certificate:** Fisher diagonal working weights as the inner *direction*
  (`location_scale.rs:1010-1019`). Add the Thm 3.11 termination test: Cholesky of H_obs + S_λ at
  the returned β, and a negative-curvature continuation when it fails. Verify that the IFT
  sensitivities dβ̂/dρ use H_obs + S (the same matrix as the log-determinant). Any mismatch makes the
  outer gradient inconsistent (Prop 3.12, last paragraph).
- **Delete:** the multistart seed configuration `max_seeds = 4, seed_budget = 2` and the
  risk-profile keep-best logic (`location_scale.rs:946-958`). SPEC bans grid search and seeds chosen
  by calibration. The documented reason ("capped screening … ranks an over-smoothed scale seed
  cheapest") is an inner-convergence defect. The cure is inner certification (Thm 3.11), not more
  seeds.
- **Build (optional model):** `sigma_link = "inverse"`, the (ν, τ) chart of Thm 3.9, whose inner
  convergence is a self-concordance theorem, with the domain {Zc > 0} enforced by the exact ratio
  test min_i {−z_iᵀc / z_iᵀd_c : z_iᵀd_c < 0}.
- **Existence pre-check:** Prop 3.10. Refuse when μ can interpolate the rows of a log-σ null
  direction, e.g. p_μ ≥ n.

### 6.4 How each tolerance is derived

- **δ_H (PD at the mode).** Cholesky of fl(H) succeeds ⇒ H + ΔH ≻ 0 with
  |ΔH| ≤ c_p ε |R̂ᵀ||R̂| (Higham Thm 10.3). Certify λ_min(H) > ‖ΔH‖_2 ≤ p(p+1)ε‖H‖_2 by one
  Rayleigh check on R̂. This gives no constant beyond ε and p.
- **δ_A (face coefficient).** A_j is a difference of two O(r_j) quantities, T = tr(Σ^{−1}G) and
  Q = gᵀΣ^{−1}g/φ̂. The forward error bound is
  |δA| ≤ ½ε[c_r κ(Σ_j)(T + Q) + c_p κ(H_∞) T], where the first part comes from the solves with
  Σ_j and the second from the Schur complement solve. Certify only if A_j > δ_A. Statistically, A_j
  is the REML score-test statistic (Thm 3.6), and its sign is what matters.
- **δ_g (inner KKT).** The forward error of the score sum, ‖δg‖ ≤ γ_n Σ_i |x_i||y_i − μ_i|_row + ε
  terms of S_λβ, measured in the H^{−1} norm so that the band is chart-invariant.
- **Outer Newton-decrement band.** From the errors of V_ρ and V_ρρ, which are themselves built from
  traces of H^{−1} products with the bounds above. See the floating-point companion report.

### 6.5 Survival LS cluster

- `survival/location_scale/prepare.rs:504-534`: the G2 pin is correct (Prop 3.14). The test
  (`quality_vs_gamlss_gaussian_survival_ls.rs` around the RMSE block) should compare gauge-invariant
  quantities: μ̃ = η_t/ā and log σ̃ = η_σ − log ā, with ā the mean log-t slope of h.
- Magic constants to replace, all in `survival/location_scale/constants.rs`:
  - `LEVENBERG_INITIAL_DAMPING_REL = 1e-8`, `LEVENBERG_DAMPING_GROWTH = 10`,
    `LEVENBERG_MAX_DAMPING_REL = 1e8` (lines 57, 59, 61): replace with Thm 3.11's modified Newton;
    no damping schedule is needed;
  - `SCALE_COUPLED_TRUST_METRIC_FLOOR_REL = 1e-6` (84): an affine-invariant Newton step needs no
    metric floor;
  - `BLOCKWISE_OUTER_MAX_ITER = 60` and `BLOCKWISE_OUTER_TOL = 1e-5` (90, 92): an iteration cap is
    banned; use the derived bands;
  - `REDUCED_AFT_NEWTON_STALL_TOL = 1e-4` (106): replace with the exact self-concordant or
    decrement band;
  - `STRUCTURAL_GUESS_RIDGE_REL = 1e-6` (112): a warm start does not need a ridge; use the
    minimum-norm least-squares solution.

  The real-data TIMEOUT is consistent with the capped blockwise loop grinding without a certificate
  **[C]**.

### 6.6 Which fix addresses which failure

| fix | failing cluster |
|---|---|
| Thm 3.6 face certificate + rail by sign of ∂_uV | mnlogit ARC stall; more broadly, all "decrement stopped contracting" asymptote cases |
| design-only construction in the test | mnlogit (removes the incidental REML fit) |
| chart (B) shared metric, or (A) cone λ-chart | vgam penguins (railed [1,2,5,6,7,11,14], λ_min = −23); vgam main arm (λ ≈ 1e10–1e11 is plausibly two class λ's at ∞, which collapse the term, since for K=3 γ₀, γ₁ ∈ null ⇒ γ₂ ∈ null) [C] |
| LP separation certificate | penguins "softest curvature at rounding band" |
| inner Cholesky + negative-curvature certificate; observed-H consistency | gamlss LS truth gaps, custom-LS METRIC_OFF |
| gauge-invariant scoring; delete constants.rs magic | survival LS METRIC_OFF / TIMEOUT (partial) |

---

## 7. Open problems

1. **Survival LS rmse(loc) = 5.09.** This is far beyond the G2 gauge factor. Needed next: dump β̂
   and the Cholesky of H_obs + S on the pinned chart for the seed-1234 data, and check whether
   the inner stopped at a saddle or on a residual G1/G2 direction. The G1 pin must be verified to be
   structural, not penalty-based. Undiagnosed.
2. **Real-data survival LS TIMEOUT.** Undiagnosed. The hypothesis is the capped blockwise loop
   (constants.rs:90-92) without an inner certificate.
3. **K > 3, option (A).** The λ-slice of Sym⁺(K−1) is K-dimensional in a K(K−1)/2-dimensional cone.
   Whether its REML optimum can lie on ∂Λ_t with a finite mode is not settled. The general
   statement "V → +∞ at ∂Λ" needs log|H| bounded below there, which holds without separation
   (Prop 3.5) but has not been proven uniformly.
4. **The exact mapping of the three mnlogit coordinates** (x1 ridge, bend, fundamental) is inferred
   from the penalty-construction order and the replica, not read from a main-built binary (none was
   available and cargo was not allowed).
5. **The no-go claim** "no smooth chart with an *additive log-σ* scale predictor makes the Gaussian
   NLL convex for all data" is proven for location charts μ and μ/σ (Prop 3.8 a, b) and conjectured
   in general.
6. **The O(n^{−1/2}) Fisher-LAML error** is shown to O_p order with a sketch and numerics. A sharp
   constant, and the induced bias in λ̂, remain to be derived.
7. **The penguins mechanism** (optimum over Λ_t at λ_c ≤ 0 for the railed coordinates) is inferred
   from the rail pattern and Cor 3.3. The penguins data were not refitted here.
8. **Full-cone optimum on the toy DGP.** §4.1 checks only the slice λ = (λ0, 1, 1). The joint
   minimization over Λ, a test-only Nelder-Mead run in `mn_cone.py`, was stopped before it
   finished. A Newton run in the λ-chart with the exact α_max rule (§3.2) is the right way to settle
   whether this DGP has a λ_c < 0 optimum.
