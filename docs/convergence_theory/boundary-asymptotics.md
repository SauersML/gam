# Boundary asymptotics of the REML/LAML criterion as ρ_j → ±∞

Lane: `boundary-asymptotics` (convergence theory team). Scope: the exact behaviour of the
outer criterion `V(ρ)` at `ρ_j = log λ_j → ±∞`, an exact one-sided optimality certificate at
infinity (faces and corners), and what that means for `rail_face_limit.rs`,
`asymptote_certificate.rs`, `rail.rs` and the tail-snap machinery in `run.rs`.

Status tags used throughout: **[P]** proven here (proof given or sketched to a checkable
level), **[P+N]** proven and numerically checked at 80–250 digits, **[N]** numerically checked
only, **[C]** conjectured.

Scripts (mpmath, venv `SP/theory/venv`), all under
`SP/theory/boundary-asymptotics/` (`SP` = the session scratchpad):
`crit.py` (brute-force criteria), `g1_gauss_one_penalty.py`, `g2_multi_penalty.py`,
`g3_laml_binomial.py`, `g4_gate_power.py`, `g5_lower_end.py`, `g6_separation.py`; outputs in the
matching `*.out` files.

---

## 1. Summary

* **The upper end is analytic, not merely asymptotic.** For Gaussian REML and for GLM LAML
  with a proper limit fit, `V` is a real-analytic function of `t_j = e^{−ρ_j}` at `t_j = 0`.
  It expands as `V = V_∞ + a_1 t + a_2 t² + …`, with no `e^{−ρ/2}` terms and no terms
  polynomial in ρ. For one Gaussian penalty the radius of convergence is exactly the smallest
  positive Demmler–Reinsch eigenvalue `s_min` of the pencil `(A, B)`. All closed forms for
  `V_∞`, `a_1` and `a_2` are given in the DR basis **[P+N]**. The code's
  `c = ½tr(A⁻¹C)` equals `a_1` exactly, both for Gaussian (which needs the
  `−Schur_Z(S_R)` pseudo-determinant term) and for binomial LAML (which needs the oblique
  drift term the code already has) **[P+N]**.
* **The tail-noise pathology comes from the formula, not the geometry.** The fp64 gradient
  `½(λβᵀSβ/φ + tr(H⁻¹λS) − r)` subtracts two O(r) numbers to get an O(e^{−ρ}) result. It
  returned exactly `0.000` at ρ=40 against a true value of 3.7e-17. An exact Schur-split
  gradient keeps full relative precision at every depth:
  `∂V/∂ρ_j = ½λ_jβᵀS_jβ/φ − ½tr((λ_jA_j+B_j)⁻¹B_j) + ½tr((λ_jA_j+B_j^S)⁻¹B_j^S)`.
  It matched the 120-digit brute force to 1e-87…1e-117 **[P+N]**. This single change removes
  three things: the "sign-corrupted curvature tie" (`run.rs:6561-6570`), the ĉ drift that
  `TAIL_SNAP_DRIFT_REL` papers over, and the need to probe.
* **Newton in ρ can never reach the upper face.** On the tail `−g/H → 1`, so every Newton
  step moves exactly one e-fold **[P+N]**. In `t = e^{−ρ} ≥ 0` the face is an ordinary bound
  constraint, and a projected Newton step lands on `t=0` in one step whenever `a_1 > 0`. The
  hand box of `rail.rs` should be replaced by this compactification. `t ≥ 0` is a domain
  constraint, not a hand bound.
* **An exact one-sided certificate exists at a single face.** It is second-order KKT at `t_F=0`:
  the face problem `V_∞(ρ_R)` must be stationary with a PD reduced Hessian, and every
  `c_j = ∂V/∂t_j > 0` **[P]**. The code instead requires `λ_min(C) > 0`
  (`rail_face.rs:374-383`). That condition is sufficient but massively over-refuses: on a true
  null face it accepts with probability `P(χ²_r < 1)`, which is 0.09 at r=4, 1.7e-4 at r=10
  and 1.7e-10 at r=20. The exact test accepts about 0.68 of the time **[P+N]**. The sharp
  replacement is simply `c_j > τ_j`.
* **At overlapping corners the per-coordinate test is wrong.** Take several penalties with
  intersecting ranges sent to ∞ together. Then `V − V_∞ = f(t) + O(|t|²)` with
  `f(t) = ½tr((Σ_j A_j/t_j)⁻¹C)`. This `f` is homogeneous of degree 1 but not linear, and
  positive vertex slopes do **not** imply `f > 0`. In the explicit counterexample both vertex
  slopes are +0.5 while `f = −4.5` at the simplex midpoint, and the brute force confirms
  `V < V_∞` along that ray **[P+N]**. The ρ-gradient tail at such a corner depends on the
  order in which the coordinates diverge (+0.5 or −19.5 in the example) **[P+N]**. The exact
  corner test is `min_{simplex} f > 0`:
  * if the ranges are independent, `f` is linear and the vertex test is exact;
  * for |F|=2, simultaneous congruence gives a univariate rational function and a Sturm test;
  * in general, Pólya/Powers–Reznick gives a finite certificate **[P]**.
* **The lower end (λ_j → 0) is classified by integers.** `∂V/∂ρ_j → ½[k_H − k_S + νκ_D]`,
  with `k_H ≤ k_S` **[P+N]**. In the generic case (`n > p`, X full rank) the slope is `−r/2`,
  so V → +∞: the end repels, is never a face, and needs no box. A finite lower limit
  (slope 0) occurs only in the interpolation, redundant-penalty and unidentified cases. There
  `V = V_{−∞} + b u + O(u²)` with a closed-form `b` **[P+N]**. The generic lower-rail law
  `∂V/∂ρ = +c e^{ρ}` in `asymptote_certificate.rs:26` is false except in those cases.
* **Separation gives a logarithmic tail.** Complete separation along a penalized direction
  produces `∂V/∂ρ ≈ m/(2ρ)` as ρ → −∞ **[N; coefficient C]**. This tail is not exponential:
  it passes any fixed gradient tolerance at finite ρ while V → +∞. It must be diagnosed
  directly (Albert–Anderson LP), not left to gradient tests.
* **Every run.rs tail constant is a magic constant, and every one has an exact replacement.**
  The constants are 18 e-folds, the (0.25, 4) curvature band, 1e-2 drift, 1e-4 estimand,
  0.5/6 local probes, and the window of 12, 3 samples, 1e-8, 1e-6 and 1e-3 in
  `asymptote_certificate.rs`. The replacements are:
  * the analytic `c_j` from `rail_face.rs`;
  * an FP-derived sign tolerance `τ_j`;
  * the exact estimand travel `β(t) − β_∞ = t·v + O(t²)`;
  * the regime-onset radius `s_min`.

  Probing (a derivative-free sampling of `−e^{ρ}g`) becomes unnecessary.
* **`M_p` in the face limit uses the wrong rank.** `rail_face.rs:1027-1037` uses `Σ_j rank S_j`,
  and its comment claims the criterion does the same. The dense criterion
  (`objective.rs:1361/1457`, `1776/1917`, via `penalty_logdet.rs:237`) uses the joint
  structural rank of `Σ S_j`, and REML theory requires the joint rank. For a 5×5 tensor
  product with second-order margins, `Σ rank = 30 > p = 25` gives `M_p = 0` (through
  `saturating_sub`) where the correct value is 4.

## 2. Setup and notation

Profiled Gaussian REML, as assembled in `reml_outer_engine/objective.rs`:

```
V(ρ) = D_p/(2φ) + ½(log|H| − log|S_λ|₊) + (ν/2) log(2πφ),   φ = D_p/ν,  ν = n − M_p,
H = K + S_λ,  K = XᵀWX,  S_λ = Σ_j λ_j S_j,  λ_j = e^{ρ_j},
D_p = ‖y − Xβ̂‖²_W + β̂ᵀS_λβ̂,   M_p = p − rank(Σ_j S_j).
```

LAML (fixed dispersion):
`V = −ℓ(β̂) + ½β̂ᵀS_λβ̂ + ½log|XᵀW(β̂)X + S_λ| − ½log|S_λ|₊`, where β̂ = β̂(ρ) is the penalized
mode.

**Face notation.** A face is `F ⊂ {1..m}`, the set of coordinates sent to +∞. The other
coordinates are `R`, with finite `ρ_R`.
* `Q` is an orthonormal basis of `range(Σ_{j∈F} S_j)`, with q columns. `Z` is an orthonormal
  basis of `N = ∩_{j∈F} null S_j`.
* `A_j = QᵀS_jQ` (PSD, and `Σ_F A_j ≻ 0`). `S_R = Σ_{k∈R} λ_k S_k`. `K̄ = K + S_R`.
* `B = Schur_Z(K̄) = Q ᵀK̄Q − QᵀK̄Z(ZᵀK̄Z)⁻¹ZᵀK̄Q`.
* `Schur⁺_Z(S_R)` is the same with the pseudo-inverse of `ZᵀS_RZ`.
* The limit fit is `β_∞ = Z(ZᵀK̄Z)⁻¹ZᵀXᵀWy`. The Q-score at the limit is
  `g_Q = Qᵀ(XᵀWy − K̄β_∞)`. Also `D_∞ = D_p(β_∞)` and `φ_∞ = D_∞/ν`.

**Compactified coordinates.** `t_j = e^{−ρ_j} ∈ [0, ∞)` at the upper end and
`u_j = e^{ρ_j} ∈ [0, ∞)` at the lower end.

**DR basis (one penalty).** Solve the generalized eigenproblem `A v = s B v` with `VᵀBV = I`
and `VᵀAV = diag(s_i)`, `s_i > 0`, i=1..r. Set `ỹ = Vᵀg_Q`. Equivalently, the `s_i` are the
positive eigenvalues of `S v = s K v`, and ỹ is the DR-rotated response restricted to
`range(S)`.

## 3. Results with proofs

### R1. One penalty, Gaussian: closed forms **[P+N]**

In the DR basis, `H` is diagonalised as `diag(1+λs_i)` on the r penalized coordinates.
With `μ = t = e^{−ρ}`:

```
log|H| − log|λS|₊ = log|K_ZZ| + Σ_i log(1 + μ/s_i)                 (using |K|·Πs_i/|A| = |K_ZZ|)
D_p(μ)            = D_∞ − Σ_i ỹ_i² μ/(μ + s_i)
V(μ)              = ν/2 + ½log|K_ZZ| + ½Σ_i log(1+μ/s_i) + (ν/2) log(2π D_p(μ)/ν).
```

Hence, with `u_1 = Σỹ_i²/s_i` and `u_2 = Σỹ_i²/s_i²`:

```
V_∞ = ν/2 + ½log|K_ZZ| + (ν/2) log(2πφ_∞)            (general face: + ½(log|K̄_ZZ| − log|ZᵀS_RZ|₊))
a_1 = ½ Σ_i (1 − ỹ_i²/φ_∞)/s_i  = ½tr(A⁻¹C),  C = B − g_Qg_Qᵀ/φ_∞
a_2 = −¼Σ_i 1/s_i² + u_2/(2φ_∞) − u_1²/(4νφ_∞²).
```

*Proof.* Expand `log(1+μ/s)` and `log(1 − δ)` with `δ = (μu_1 − μ²u_2 + O(μ³))/D_∞`. The
identity `|K|·Π s_i = |A|·|K_ZZ|` follows from `Π s_i = |A|·|QᵀK⁻¹Q|` and
`(QᵀK⁻¹Q)⁻¹ = B`, together with `|K| = |K_ZZ||B|`. ∎

*Check (g1, dps 120).* The DR `a_1` and the code-form `½tr(A⁻¹C)` agree to 20 digits
(−8.7859832581826449737). The relative error of `a_1μ` alone decays like μ: 2.5e-5, 1.1e-9,
2.3e-18 and 1.0e-35 at ρ = 10, 20, 40, 80. With `a_2μ²` added it decays like μ²: 6.6e-10,
1.4e-18, 5.8e-36 and 1.0e-70.

### R2. Radius of convergence **[P]**, checked via R1's error decay **[N]**

The expansion of R1 converges for `|μ| < s_min` and diverges beyond it. The term
`log(1+μ/s_min)` has a branch point at `μ = −s_min`. The other possible singularities are the
zeros of `D_p(μ) = D_0 + Σ ỹ_i² s_i/(μ+s_i)`, where `D_0 ≥ 0` is the unpenalized RSS.
* Its imaginary part is `−Im μ·Σ ỹ_i² s_i/|μ+s_i|²`, which is non-zero off the real axis. So
  all zeros are real.
* On `(−s_min, ∞)` every term is positive. So all zeros lie below `−s_min`: they interlace the
  poles of a secular equation (Golub 1973).

∎ Consequence: the one-term tail law has relative error `≈ |a_2/a_1|·e^{−ρ}` exactly. The law
is in force once `ρ > −log s_min + log(|a_2/a_1|/ε_target)`. That number is computable, which
the fixed "18 e-folds" (`ASYMPTOTE_PROBE_COUNT`, `run.rs:5836`) is not.

### R3. Exact gradient and Hessian tails; the fp64 floor **[P+N]**

From R1, `dV/dρ = −μ dV/dμ`. In code quantities this is

```
∂V/∂ρ = −½ μ [ tr((A+μB)⁻¹B) − g_Qᵀ(A+μB)⁻¹A(A+μB)⁻¹g_Q / φ(μ) ]
      = −a_1μ − 2a_2μ² − …,
∂²V/∂ρ² = a_1μ + 4a_2μ² + O(μ³),        H_ρρ / (−g_ρ) = 1 + 2(a_2/a_1)μ + O(μ²).
```

Each term in the bracket is O(1) and is multiplied by μ. Nothing cancels. For several
penalties the same block-inverse identity, applied to coordinate j with its own split
(`Q_j = range S_j`, `Z_j = null S_j`, `M = H − λ_jS_j`), gives

```
∂V/∂ρ_j = ½λ_jβ̂ᵀS_jβ̂/φ − ½tr((λ_jA_j + B_j)⁻¹B_j) + ½tr((λ_jA_j + B_j^S)⁻¹B_j^S),
B_j = Schur_{Z_j}(M),   B_j^S = Schur⁺_{Z_j}(S_{−j})   (pseudo-Schur within range ΣS).
```

*Proof.* `tr(H⁻¹λ_jS_j) = tr((H⁻¹)_{Q_jQ_j}λ_jA_j)` and
`(H⁻¹)_{Q_jQ_j} = (λ_jA_j + B_j)⁻¹`. Hence `tr(H⁻¹λ_jS_j) = r_j − tr((λ_jA_j+B_j)⁻¹B_j)`. The
same argument applies to the pseudo-inverse `S_λ⁺` on `range(ΣS)`. The two `r_j` terms cancel
**analytically**. ∎

*Check (g1).* The split gradient matches the brute-force gradient to 1e-117 (ρ=10) and 2e-87
(ρ=80).
* The fp64 trace-formula gradient, assembled the way the code assembles it, gives 3.989e-4
  and 1.811e-8 at ρ=10 and 20. At ρ=40 and 80 it gives **0.000e+00** against true values of
  3.73e-17 and 1.59e-34.
* The Hessian relation holds to 6e-9, 1e-17, 5e-35 and 9e-70.

In this instance `a_1 < 0` (the face is not a minimum), so `g > 0` and `H < 0`. This is the
sign content that the code's magnitude-only tie (`run.rs:6561-6570`) throws away.

**The fp64 floor.** The trace formula's absolute error is about `ε·r·κ(H)`. Once
`|a_1|e^{−ρ}` falls below that, the computed gradient is noise.
* It then has the right magnitude only by accident, and a random sign.
* The Hessian `λV_λ + λ²V_λλ` has the same `r − r` cancellation. That explains the code's own
  measurement on #2299 (`g = −1.040e-2`, `H_kk = −1.018e-2`).
* It also explains why ĉ "drifts".

The split form has only relative error `O(ε·κ(A_j + μB_j))`.

### R4. One coordinate to ∞ with the others finite (range intersections) **[P+N]**

With `F = {j}` and `S_R` possibly overlapping `range S_j`, R1 holds with
`C = Schur_Z(K̄) − Schur⁺_Z(S_R) − g_Qg_Qᵀ/φ_∞`. The middle term comes from
`−½log|S_λ|₊ = −½log|λ_jA_j + Schur⁺_Z(S_R)| − ½log|ZᵀS_RZ|₊`.

The cross-gradient tail is
`∂V/∂ρ_k − ∂V_∞/∂ρ_k = O(e^{−ρ_j})` for k ∈ R. It is analytic in `t_j` with an explicit
coefficient, `∂/∂ρ_k` of `a_1(ρ_R)`.

*Check (g2 B).* The joint rank is 8, the ranks are 6 + 4, and `M_p = 0`.
* `c_0 = −2678.0646`. Without the `Schur⁺_Z(S_R)` term it would be −2677.4605, which is wrong.
* The relative errors of the gap and the gradient are 7e-7, 1.5e-15 and 6e-33 at ρ = 20, 40, 80.
* The cross tail `(g_1 − g_1∞)/μ` converges to −30.029374.

### R5. Faces with several coordinates; overlapping corners **[P+N]**

Let `P(t) = (Σ_{j∈F} A_j/t_j)⁻¹` on `Q` (continuous, PSD, `‖P(t)‖ ≤ c|t|`). The Gaussian
criterion depends on the Q-block only through `P`:
* `β_Q = P(I + BP)⁻¹g_Q`;
* `log|H| − log|S_λ|₊ = const + log|I + PB| − log|I + PB^S|`.

So `V = G(P(t), ρ_R)` with G analytic at `P = 0`, `G(0) = V_∞` and `DG(0)[P] = ½tr(PC)`.
Therefore

```
V(t, ρ_R) = V_∞(ρ_R) + f(t) + O(|t|²)  uniformly,    f(t) = ½tr((Σ_j A_j/t_j)⁻¹ C).
```

* **Independent ranges ⇒ linear.** Suppose `rank Σ_F A_j = Σ_F rank A_j`. Write
  `A_j = M_jG_jM_jᵀ` with `M = [M_1 … M_k]` square and invertible. Then
  `(ΣA_j/t_j)⁻¹ = M⁻ᵀ blockdiag(t_jG_j⁻¹) M⁻¹`, so f is linear, `f(t) = Σ c_j t_j`. The slope
  `c_j = f(e_j) = ½tr(N_j(N_jᵀA_jN_j)⁻¹N_jᵀC)`, where `N_j` is a basis of `null(Σ_{k≠j}A_k)`.
  This is exactly `rail_face.rs:389-425`. Per-coordinate positivity is then exact.
* **Overlap ⇒ nonlinear.** `f` is degree-1 homogeneous and Lipschitz but not differentiable
  at 0. The vertex slopes `c_j = f(e_j)` do not control `f` on the simplex interior.
  * *Counterexample (g2 C).* `S_0 = diag(0,1,ε,0)` and `S_1 = diag(0,0,ε,1)` with ε = 0.05,
    and `diag C = (1, −2, 1)`. The vertex slopes are `c_0 = c_1 = 0.5 > 0`, but
    `f(0.1, 0.9) = −1.3`, `f(0.3, 0.7) = −3.7` and `f(0.5, 0.5) = −4.5`.
  * The brute force confirms `(V − V_∞)/(s f(d)) − 1 = O(s)` on each ray
    `t = s·d`: 5e-35 and 1e-34 at ρ=80. So V < V_∞ arbitrarily deep inside this corner even
    though both per-coordinate tail constants are positive.
* **Order dependence.** `∂V/∂ρ_0 = −t_0∂f/∂t_0 + O(|t|²)`, and `∂f/∂t_0` is homogeneous of
  degree 0, so it depends on the direction of approach. For the example the predictions are:
  * `ρ_1 = 2ρ_0` (λ_1 ≫ λ_0): `−e^{ρ_0}∂V/∂ρ_0 → C_11/2 = 0.5`;
  * `ρ_1 = ρ_0/2`: `−e^{ρ_0}∂V/∂ρ_0 → ½(C_11 + C_22/ε) = −19.5`.

  Measured: 0.4999772598, 0.499999999, 0.5, 0.5 for the first, and −19.19, −19.498,
  −19.49999992, −19.5 for the second.
* **Exact corner test.** The test is `m_F := min_{t ∈ Δ_F} f(t) > 0` **[P]**.
  * `C ⪰ 0` is sufficient (`f = ½tr(PC)` with `P ⪰ 0`). This is the code's gate.
  * If `C ⪰ 0`, f is concave. The map `t ↦ P(t)` is the parallel sum of the `t_jA_j⁺`, which
    is jointly matrix-concave (Anderson & Duffin 1969; Ando 1979). Its minimum over the
    simplex therefore sits at a vertex.
  * **Commuting penalties** (tensor-product marginals, `A_j = U diag(σ^{(j)}) Uᵀ`):
    `f(t) = ½ Σ_i C̃_ii / Σ_j (σ_i^{(j)}/t_j)` with `C̃ = UᵀCU`.
  * **|F| = 2 in general.** `A_0 + A_1 ≻ 0` on Q, so the pair is simultaneously congruent to
    diagonal form: `A_k = W D_k Wᵀ` with W invertible. Then
    `f(s, 1−s) = ½Σ_i (W⁻¹CW⁻ᵀ)_ii · s(1−s)/(d_{0i}(1−s) + d_{1i}s)`. Clearing the positive
    denominators gives a polynomial of degree ≤ q on [0, 1]. Its sign is decided exactly by a
    Sturm sequence plus the two vertex values.
  * **General |F|.** `f = Π_k t_k · tr(adj(M(t))C)/det M(t)`, with
    `M(t) = Σ_j A_j Π_{k≠j} t_k`, a homogeneous polynomial in t. `det M > 0` on the open
    simplex, so `f > 0` there iff the homogeneous numerator N(t) is > 0. Pólya's theorem
    (Hardy–Littlewood–Pólya 1952, §2.24) states: `N > 0` on Δ ⇔ `(Σt_k)^N·N(t)` has all
    coefficients positive for some N. Powers & Reznick (2001) bound N explicitly. The
    boundary faces of Δ are lower-dimensional corners and are treated recursively. This gives
    a finite, exact certificate.
  * **Zero vertex slope** ("Unidentified", `rail_face.rs:401-415`). If `range A_j ⊂
    range Σ_{k≠j}A_k`, then on the face `{t_k = 0, k≠j}` V is *exactly* independent of `t_j`.
    `ρ_j` is unidentified there, and reporting it as such (not as a certified value) is
    correct.

### R6. GLM LAML: β̂ moves with ρ **[P+N]**

Set `β_Q = tγ` in the `(Z, Q)` coordinates. The mode equations become `∇_Zℓ(β_Z, tγ) = 0`
and `Aγ = ∇_Qℓ(β_Z, tγ)`. At `t = 0` the Jacobian is `blockdiag(K_ZZ, A)`, which is
nonsingular when `K_ZZ ≻ 0`. By the real-analytic implicit function theorem (Krantz & Parks
2002), `(β_Z(t), γ(t))` is analytic. Hence β̂ is analytic in t, with `γ(0) = A⁻¹g_Q` and
`β_Z'(0) = −K_ZZ⁻¹K_ZQA⁻¹g_Q`. V is then analytic in t. The envelope theorem gives
`∂V/∂t|_0 = ½tr(A⁻¹B) − ½g_QᵀA⁻¹g_Q + ½tr(K_ZZ⁻¹ dK_ZZ/dt)`, and the last term equals
`d̃_QᵀA⁻¹g_Q`. So

```
c = ½tr(A⁻¹C),   C = Schur_Z(K) − g_Qg_Qᵀ + g_Qd̃_Qᵀ + d̃_Qg_Qᵀ,
d = ½Xᵀ(w′ ⊙ a),  a_i = x_iᵀZ(ZᵀKZ)⁻¹Zᵀx_i,   d̃ = d − KZ(ZᵀKZ)⁻¹Zᵀd   (oblique reduction),
```

This is exactly `laml_rail_face_limit` in `rail_face.rs:1113-1255`.

*Check (g3, binomial, n=60, p=6, r=4, dps 140).*
* `c = 0.96269887193`. The relative errors of the gap and gradient are 8e-9, 1.7e-17 and
  7e-35 at ρ = 20, 40, 80.
* `(V − V_∞ − cμ)/μ²` converges to 3.8605895698, which confirms analyticity.
* `‖β_Q‖/μ → 2.5308977`.
* Dropping the drift term gives c = 0.7877, and using the un-reduced `d_Q` gives c = 0.5569.
  Both are wrong.
* `λ_min(C) = −2.50` although `c > 0`, so the code's gate refuses a genuine one-sided
  minimum.

*Conditions.* The limit fit on `N` must exist: no separation within `N`, `K_ZZ ≻ 0`, and
ranks constant near `t = 0`. If separation exists within `N`, no fit exists at any λ, so this
is not a boundary phenomenon.

### R7. Newton in ρ vs projected Newton in t **[P+N]**

On the tail, the Newton step in ρ is `−g/H = 1 − 2(a_2/a_1)μ + O(μ²)`. Pure Newton therefore
advances exactly one e-fold per iteration and never reaches the face. This is the "grind" that
`try_tail_snap_to_rail` exists to kill.

In t, `∂V/∂t = a_1 + 2a_2t + …` is an ordinary smooth derivative at a bound. The projected
Newton / active-set method of Bertsekas (1982) behaves as follows:
* An index with `t_j ≤ ε_k` and `∂V/∂t_j > 0` enters the active set.
* It is then fixed at `t_j = 0` in one step.
* The method keeps superlinear convergence in the free coordinates.

Chain rule: `∂V/∂t_j = −e^{ρ_j}∂V/∂ρ_j` and `∂²V/∂t_j² = e^{2ρ_j}(H_jj + g_j)`. With R3
supplying both at relative precision, no cancellation re-enters.

### R8. The certificate at infinity **[P]**

**Theorem (disjoint/independent faces).** Assume the following on a neighbourhood of
`(t_F, ρ_R) = (0, ρ_R*)` in `[0,∞)^F × ℝ^R`:
(i) V is C², which holds by R1/R5/R6 under the R6 conditions;
(ii) `∇V_∞(ρ_R*) = 0` and `∇²V_∞(ρ_R*) ≻ 0`;
(iii) `c_j = ∂V/∂t_j(0, ρ_R*) > 0` for every j ∈ F.

Then `(0, ρ_R*)` is a strict local minimizer on the closed domain. Every `ρ` with
`ρ_R` near `ρ_R*` and `ρ_F` large enough has `V(ρ) > V_∞(ρ_R*)`.

*Proof.* Taylor expansion with the multiplier-free KKT structure gives
`V ≥ V_∞(ρ_R*) + ½κ|Δρ_R|² + Σ_j t_j(c_j − L|Δρ_R|) − M|t|²`. Every bracket is positive
near the point. Equivalently this is second-order sufficiency with strict complementarity
(Nocedal & Wright 2006, Thm 12.6): the critical cone is `{d_F = 0}`, and on it the reduced
Hessian is `∇²V_∞`. ∎

* **Overlapping F.** Replace (iii) by `m_F > 0` (R5). The same proof works with `f(t) ≥ m_F|t|_1`.
* **If some `c_j = 0`.** The first-order test is inconclusive. For |F| = 1 the condition
  `a_2 > 0` makes the point a strict local minimum, and `a_2 < 0` rules it out.
* **The corner is part of the problem, not an exception.** The test is a KKT test on the
  compactified domain `[0,∞]^m`. It has the same logical standing as an interior
  `∇V = 0, ∇²V ≻ 0` certificate.

**Power of the gate [P+N] (g4).** Take a true null face, known φ, and the DR frame. Then
`g_Q/√φ = z ~ N(0, I_r)`, `C = I − zzᵀ` and `c = ½Σ(1−z_i²)/s_i`.
* The code gate `λ_min(C) > 0` is equivalent to `‖z‖² < 1`. Its acceptance probability is
  `P(χ²_r < 1)`: 0.683, 0.394, 0.090, 1.75e-3, 1.72e-4, 1.7e-10 and 2.4e-25 for
  r = 1, 2, 4, 8, 10, 20, 40.
* The exact test `c > 0` accepts with probability 0.682 for `s_i = i⁴` at every r (the
  sum is dominated by the lowest mode). For a flat spectrum it accepts 0.63 down to 0.53.

The first-order event `c ≤ 0` is the "λ̂ < ∞" event whose probability Crainiceanu & Ruppert
(2004) compute for one variance component. It is the finite-sample version of the Self &
Liang (1987) boundary mass.

### R9. The lower end, ρ_j → −∞ **[P+N]**

Let `u = e^{ρ_j} → 0` with the other coordinates fixed. Then
`∂V/∂ρ_j → ½[k_H − k_S + νκ_D]`, where:
* `k_H` is the number of eigenvalues of H that vanish like u. It equals
  `dim null(K + S_{−j})`, the directions only S_j controls.
* `k_S = rank(ΣS) − rank(S_{−j})`, the rank S_j adds to the pseudo-determinant.
* `κ_D = 1` iff `D_p → 0` like u (exact fit). `κ_D = 0` for LAML without a scale.

*Proof.* Each of `log|H|`, `log|S_λ|₊` and `log D_p` is `(count)·log u + analytic`. ∎

**`k_H ≤ k_S`** [P]. H ≻ 0 for u > 0 forces `null(K+S_{−j}) ∩ null S_j = 0`. Since
`null(K+S_{−j}) ⊂ null S_{−j}`, we get
`N ⊕ (null S_{−j} ∩ null S_j) ⊂ null S_{−j}`, which is the claim.

| case | k_H | k_S | κ_D | slope | behaviour |
|---|---|---|---|---|---|
| (a) n > p, X full rank, one penalty of rank r | 0 | r | 0 | −r/2 | V → +∞ linearly: repelling, never a face |
| (b) interpolation n < p, S ≻ 0 | p−n | p | 1 | 0 | finite limit, `V = V_{−∞} + bu + O(u²)` |
| (c) redundant penalty, range S_j ⊂ range S_{−j} | 0 | 0 | 0 | 0 | finite limit, analytic in u |
| unidentified (S_j invisible) | – | – | – | 0 | V exactly flat |
| noise-free y ∈ X·null(S_{−j}) | | | 1 | > 0 possible | V → −∞: degenerate exact fit |

Closed forms **[P+N]**:
* (b): `b = ½tr(G⁻¹) − (n/2)·yᵀG⁻²y/yᵀG⁻¹y`, with `G = XS⁻¹Xᵀ`, and
  `V_{−∞} = ½log|G| + (n/2)log(2π yᵀG⁻¹y/n) + n/2`.
* (c): `b_j = ½[β̂ᵀS_jβ̂/φ + tr(H⁻¹S_j) − tr(S_{−j}⁺S_j)]`, evaluated at the `λ_j = 0` fit.

*Checks (g5).*
* (a): slopes −1.07, −1.99935, −2.0, −2.0 at ρ = −10, −20, −40, −80 (r = 4).
* (b), smooth y: b = 1.8668e7 > 0, so λ = 0 is a one-sided local minimum. The relative
  error of `bu` is 1.6e-10 at ρ=−40 and 6.7e-28 at ρ=−80.
* (b), rough y: b = −1.2506e6.
* In case (b) the tail regime starts only at `ρ ≲ log γ_min(G)`. At ρ=−10 the one-term law is
  still off by 99.9%, and at ρ=−20 by 7%.
* (c): b_1 = 1.026707. The relative errors are 3.6e-5, 1.6e-9, 3.4e-18 and 1.4e-35.

**Consequence.** The law `∂V/∂ρ = +c e^{ρ}` (`asymptote_certificate.rs:26`) holds exactly in
the slope-0 cases (b) and (c), with `c = b` when b > 0. In the generic case (a) it is false.
There the gradient tends to the constant `−(r−k_H)/2`, so a coordinate pinned at a lower box
with `g ≈ −r/2` is simply not at an optimum. The box put it there.

### R10. Separation (LAML lower end) **[N], coefficient [C]**

Consider complete separation along a penalized direction, binomial with a ridge penalty.
* **p = r = 1.** `β̂ ≈ (1/a)log(1/λ)` (β̂ = 49.6, 111.4, 239.1 at ρ = −20, −40, −80). The
  curvature `XᵀWX ≈ aλβ̂` in that direction. So `V ≈ ½log|ρ| + const → +∞`, and
  `∂V/∂ρ ≈ 1/(2ρ)`: measured −0.02702, −0.013554, −0.006625 against predictions −0.025,
  −0.0125, −0.00625. The ratio tends to 1 with `log|ρ|/ρ²` corrections.
* **p = 2** (x_1 separating, x_2 noise). Complete separation drives every `w_i → 0`, so the
  noise direction also loses its likelihood curvature. The slope is then not −½ but about
  `≈ 0.886/ρ` (−0.01107 at ρ = −80).
* **Conjecture.** `∂V/∂ρ = −(r−k_H^{sep})/2 + m/(2ρ) + o(1/ρ)`. Here `k_H^{sep}` counts the
  penalized directions invisible to the non-separated observations, and m is the number of
  directions whose curvature collapses to O(λβ̂).

The practical point is proven by the numbers. This tail is algebraic in ρ. A fixed gradient
tolerance τ is met at `|ρ| ≈ m/(2τ)`, well inside any box, while V → +∞ and β̂ diverges.
Separation must be detected combinatorially (Albert & Anderson 1984, the LP feasibility test
for a separating direction), never inferred from a small gradient.

### R11. Exact estimand travel **[P+N via R6's ‖β_Q‖/μ]**

`β̂(t) − β_∞ = t·v + O(t²)`, with `v = (Q − Z K_ZZ⁻¹K_ZQ)A⁻¹g_Q` (Gaussian: `/φ` absorbed
in g_Q scaling). This replaces the geometric estimate `‖Δβ_last‖·q/(1−q)` in
`asymptote_certificate.rs:308` and its caller.

With the t-compactification, the certified point **is** the face, so `β̂ = β_∞` exactly and
no travel tolerance is needed. v is only needed to report the local sensitivity.

### R12. FP tolerance for the sign of c_j **[P]**

A computed C (Schur complement via a Cholesky solve with `K_ZZ`, then rank-one updates)
satisfies

```
‖ΔC‖₂ ≤ γ_q [ ‖K‖(1 + κ(K_ZZ)) + ‖g_Q‖²/φ + 2‖g_Q‖‖d̃_Q‖ ],   γ_q = q·u/(1 − q·u),
```

by standard backward-error bounds (Higham 2002, Thms 10.3, 10.4).
For `A_j ≻ 0`, `|tr(A_j⁻¹ΔC)| ≤ tr(A_j⁻¹)‖ΔC‖₂`. Including the error of the `A_j`
factorization gives

```
τ_j = ½ tr(A_j⁻¹) ‖ΔC‖₂ + |ĉ_j|·γ_q κ(A_j).
```

The sign of `c_j` is a measured fact iff `|ĉ_j| > τ_j`. `curvature_margin` at
`rail_face.rs:374` already has this form for `λ_min(C)`. It is principled; it is attached to
the wrong statistic. For overlapping faces the same bound propagates to `m_F` through the
Sturm or Pólya coefficients. Those use rational arithmetic on rounded inputs, so only the input
rounding enters.

### R13. The dimension M_p **[P]**

`log|S_λ|₊` has `rank(Σ_j S_j)` terms for generic λ, and the REML marginal likelihood
integrates over `p − rank(ΣS_j)` improper directions (Harville 1977; Wood 2011). Hence
`ν = n − p + rank(Σ S_j)`. `Σ_j rank S_j` over-counts whenever the ranges intersect. For a
tensor product of two 5-dimensional second-order marginals this is 30 against a joint rank of
21 at p = 25.

## 4. Numerical checks (summary table)

All checks are brute force from the definitions in mpmath (dps given), compared with the
closed forms. Here `err(k)` means the relative error of the k-term expansion.

| script | claim | ρ=10 | ρ=20 | ρ=40 | ρ=80 |
|---|---|---|---|---|---|
| g1 (dps 120) | err(1) of `a_1μ` | 2.5e-5 | 1.1e-9 | 2.3e-18 | 1.0e-35 |
| g1 | err(2) of `a_1μ + a_2μ²` | 6.6e-10 | 1.4e-18 | 5.8e-36 | 1.0e-70 |
| g1 | split gradient vs brute | 1e-117 | 1e-113 | 5e-106 | 2e-87 |
| g1 | fp64 trace gradient (true value) | 3.989e-4 (3.989e-4) | 1.811e-8 (1.811e-8) | **0** (3.7e-17) | **0** (1.6e-34) |
| g1 | Hessian `a_1μ + 4a_2μ²` | 6e-9 | 1e-17 | 5e-35 | 9e-70 |
| g2 A (dps 250) | `c = 0.98 > 0` with `λ_min(C) = −4`; gap err | 2.3e-5 | 1.1e-9 | 2.2e-18 | 9.2e-36 |
| g2 B | overlap, `c_0 = −2678.0646`; gap err | 1.5e-2 | 7.1e-7 | 1.5e-15 | 6.2e-33 |
| g2 B | cross tail `(g_1 − g_1∞)/μ` | −28.71 | −30.0293 | −30.029374 | −30.029374 |
| g2 C | corner, `(V−V_∞)/(s f(d)) − 1` at d=(.5,.5), f=−4.5 | −2.9e-4 | −1.3e-8 | −2.8e-17 | −1.2e-34 |
| g2 C | order dependence, `−e^{ρ_0}g_0` (pred. 0.5 / −19.5) | 0.49998 / −19.19 | 0.5 / −19.498 | 0.5 / −19.5 | 0.5 / −19.5 |
| g3 (dps 140) | LAML `c = 0.962698872`; gap err | 1.8e-4 | 8.3e-9 | 1.7e-17 | 7.2e-35 |
| g3 | `(V − V_∞ − cμ)/μ²` | 3.85901 | 3.8605895 | 3.86058957 | 3.86058957 |

Lower end (g5, dps 120), at ρ = −10, −20, −40, −80:
* (a) slope: −1.070, −1.99935, −2.0, −2.0.
* (b) smooth y, `bu` relative error: −0.999, −0.071, −1.6e-10, −6.7e-28.
* (c) `b_1u` relative error: −3.6e-5, −1.6e-9, −3.4e-18, −1.4e-35.

Separation (g6, dps 80) is shown in R10. Gate power (g4) is shown in R8.

## 5. Literature

* Demmler, A. & Reinsch, C. (1975). Oscillation matrices with spline smoothing.
  *Numer. Math.* 24, 375–382. The DR basis used in R1–R2.
* Harville, D. A. (1977). Maximum likelihood approaches to variance component estimation
  and to related problems. *JASA* 72, 320–338. The REML dimension count (R13).
* Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood
  estimation of semiparametric generalized linear models. *JRSS B* 73(1), 3–36. The
  REML/LAML criterion and outer Newton in ρ. mgcv handles "λ → ∞" heuristically (it treats
  parameters as fixed when the criterion is flat). R3/R7 make that exact.
* Wood, S. N., Pya, N. & Säfken, B. (2016). Smoothing parameter and model selection for
  general smooth models. *JASA* 111(516), 1548–1563. LAML for general families.
* Reiss, P. T. & Ogden, R. T. (2009). Smoothing parameter selection for a class of
  semiparametric linear models. *JRSS B* 71(2), 505–523. REML vs GCV behaviour and REML's
  point mass at the linear fit (λ = ∞).
* Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators
  and likelihood ratio tests under nonstandard conditions. *JASA* 82(398), 605–610. Boundary
  mass.
* Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear mixed models
  with one variance component. *JRSS B* 66(1), 165–185. Exact probability of λ̂ = ∞. Our
  first-order event `c ≤ 0` (R8) is its local form.
* Albert, A. & Anderson, J. A. (1984). On the existence of maximum likelihood estimates in
  logistic regression models. *Biometrika* 71(1), 1–10. Separation (R10).
* Anderson, W. N. & Duffin, R. J. (1969). Series and parallel addition of matrices.
  *J. Math. Anal. Appl.* 26, 576–594. Parallel sum, and the concavity used in R5.
* Ando, T. (1979). Concavity of certain maps on positive definite matrices and applications
  to Hadamard products. *Linear Algebra Appl.* 26, 203–241. Joint concavity of the parallel
  sum.
* Golub, G. H. (1973). Some modified matrix eigenvalue problems. *SIAM Review* 15(2),
  318–334. Secular-equation interlacing (R2).
* Hardy, G. H., Littlewood, J. E. & Pólya, G. (1952). *Inequalities*, 2nd ed., CUP, §2.24.
  Pólya's theorem. Also Pólya, G. (1928), Über positive Darstellung von Polynomen,
  *Vierteljschr. Naturforsch. Ges. Zürich* 73, 141–145.
* Powers, V. & Reznick, B. (2001). A new bound for Pólya's theorem with applications to
  polynomials positive on polyhedra. *J. Pure Appl. Algebra* 164(1–2), 221–229.
* Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple
  constraints. *SIAM J. Control Optim.* 20(2), 221–246. The t ≥ 0 active-set step (R7).
* Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Springer,
  Thm 12.6 (second-order sufficient conditions). Used in R8.
* Krantz, S. G. & Parks, H. R. (2002). *A Primer of Real Analytic Functions*, 2nd ed.,
  Birkhäuser. Real-analytic implicit function theorem (R6).
* Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
  Ch. 10. Cholesky backward error (R12).

## 6. Consequences for gamfit

Paths are relative to `crates/gam-solve/src/`. Line numbers are for the working tree at
`486fd7441a`.

### 6.1 Replace the ρ-box rail with the t-compactification (build)

* `rho_optimizer/rail.rs:96-99` (`is_railed`: `theta <= lo+margin || theta >= hi-margin`)
  and `bridges.rs:3809` (`coordinate_rail_margin`, `CERTIFICATE_RAIL_MARGIN`) together form
  a hand box on hyperparameters. **Delete** them as the rail criterion.
* **Build** a projected Newton / active-set outer step in the mixed coordinates:
  * `ρ_j` stays free in the interior;
  * a coordinate whose split gradient (6.2) shows it on an upper tail is represented by
    `t_j = e^{−ρ_j} ≥ 0`, with the bound active at 0 (Bertsekas 1982);
  * a coordinate is on the upper tail when `∂V/∂t_j > 0` and `t_j` is inside the analytic
    radius R2.
* A step lands on the face exactly (R7). The lower end needs no box: in the generic case V
  → +∞ there (R9a). In cases (b) and (c), `u_j = e^{ρ_j} ≥ 0` is the matching domain
  constraint.

### 6.2 Cancellation-free derivatives (build; the root fix)

Implement R3's split gradient
`∂V/∂ρ_j = ½λ_jβᵀS_jβ/φ − ½tr((λ_jA_j+B_j)⁻¹B_j) + ½tr((λ_jA_j+B_j^S)⁻¹B_j^S)`
and its ρ-derivative for the Hessian diagonal. Use them wherever `λ_j` is large relative to
`‖K‖/‖S_j‖`. The switch point is not a constant: use the split form whenever
`ε·r_j·κ(H) > |computed g_j|·(rel. target)`, or simply always (it is exact everywhere). This
removes:
* the sign-corrupted curvature tie (`run.rs:6561-6570`); the sign becomes a valid datum;
* the ĉ drift (`TAIL_SNAP_DRIFT_REL`, `run.rs:6496`);
* the "no finite-difference-clean tail run" declines in the test log.

### 6.3 Delete the probing/tail machinery; certify analytically

**Delete** the following constants and code paths:
* `run.rs:5823 ASYMPTOTE_ESTIMAND_REL_TOL = 1e-4`;
* `5836 ASYMPTOTE_PROBE_COUNT = 18` ("18 e-folds", justified by one fixture's 13–16 e-fold
  band);
* `5848-5849 ASYMPTOTE_LOCAL_PROBE_DELTA = 0.5 / COUNT = 6`;
* `6435 TAIL_SNAP_CURVATURE_BAND = (0.25, 4.0)`;
* `6496 TAIL_SNAP_DRIFT_REL = 1e-2`;
* `try_tail_snap_to_rail` (`6498-`), including `deep_enough` (`6549-6560`) and
  `extrapolated_gap = ĉ·e^{∓ρ}` (`6638-6649`);
* `probe_tail_window[_at_resolution]` (`6980`, `7030`) and the probe loops;
* in `asymptote_certificate.rs`: `DEFAULT_ASYMPTOTE_WINDOW = 12` (81),
  `MIN_TAIL_SAMPLES = 3` (86), `EXP4_INTERIOR_GRAD_TOL = 1e-8`,
  `EXP4_TAIL_NOISE_FLOOR = 1e-6` and `EXP4_TAIL_DRIFT_REL = 1e-3` (221-230), and
  `coef_step_ratio` (308) with its `q/(1−q)` travel bound.

Probing is a derivative-free sampling of `−e^{ρ}g` at ρ + k. It violates the no-FD /
no-derivative-free rule in spirit, and every tolerance attached to it is fixture-tuned.

**Replacement.** The certificate is R8, evaluated at the face point itself:
* the face value `V_∞` and face gradient/Hessian in `ρ_R` (`rail_face.rs`,
  `gaussian_rail_face_limit` 956-1076 and `laml_rail_face_limit` 1113-1255, which are already
  correct except for M_p);
* the analytic `c_j` (already computed at `rail_face.rs:389-425`), accepted iff
  `c_j > τ_j` (R12);
* estimand travel is exactly zero at the face. Report `v` from R11 as sensitivity.

If the "how deep is the tail" question is still needed diagnostically, the answer is R2:
`ρ_onset = −log s_min + log(|a_2/a_1|/ε)`, with `a_2` from R1 or from one extra derivative of
the split formula.

### 6.4 Fix the face gate (edit `rail_face.rs:364-383`, 389-436)

* Replace `λ_min(C) > curvature_margin` as the **necessary** gate. Keep it only as a fast
  sufficient path.
* When the face coordinates have independent ranges (`rank ΣA_j = Σ rank A_j`), accept iff
  every `c_j > τ_j`. That is exact (R5).
* With overlap, decide `min_Δ f > 0`:
  * commuting `A_j` (tensor products): the explicit formula `½Σ_i C̃_ii/Σ_jσ_i^{(j)}/t_j`;
  * |F| = 2: simultaneous congruence, then a Sturm sequence on the degree-≤q polynomial;
  * |F| ≥ 3: Pólya with the Powers–Reznick N, done recursively over the sub-faces.
* The error message "releasing the face would not raise the criterion" (377-381) is false
  when C is indefinite but the `c_j > 0` (R8, g2 A, g3). This gate is exactly why binomial
  and Gaussian faces with r ≥ 4 are refused almost always (R8 table).
* The `c_j ≤ 0` branch comment at 419-421 ("can only be reached by … collapsed geometry") is
  only true under the old gate. Under the new gate `c_j ≤ τ_j` is the ordinary statistical
  outcome "λ_j = ∞ is not optimal".
* With `c_j = 0` and |F| = 1, decide on `a_2`. Keep the Unidentified typing
  (`rail_face.rs:401-415`): it is correct (R5, last bullet).

### 6.5 Fix M_p (`rail_face.rs:1027-1037`)

Use the joint structural rank of `Σ_j S_j`, obtained from the same routine the criterion
uses: `penalty_logdet.rs:237 structural_rank_from_canonical_penalties` →
`balanced_penalty_structural_rank`. Correct the comment. As written it asserts that the
criterion counts `Σ rank`. The dense criterion (`reml/objective.rs:1361→1457` and
`1776→1917`, passed on via `assembly.rs:374`) does not.

Note also that `reml_outer_engine/inner_solution.rs:629-636`'s *default* (no override) is the
Σ-rank count. Any path that reaches it without the override computes a different ν from the
dense path. That is a separate latent inconsistency, and it should become the joint rank too.

### 6.6 The lower end (`asymptote_certificate.rs:26`, `AsymptoteSide::Lower`)

* **Delete** the generic lower law.
* **Build** the integer classification of R9 from structural ranks: `k_S` from ranks and
  `k_H = dim null(K + S_{−j})`.
* If the slope is negative, the lower end is repelling, and a coordinate there with
  `g < 0` is simply unconverged. This covers the iso-kappa cluster's lower-railed coordinates
  with `|Pg|` = 3.6, 0.33 and 0.028, if those are ρ rather than ψ coordinates.
* Slope 0 means cases (b) and (c): certify the face `u_j = 0` iff `b_j > τ`, with the R9
  closed forms.
* Slope > 0 is the degenerate exact fit: report "criterion unbounded below", with no fit.

### 6.7 Separation diagnosis (build)

Before any lower-end decision in a binomial, multinomial or ordinal family, run the
Albert–Anderson LP check. It asks whether a direction `d` with `z_i x_iᵀd ≥ 0` for all i
exists, and restricts to the penalized subspace for the R10 case. If separated, report the
separating direction. Never certify via a small gradient: the `m/(2ρ)` tail satisfies any
fixed tolerance.

### 6.8 How the test-log clusters map (`SP/q1561/all-tests.log`, 13 "did not certify")

* **Standard REML, BFGS `line_search_failed`** with coordinate #2 railed at θ = 22.73 and
  `|Pg| = 2.28e-5 > 7.3e-6`. This is the fp64 floor (R3) sitting at the box, which is the
  wrong object (6.1). Fixed by 6.1 + 6.2.
* **ARC Newton-decrement stall** ("railing [2] changes criterion by 1.348e-1"). Pure Newton in
  ρ at one e-fold per step (R7). Fixed by 6.1.
* **Survival transformation** ("tail-snap declined: … 18 e-folds inside the box …
  |ratio| = 4.961e1 outside tie band"). Both declines come from magic constants (6.3). The
  ratio 49.6 means the coordinate is not on an exponential tail at all. R2's onset radius says
  so exactly, without a band.
* **Iso-kappa lower-railed**, and **custom family with many lower-railed coordinates, H not
  PSD**: see 6.6. With a negative lower slope those coordinates are unconverged, not railed.
* **Binomial and Gaussian rail refusals with indefinite C**: see 6.4.

## 7. Open problems

1. **|F| ≥ 3 non-commuting overlapping corners.** The Pólya degree bound can be large.
   Is there a cheaper exact test, such as an SDP/SOS certificate with exact rational
   rounding, or a structural reduction using the Kronecker structure of tensor-product
   penalties beyond the commuting case? **[open]**
2. **The separation coefficient m in R10.** p=1 gives m=1 (checked), and p=2 gives about
   1.77 at ρ=−80 while still drifting. Prove `∂V/∂ρ = −(r−k_H^{sep})/2 + m/(2ρ) + o(1/ρ)`
   and identify m. **[C]**
3. **LAML radius of convergence.** For Gaussian it is `s_min` exactly (R2). For LAML it is the
   nearest complex singularity of the analytic IFT branch. It is bounded below by quantities
   involving `s_min` and the Lipschitz constant of W. No sharp formula yet. **[open]**
4. **Interaction with ψ coordinates** (Matérn κ, anisotropy, link/transformation
   parameters). `run.rs:6526-6533` rejects ψ coordinates from tail-snap. The ψ → boundary
   limits (κ → 0/∞) need their own asymptotics; other lanes cover Matérn identifiability.
   Joint faces where a ρ goes to ∞ while ψ moves are not analyzed here. **[open]**
5. **Non-constant rank near t = 0.** The rank of `K_ZZ(β)` could change along the path in LAML
   families with degenerate weights (w_i → 0). R6 requires constant rank. What replaces the
   certificate at such points? **[open]**
6. **Statistical-resolution tolerance.** The first-order test `c_j > τ_j` is exact in exact
   arithmetic. Whether gamfit should additionally refuse to *distinguish* `λ_j = ∞` from large
   finite `λ_j` when `c_j` is below its sampling SE (the Crainiceanu–Ruppert mass) is a
   reporting question, not an optimization one. **[design question]**
