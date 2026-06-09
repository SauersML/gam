# Manifold geometry and Matérn penalty structure

Two self-contained mathematical problems. Each is posed from first principles for a
reader with no access to any codebase and no ability to execute code. Every object is
defined before it is used. The asks are pure derivations, proofs, and closed forms.

---

## Notation and standing conventions

- `ℝ` is the field of real numbers; `ℝⁿ` is Euclidean `n`-space with the standard inner
  product `⟨a,b⟩ = aᵀb` and norm `‖a‖ = √(aᵀa)`.
- For a real matrix `A`, `Aᵀ` is its transpose, `‖A‖_F = √(tr(AᵀA)) = √(Σ_{ij} A_{ij}²)`
  is its Frobenius norm, and `σ_i(A)` denotes the `i`-th singular value (in non-increasing
  order). `I_k` is the `k×k` identity.
- A matrix `Y ∈ ℝ^{n×p}` is *orthonormal* (a *Stiefel frame*) when `YᵀY = I_p`; its column
  span `span(Y) ⊆ ℝⁿ` is a `p`-dimensional linear subspace, i.e. a point of the
  Grassmann manifold defined below.
- `arccos: [−1,1] → [0,π]`, `arcsin: [−1,1] → [−π/2, π/2]`, `atan2(y,x)` is the
  two-argument arctangent returning the polar angle of `(x,y)` in `(−π, π]`.
- "Machine precision" / "ulp" refer to IEEE-754 binary64: unit roundoff
  `u = 2^{−53} ≈ 1.11·10^{−16}`, and a *1-ulp error in* `s ∈ (0,1]` means an absolute
  perturbation of size up to `½·ulp(s) ≈ u·s` (bounded above by `u` for `s ≤ 1`).

---

## Problem 1 — Grassmann geodesic distance vs. the projector (chordal) metric near the cut locus

### 1.1 The Grassmann manifold

Fix integers `0 < p ≤ n`. The **Grassmann manifold** `Gr(p, n)` is the set of all
`p`-dimensional linear subspaces of `ℝⁿ`. A subspace `𝒮 ∈ Gr(p,n)` is represented
(non-uniquely) by any orthonormal frame `Y ∈ ℝ^{n×p}`, `YᵀY = I_p`, with `span(Y) = 𝒮`.
Two frames `Y, Y'` represent the same subspace iff `Y' = YQ` for some orthogonal
`Q ∈ O(p)`. Thus `Gr(p,n) = St(p,n) / O(p)` where `St(p,n) = {Y : YᵀY = I_p}` is the
compact Stiefel manifold. `Gr(p,n)` is a smooth compact manifold of dimension `p(n−p)`.

`Gr(p,n)` carries a *canonical* Riemannian metric, induced from the standard inner
product on `ℝ^{n×p}` (equivalently, the bi-invariant metric on `O(n)` pushed down through
the quotient `O(n)/(O(p)×O(n−p))`). On the horizontal space at `Y` (tangent vectors
`H ∈ ℝ^{n×p}` with `YᵀH = 0`), the metric is `⟨H₁,H₂⟩ = tr(H₁ᵀH₂)`. This is the metric
relative to which all "geodesic distance" statements below are made.

### 1.2 Principal angles and the two competing distances

Let `𝒮 = span(Y)` and `𝒯 = span(Z)` with `Y, Z ∈ ℝ^{n×p}` orthonormal. The **principal
angles** `0 ≤ θ_1 ≤ θ_2 ≤ … ≤ θ_p ≤ π/2` between `𝒮` and `𝒯` are defined recursively by
`cos θ_k = max{ uᵀv : u ∈ 𝒮, v ∈ 𝒯, ‖u‖=‖v‖=1, u ⟂ u_j, v ⟂ v_j (j<k) }`, with
maximizers `(u_k, v_k)` the principal vectors. Equivalently, if `YᵀZ = U Σ Vᵀ` is a
singular value decomposition with singular values `σ_1 ≥ … ≥ σ_p ∈ [0,1]`, then
`σ_i = cos θ_{p+1−i}`; we write simply `σ_i = cos θ_i` after matching the orderings, so
each principal angle is `θ_i = arccos(σ_i)`. The principal angles depend only on the
subspaces, not on the chosen frames (replacing `Y → YQ₁`, `Z → ZQ₂` multiplies `YᵀZ`
by orthogonal matrices on each side, preserving its singular values).

Two distance notions arise.

**(A) Canonical geodesic (arc-length) distance.** The Riemannian distance induced by the
canonical metric of §1.1 is
```
    d_geo(𝒮, 𝒯) = √( Σ_{i=1}^{p} θ_i² ) = √( Σ_{i=1}^{p} arccos²(σ_i) ).
```
This is the length of the minimizing geodesic; the geodesic itself is `t ↦ span(Y V cos(tΘ) + W sin(tΘ))` for a suitable orthonormal `W` and `Θ = diag(θ_i)`.

**(B) Projector / chordal metric.** Each subspace has a unique orthogonal *projector*
`P_Y = Y Yᵀ ∈ ℝ^{n×n}` (symmetric, idempotent, rank `p`), independent of the frame
representative. A common, frame-free metric is the Frobenius distance between projectors,
```
    d_proj(𝒮, 𝒯) = ‖P_Y − P_Z‖_F = √2 · √( Σ_{i=1}^{p} sin²θ_i ),
```
the second equality being a standard identity (proven via
`‖P_Y − P_Z‖_F² = 2p − 2‖YᵀZ‖_F² = 2 Σ(1 − cos²θ_i) = 2 Σ sin²θ_i`). Variants in use
("chordal", "projection F-norm", `√2·sin`-type, `2·sin(θ/2)` "chord on the sphere of
projectors") all share the property that they are smooth functions of `{sinθ_i, cosθ_i}`
that **saturate** as `θ_i → π/2`, in contrast to `d_geo` which is *linear* in `θ_i`.

### 1.3 The observed discrepancy (the phenomenon to explain)

On real data (subspaces obtained as leading principal-component spans of grouped
datasets), the canonical arc-length `d_geo = √(Σ arccos²σ_i)` is computed to ≈`10^{−14}`
agreement with the analytic principal-angle formula, including pairs whose **largest
principal angle approaches `π/2`** (the *cut locus* direction). A reference library that
instead reports a **projector-based** distance `metric.dist` disagrees from `d_geo` by up
to ≈ `0.64` on exactly those near-`π/2` pairs. The disagreement is not roundoff: it is
the structural gap between (A) and (B).

### 1.4 Asks

**(1a) Closed-form gap and identification of the true geodesic distance.** Define the
per-angle discrepancy and prove the exact relationship between the canonical geodesic
distance and the chordal/projector family. Specifically:

  - Show that for a single principal angle `θ ∈ [0, π/2]`, the canonical contribution is
    `θ` while the chordal contribution is `√2·sinθ` (per the `‖P_Y−P_Z‖_F` normalization),
    and characterize the pointwise gap `g(θ) = θ − sinθ` (and its `√2`-scaled / `2sin(θ/2)`
    counterparts). Prove `g` is non-negative, strictly increasing on `(0, π/2)`, with
    `g(θ) = θ³/6 + O(θ⁵)` as `θ→0` and maximal endpoint gap at `θ = π/2`
    (`θ = π/2 ≈ 1.5708` vs. `sin θ = 1`). Aggregate over the `p` angles to give the exact
    multivariate gap `d_geo − d_proj` as a function of `(θ_1,…,θ_p)`, and show it is
    consistent in magnitude with an observed ≈`0.64` when one angle is near `π/2`.
  - Prove that `d_geo = √(Σ θ_i²)` is the genuine Riemannian (geodesic, length-minimizing)
    distance for the canonical metric of §1.1 — e.g. by exhibiting the minimizing geodesic
    and showing no shorter admissible path exists — and that every member of the
    projector/chordal family is an *extrinsic chord length*, hence a lower bound
    `d_proj ≤ d_geo` that is a metric on `Gr(p,n)` but **not** the geodesic distance. State
    precisely why the two necessarily diverge as `θ_max → π/2` (arc vs. chord; the chord
    saturates at the diameter of the projector sphere while the arc keeps growing).

**(1b) Cut-locus structure.** The cut locus of `𝒮` consists of subspaces `𝒯` reaching at
least one principal angle `θ_i = π/2` (`σ_i = 0`). Characterize precisely:

  - Why the minimizing geodesic from `𝒮` to such `𝒯` is **non-unique** and the Riemannian
    logarithm (the initial tangent `H` with `exp_𝒮(H) = 𝒯`, `‖H‖ = d_geo`) is
    **multivalued** there. Give the dimension/parametrization of the set of minimizers
    (the choice of principal-vector pairing in the degenerate `σ_i = 0` block, i.e. the
    `O(m)` freedom when `m` angles equal `π/2`).
  - State the correct distance at the cut locus (it remains `√(Σ θ_i²)` with the offending
    `θ_i = π/2`, single-valued even though the log is not) and give a well-defined
    *selection* of one valid logarithm there, proving its norm equals `d_geo`.

**(1c) Optimal conditioning of recovering `θ` from `σ`.** Recovering `θ_i = arccos(σ_i)`
is ill-conditioned near `θ = 0` (`σ = 1`): `d/dσ arccos(σ) = −1/√(1−σ²) → −∞` as `σ → 1`.

  - Derive the exact first-order error amplification: for a perturbation `δσ` of the
    singular value, `δθ ≈ −δσ/√(1−σ²) = −δσ/ sinθ`. Conclude that a 1-ulp error in `σ`
    near `σ = 1` (so `δσ ≲ u`) inflates to `δθ ≈ u / sinθ`, and using `sinθ ≈ √(2(1−σ))`
    show the bound degrades to `δθ ≈ √(2u)` at the worst small-angle scale
    (`≈ √(2·1.11·10^{−16}) ≈ 1.5·10^{−8}`), losing ~half the significant digits — exactly
    the ≈`5.97·10^{−9}`-vs-`10^{−9}` failure scale observed when angles are extracted by
    `arccos(σ)` at the smallest angles.
  - Give and **prove** an optimally-conditioned formula for `θ_i` that attains machine
    precision *uniformly* over `θ ∈ [0, π/2]`. Two routes to analyze:
    (i) the **atan2 of singular values of an off-diagonal block** — split an orthonormal
    completion so that for the pair of frames the principal angles satisfy
    `θ_i = atan2(σ_i^{⟂}, σ_i^{∥})` where `σ^{∥}` are the singular values of `YᵀZ`
    (`=cosθ`) and `σ^{⟂}` are the singular values of `(I − YYᵀ)Z` (`=sinθ`); equivalently
    `θ_i = atan( σ_i(M) )` where `M = (I−YYᵀ)Z·(YᵀZ)⁻¹` has singular values `tanθ_i`
    (so the eigenvalues of `MᵀM` are `tan²θ_i` and `θ_i = arctan(√eval_i)`).
    (ii) the **half-chord arcsine** `θ_i = 2·arcsin( ½‖principal-vector chord‖ )`.
    Prove route (i): show `θ ↦ atan(tanθ)` and `θ ↦ arctan(√(tan²θ))` are well-conditioned
    for all `θ ∈ [0, π/2)` (derivative of `arctan(t)` is `1/(1+t²)`, bounded by 1; the
    composite condition number is `O(1)` away from `π/2`) and that the sine branch handles
    the `θ→π/2` end, so that combining a `cos`-stable branch near `0` and a `sin`-stable
    branch near `π/2` (or the single `atan2(sin,cos)` form) yields uniform relative
    accuracy `O(u)` in `θ` across the whole range, in contrast to `arccos(σ)`. Identify the
    crossover and quantify the worst-case error of each branch.

---

## Problem 2 — Matérn smoothness-dependent penalty enumeration and the κ-derivative index invariant

### 2.1 The Matérn RKHS and its Sobolev order

Fix a spatial dimension `d ≥ 1` and a smoothness parameter `ν > 0`. The **Matérn
covariance kernel** on `ℝ^d` with smoothness `ν`, marginal variance 1, and inverse
length-scale `κ = e^ψ > 0` (we work in the log-parameter `ψ = log κ`) is
```
    k_ν(r) = (2^{1−ν}/Γ(ν)) (κ r)^ν K_ν(κ r),   r = ‖x − x'‖,
```
where `K_ν` is the modified Bessel function of the second kind. Its reproducing-kernel
Hilbert space (RKHS) `H_ν` is norm-equivalent to the Sobolev space `H^m(ℝ^d)` with
**Sobolev order**
```
    m = ν + d/2.
```
This is the classical spectral fact: the Matérn spectral density decays as
`(κ² + ‖ω‖²)^{−(ν + d/2)}`, so the RKHS squared norm `‖f‖²_{H_ν} ≍ ∫ |f̂(ω)|² (κ² + ‖ω‖²)^m dω`
controls exactly the derivatives of `f` up to order `m`: the order-`j` derivative seminorm
`‖D^j f‖_{L²}² = ∫ ‖ω‖^{2j} |f̂(ω)|² dω` is finite (controlled by the RKHS norm) precisely
when `j ≤ m`, and a derivative-`j` penalty with `j > m` imposes roughness control the
kernel's own RKHS does **not** possess.

### 2.2 The operator-penalty overlay and its smoothness gate

A reduced-rank Matérn smooth is fit with an overlay of differential-operator penalties
built from collocation matrices `D_0, D_1, D_2` (discretizations of the value `D^0=I`,
gradient `D^1=∇`, and Hessian `D^2=∇²` operators on the basis), giving squared-`L²`-seminorm
penalties
```
    S_j = c_j^{−1} · D_jᵀ D_j     (j = 0: mass; j = 1: tension; j = 2: stiffness),
```
each normalized by its own Frobenius scale `c_j = ‖D_jᵀD_j‖_F`. The **smoothness gate**
admits operator `j` into the penalty set iff its seminorm is finite on the Matérn RKHS,
i.e. iff `j < m` *strictly* (with a small tolerance so an exact half-integer boundary
disables the matching-order operator):
```
    admissible(j) ⇔ m > j   ⇔ ν + d/2 > j.
```
Concretely: mass (`j=0`) is always admitted; tension (`j=1`) requires `m > 1`; stiffness
(`j=2`) requires `m > 2`. For `ν ≥ 3/2`, or for any `d ≥ 2`, all three are admitted; only
the genuinely rough `ν = 1/2, d = 1` case (`m = 1`, the Ornstein–Uhlenbeck / exponential
kernel, an `H¹` process with continuous but non-differentiable paths) drops tension and
stiffness, leaving the single mass penalty. Admitting `j = 1` there would bias the
reduced-rank fit toward `C¹` functions the kernel does not favour, over-smoothing relative
to the exact GP and collapsing held-out oscillation.

The **forward** penalty builder emits the gated list `{ S_j : j < m }` — its length is
`G(ν,d) = #{ j ∈ {0,1,2} : ν + d/2 > j }` (so `G = 1` for `ν=1/2,d=1`, `G = 3` for the
admit-all cases). For smoothness optimization, gradients are taken with respect to
`ψ = log κ`; a separate builder produces the **ψ-derivatives** `∂S_j/∂ψ` (and second
derivatives `∂²S_j/∂ψ²`) of those same penalties via the exact quotient/chain rule on the
normalized Grams.

### 2.3 The defect (the structure to repair, stated as a mathematical inconsistency)

The forward builder enumerates the **gated** set `{ S_j : j < m }` of length `G(ν,d)`. The
ψ-derivative builder instead enumerates the **ungated** triple `[∂S_0/∂ψ, ∂S_1/∂ψ,
∂S_2/∂ψ]` of fixed length 3, regardless of `(ν,d)`. When `G(ν,d) < 3` — i.e. for a rough
Matérn (`m ≤ 2`, in particular `ν=1/2,d=1` with `m=1` and a single, non-double penalty) —
the two lists are **index-misaligned**: the optimizer pairs each penalty `S_{a}` (indexed
in the gated forward list) with smoothing weight `λ_a` but reads the derivative entry at
the same position `a` from the ungated list, which corresponds to a *different* operator
`j`. The assembled `ψ`-gradient of the penalized criterion is then taken against a
mismatched penalty, so it is not the derivative of the objective actually being optimized.

### 2.4 Asks

**(2a) The correct `(ν,d) → admissible-penalty-set` rule from Matérn RKHS theory.** State
and prove, from the spectral characterization of the Matérn RKHS as `H^{ν+d/2}(ℝ^d)`,
exactly which derivative-operator seminorms `‖D^j f‖_{L²}` are finite (a.s. / RKHS-bounded)
for given `(ν, d)`. Precisely:

  - Prove that `‖D^j f‖_{L²}² = ∫ ‖ω‖^{2j} |f̂(ω)|² dω` is bounded by a constant times
    `‖f‖²_{H_ν}` iff `‖ω‖^{2j} / (κ² + ‖ω‖²)^{ν+d/2}` is bounded as `‖ω‖→∞`, i.e. iff
    `2j ≤ 2(ν + d/2)`, i.e. `j ≤ ν + d/2 = m`. Distinguish the boundary case `j = m`
    (the seminorm is exactly the borderline-divergent / non-controlled case; explain why
    the strict gate `j < m` is the correct admissibility rule for a *finite, kernel-honest*
    penalty and not merely a numerical convenience), and confirm the resulting count
    `G(ν,d) = #{ j ∈ {0,1,2} : j < ν + d/2 }`. Discuss the `ν=1/2, d=1` exponential-kernel
    case as the canonical `m=1` boundary where only `j=0` survives.

**(2b) The index-alignment invariant.** Prove the consistency requirement that the
`κ`-derivative penalty list must enumerate the **same gated set** as the forward penalty
list, in the **same order**. Formalize as follows. Let the penalized objective be
`F(β, ψ) = ℓ(β) + ½ Σ_{a=1}^{G} λ_a · βᵀ S_{(a)}(ψ) β`, where `(a) ↦ j` is the gated
indexing map (the `a`-th admitted operator order) and each `S_{(a)}(ψ)` depends on `ψ`
through `κ = e^ψ`. The exact `ψ`-gradient of the penalty block is
`∂F/∂ψ = ½ Σ_{a=1}^{G} λ_a · βᵀ (∂S_{(a)}/∂ψ) β`. Prove:

  - (Correctness ⇔ alignment) The assembled gradient equals the true `∂F/∂ψ` **iff** the
    derivative list supplied to position `a` is `∂S_{(a)}/∂ψ` for the *same* gated index
    map `(a)` used by the forward list — i.e. iff the derivative builder applies the
    identical gate `j < m`. If the derivative list is the ungated triple while the forward
    list is gated, positions `a ≥ 2` reference `∂S_{j'}/∂ψ` for `j' ≠ (a)`, and the
    resulting vector is the gradient of a *different* functional (a wrong-operator penalty),
    not of `F`; quantify the error as `½ Σ_a λ_a βᵀ (∂S_{j'(a)}/∂ψ − ∂S_{(a)}/∂ψ) β`, which
    is generically nonzero whenever any admitted operator differs in the two enumerations.
  - (Why it is latent under double-penalty configs) Note that if the configuration always
    admits all three operators (e.g. the `m > 2` regime, or a "double-penalty" mode that
    forces the full triple), then `G = 3`, the gate is vacuous, both lists coincide, and
    the desync cannot manifest — establishing that the inconsistency is exposed *only* in a
    rough-`ν`, non-double-penalty regime (`G < 3`). State the general invariant: *any pair
    of (penalty list, penalty-derivative list) consumed positionally by a gradient
    assembler must be generated by one and the same admissibility predicate and ordering*,
    and prove this is necessary and sufficient for the assembled `ψ`-gradient to equal the
    analytic derivative of the penalized objective for every `(ν, d)`.
