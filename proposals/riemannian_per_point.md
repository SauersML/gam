# Retraction-Based Riemannian Optimization for Per-Point Latents

Status: draft derivation.
Audience: implementers wiring `LatentCoordValues`,
`input_loc_derivatives`, the arrow-Schur latent inner solver, and IFT
warm-starts.

## 1. Scope

Per-row latent coordinates should be optimized on their natural manifold:

```text
t_i ∈ M
ξ_i ∈ T_{t_i}M
t_i,new = R_{t_i}(ξ_i).
```

The row-local arrow-Schur algebra stays intact. The only change is the
geometry of each row latent block:

```text
Euclidean row gradient      → tangent gradient
Euclidean row Hessian action → tangent Hessian action
Euclidean IFT delta          → tangent vector applied by retraction
```

Notation:

```text
M              one-row manifold
p ∈ M          current point
m              ambient embedding dimension
d              intrinsic dimension
ξ,η ∈ T_pM     tangent vectors
q = R_p(ξ)     retracted point
P_p            tangent projection
g_p            Riemannian inner product
∇_E            ambient Euclidean gradient
∇²_E · ξ       ambient Euclidean Hessian-vector product
grad_R         Riemannian gradient
Hess_R[ξ]      Riemannian Hessian action
W_p(ξ,v)       Weingarten/shape correction
```

The committed embedded-submanifold conversion is:

```text
grad_R = P_p(∇_E)
Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ, ∇_E)).
```

Equivalently, `W_p` may receive `P_p^⊥(∇_E)` because only the normal
component contributes for the sphere family.

Important convention:

```text
Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ, P_p(∇_E)))
```

would make the sphere correction vanish, since `P_p(∇_E)` is tangent.
The final Rust signature must pass the ambient gradient or the normal
gradient. For a unit sphere:

```text
W_p(ξ,v) = <p,v> ξ.
```

Using `v = ∇_E` gives the required correction
`-<p,∇_E>ξ`.

## 2. Source Alignment

Relevant source already exists:

```text
src/terms/input_loc_derivatives.rs
  basis_input_loc_grad
  basis_input_loc_hess
  tensor_product_input_loc_grad
  periodic_radial_input_loc_grad_1d
  sphere_s2_input_loc_grad
  contract_input_loc_gradient

src/terms/latent_coord.rs
  LatentManifold
  LatentCoordValues::retract_flat_delta
  LatentCoordValues::design_gradient_wrt_t_dispatch
  LatentCoordValues::contract_gradient
  LatentManifold::project_to_tangent
  LatentManifold::euclidean_to_riemannian_hessian
  LatentManifold::riemannian_hessian_matrix

src/solver/riemannian.rs
  Manifold
  Euclidean, Circle, Sphere, Interval, Torus, Product
  ManifoldKind
  RiemannianNewtonStep
  riemannian_newton_step_on_point
  retract_euclidean_delta

src/solver/arrow_schur.rs
  ArrowSystem::apply_riemannian_latent_geometry
  apply_per_row_retraction

src/solver/persistent_warm_start.rs
  ift_warm_start_latent
  apply_ift_retraction
```

There is currently a representation split:

```text
LatentManifold::Circle in src/terms/latent_coord.rs
  scalar angle with wrapping

Circle in src/solver/riemannian.rs
  embedded unit vector in ℝ²
```

The optimizer contract should be embedded. Chart coordinates can remain API
sugar, but Newton, Hessian conversion, vector transport, and IFT retraction
should operate on embedded points.

## 3. Row Chain Rule

For row `i`:

```text
Φ_i(t_i) = [Φ_{i,1}(t_i), ..., Φ_{i,K}(t_i)].
```

`input_loc_derivatives.rs` provides:

```text
J_i[k,a]     = ∂Φ_{i,k}/∂t_{i,a}
HΦ_i[k,a,b] = ∂²Φ_{i,k}/∂t_{i,a}∂t_{i,b}.
```

For radial kernels:

```text
r_{ik} = ||t_i - c_k||
Φ_{i,k} = φ(r_{ik})
q(r) = φ'(r)/r
s(r) = (φ''(r) - q(r))/r²
J_i[k,a] = q(r_{ik})(t_i - c_k)_a
HΦ_i[k,a,b] = q(r_{ik})δ_ab
            + s(r_{ik})(t_i - c_k)_a(t_i - c_k)_b.
```

These are exactly `basis_input_loc_grad` and `basis_input_loc_hess`;
`RadialScalarKind::eval_design_triplet` owns collision limits and
collision errors.

Given upstream derivatives:

```text
u_i[k]   = ∂F/∂Φ_{i,k}
A_i[k,l] = ∂²F/∂Φ_{i,k}∂Φ_{i,l},
```

the Euclidean latent row gradient and Hessian are:

```text
g_E[a] = Σ_k u_i[k] J_i[k,a]
H_E[a,b] = Σ_{k,l} J_i[k,a] A_i[k,l] J_i[l,b]
         + Σ_k u_i[k] HΦ_i[k,a,b].
```

For Gaussian residual least squares, with `ŷ_i = Φ_iβ`:

```text
u_i[k] = -resid_i β_k
A_i[k,l] = β_kβ_l
g_E = -resid_i J_iᵀβ
H_E = (J_iᵀβ)(J_iᵀβ)ᵀ
    - resid_i Σ_k β_k HΦ_i[k,:,:].
```

The Riemannian layer then applies:

```text
g_R = P_p(g_E)
H_R[ξ] = P_p(H_Eξ - W_p(ξ,g_E)).
```

## 4. Manifold S¹

### 4.1 Embedding in ℝ^m, Tangent Space T_p M

Use the embedded unit circle:

```text
S¹ = {p ∈ ℝ² : ||p|| = 1}
p = (cos θ, sin θ)
m = 2
d = 1
T_pS¹ = {ξ ∈ ℝ² : <p,ξ> = 0}.
```

Canonical tangent basis:

```text
Q(p) = [-p_y, p_x] ∈ ℝ^{2×1}
ξ = Q(p)α.
```

### 4.2 Retraction R_p(ξ) Closed-Form

Use normalization:

```text
R_p(ξ) = (p + ξ) / ||p + ξ||.
```

The exact exponential map is:

```text
Exp_p(ξ) = cos(||ξ||)p + sin(||ξ||)ξ/||ξ||.
```

Normalization is the committed retraction because it is closed-form,
first-order correct, and seam-free.

### 4.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v) = v - <p,v>p
       = Q(p)Q(p)ᵀv.
```

### 4.4 Vector Transport τ_{p→q}(ξ)

Committed default:

```text
τ_{p→q}(ξ) = P_q(ξ).
```

Exact non-antipodal transport, optional later:

```text
τ_exact(ξ) = ξ - (<q,ξ> / (1 + <p,q>))(p + q).
```

### 4.5 Inner Product g_p

Default:

```text
g_p(ξ,η) = <ξ,η>_ℝ².
```

Scale-normalized product metric:

```text
g_p(ξ,η) = <ξ,η> / (2π)².
```

### 4.6 Euclidean→Riemannian Gradient

```text
grad_R = P_p(∇_E)
       = ∇_E - <p,∇_E>p.
```

### 4.7 Euclidean→Riemannian Hessian and W

`S¹` is `S^1`, so:

```text
W_p(ξ,v) = <p,v>ξ.
Hess_R[ξ] = P_p(∇²_E · ξ - <p,∇_E>ξ).
```

The Weingarten argument must be the ambient gradient or normal gradient.

## 5. Manifold S²

### 5.1 Embedding in ℝ^m, Tangent Space T_p M

```text
S² = {p ∈ ℝ³ : ||p|| = 1}
m = 3
d = 2
T_pS² = {ξ ∈ ℝ³ : <p,ξ> = 0}.
```

Build `Q(p) ∈ ℝ^{3×2}` by choosing an anchor not parallel to `p`:

```text
a = e_z if |p_z| < 0.9 else e_x
q1 = normalize(a - <a,p>p)
q2 = p × q1
Q = [q1 q2].
```

### 5.2 Retraction R_p(ξ) Closed-Form

```text
R_p(ξ) = (p + ξ) / ||p + ξ||.
```

Exact exponential map:

```text
Exp_p(ξ) = cos(||ξ||)p + sin(||ξ||)ξ/||ξ||.
```

### 5.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v) = v - <p,v>p
       = (I - ppᵀ)v.
```

### 5.4 Vector Transport τ_{p→q}(ξ)

Default:

```text
τ_{p→q}(ξ) = P_q(ξ).
```

Exact non-antipodal transport:

```text
τ_exact(ξ) = ξ - (<q,ξ> / (1 + <p,q>))(p + q).
```

### 5.5 Inner Product g_p

```text
g_p(ξ,η) = <ξ,η>_ℝ³.
```

Optional trust-region scaling:

```text
g_p(ξ,η) = <ξ,η> / π².
```

### 5.6 Euclidean→Riemannian Gradient

```text
grad_R = P_p(∇_E)
       = ∇_E - <p,∇_E>p.
```

### 5.7 Euclidean→Riemannian Hessian and W

For the unit sphere:

```text
W_p(ξ,v) = <p,v>ξ.
Hess_R[ξ] = P_p(∇²_E · ξ - <p,∇_E>ξ).
```

This is the derivative of the projected gradient, projected back to
`T_pS²`.

## 6. Manifold S^n

### 6.1 Embedding in ℝ^m, Tangent Space T_p M

```text
S^n = {p ∈ ℝ^{n+1} : ||p|| = 1}
m = n + 1
d = n
T_pS^n = {ξ ∈ ℝ^{n+1} : <p,ξ> = 0}.
```

`Q(p) ∈ ℝ^{(n+1)×n}` can be built by projecting ambient basis vectors
with `P_p` and applying modified Gram-Schmidt. This matches the generic
`Manifold::tangent_basis` approach already present in
`src/solver/riemannian.rs`.

### 6.2 Retraction R_p(ξ) Closed-Form

```text
R_p(ξ) = (p + ξ) / ||p + ξ||.
```

Before retraction:

```text
ξ ← P_p(ξ)
clip ||ξ||_g if a trust radius is active.
```

A zero or non-finite `||p + ξ||` should be an error, not a hidden
projection to an arbitrary axis.

### 6.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v) = v - <p,v>p.
```

### 6.4 Vector Transport τ_{p→q}(ξ)

Default:

```text
τ_{p→q}(ξ) = P_q(ξ).
```

Exact non-antipodal transport:

```text
τ_exact(ξ) = ξ - (<q,ξ> / (1 + <p,q>))(p + q).
```

Avoid exact transport near `q ≈ -p`.

### 6.5 Inner Product g_p

```text
g_p(ξ,η) = <ξ,η>_ℝ^{n+1}.
```

Optional scaling:

```text
g_p(ξ,η) = <ξ,η> / π².
```

### 6.6 Euclidean→Riemannian Gradient

```text
grad_R = P_p(∇_E).
```

### 6.7 Euclidean→Riemannian Hessian and W

```text
W_p(ξ,v) = <p,v>ξ.
Hess_R[ξ] = P_p(∇²_E · ξ - <p,∇_E>ξ).
```

The correction only needs the scalar `<p,∇_E>`.

## 7. Manifold Interval[a,b]

### 7.1 Embedding in ℝ^m, Tangent Space T_p M

```text
M = [a,b] ⊂ ℝ
m = 1
d = 1
```

Interior:

```text
T_pM = ℝ,  a < p < b.
```

Boundary feasible tangent cones:

```text
C_aM = {ξ ∈ ℝ : ξ ≥ 0}
C_bM = {ξ ∈ ℝ : ξ ≤ 0}.
```

The final implementation should treat the interval as a closed constraint
with active boundary logic, not as a smooth boundaryless manifold.

### 7.2 Retraction R_p(ξ) Closed-Form

Closed interval retraction:

```text
R_p(ξ) = clamp(p + ξ, a, b).
```

Interior smooth chart, available only if explicitly requested:

```text
c = (a + b)/2
r = (b - a)/2
p = c + r tanh(z)
R_p(ξ) = c + r tanh(z + ξ / (r(1 - tanh(z)^2))).
```

The chart is ill-conditioned near the boundary, so the default should be
the closed-constraint retraction.

### 7.3 Projection P_p: ℝ^m → T_p M

Interior:

```text
P_p(v) = v.
```

Boundary cone projection for update vectors:

```text
P_a(v) = max(v,0)
P_b(v) = min(v,0).
```

For minimization, KKT stationarity is:

```text
p = a: ∇_E ≥ 0
p = b: ∇_E ≤ 0.
```

### 7.4 Vector Transport τ_{p→q}(ξ)

Interior:

```text
τ_{p→q}(ξ) = ξ.
```

Cone-aware default:

```text
τ_{p→q}(ξ) = P_q(ξ).
```

### 7.5 Inner Product g_p

```text
g_p(ξ,η) = ξη.
```

Scale-normalized:

```text
g_p(ξ,η) = ξη / (b-a)².
```

### 7.6 Euclidean→Riemannian Gradient

Interior:

```text
grad_R = ∇_E.
```

Boundary:

```text
grad_R = projected KKT gradient under the feasible tangent cone.
```

### 7.7 Euclidean→Riemannian Hessian and W

The interval is flat:

```text
W_p(ξ,v) = 0.
```

Interior:

```text
Hess_R[ξ] = ∇²_E · ξ.
```

Boundary:

```text
Hess_R[ξ] = P_p(∇²_E · ξ)
```

with `ξ` restricted to the feasible tangent cone.

## 8. Manifold ℝ

### 8.1 Embedding in ℝ^m, Tangent Space T_p M

```text
M = ℝ
m = 1
d = 1
T_pℝ = ℝ.
```

For `ℝ^d`, use a dedicated Euclidean block with `m = d` or a product of
scalar real components.

### 8.2 Retraction R_p(ξ) Closed-Form

```text
R_p(ξ) = p + ξ.
```

### 8.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v) = v.
```

### 8.4 Vector Transport τ_{p→q}(ξ)

```text
τ_{p→q}(ξ) = ξ.
```

### 8.5 Inner Product g_p

```text
g_p(ξ,η) = ξη.
```

### 8.6 Euclidean→Riemannian Gradient

```text
grad_R = ∇_E.
```

### 8.7 Euclidean→Riemannian Hessian and W

```text
W_p(ξ,v) = 0.
Hess_R[ξ] = ∇²_E · ξ.
```

## 9. Manifold Torus

### 9.1 Embedding in ℝ^m, Tangent Space T_p M

Use a product of embedded circles:

```text
T^d = (S¹)^d
p = (p_1,...,p_d)
p_j ∈ ℝ²
||p_j|| = 1
m = 2d
intrinsic dimension = d
T_pT^d = {ξ ∈ ℝ^{2d} : <p_j,ξ_j> = 0 for every j}.
```

### 9.2 Retraction R_p(ξ) Closed-Form

Blockwise:

```text
R_p(ξ)_j = (p_j + ξ_j) / ||p_j + ξ_j||.
```

### 9.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v)_j = v_j - <p_j,v_j>p_j.
```

### 9.4 Vector Transport τ_{p→q}(ξ)

```text
τ_{p→q}(ξ)_j = P_{q_j}(ξ_j).
```

### 9.5 Inner Product g_p

Default:

```text
g_p(ξ,η) = Σ_j <ξ_j,η_j>.
```

Scale-normalized:

```text
g_p(ξ,η) = Σ_j <ξ_j,η_j> / (2π)².
```

### 9.6 Euclidean→Riemannian Gradient

```text
grad_R,j = P_{p_j}(∇_{E,j}).
```

### 9.7 Euclidean→Riemannian Hessian and W

Blockwise curvature:

```text
W_p(ξ,v)_j = <p_j,v_j>ξ_j.
```

Full Hessian action:

```text
Hess_R[ξ]_j = P_{p_j}((∇²_E · ξ)_j - <p_j,∇_{E,j}>ξ_j).
```

Do not drop cross-component entries in `∇²_E · ξ`; only the curvature
correction and projection are blockwise.

## 10. Manifold Product

### 10.1 Embedding in ℝ^m, Tangent Space T_p M

For components:

```text
M = M_1 × ... × M_r
p = (p_1,...,p_r)
p_j ∈ M_j ⊂ ℝ^{m_j}
m = Σ_j m_j
d = Σ_j d_j
T_pM = T_{p_1}M_1 × ... × T_{p_r}M_r.
```

### 10.2 Retraction R_p(ξ) Closed-Form

```text
R_p(ξ) = (R_{p_1}^{M_1}(ξ_1),...,R_{p_r}^{M_r}(ξ_r)).
```

### 10.3 Projection P_p: ℝ^m → T_p M

```text
P_p(v) = (P_{p_1}^{M_1}(v_1),...,P_{p_r}^{M_r}(v_r)).
```

### 10.4 Vector Transport τ_{p→q}(ξ)

```text
τ_{p→q}(ξ)
  = (τ_{p_1→q_1}^{M_1}(ξ_1),...,τ_{p_r→q_r}^{M_r}(ξ_r)).
```

### 10.5 Inner Product g_p

Default:

```text
g_p(ξ,η) = Σ_j g_{p_j}^{M_j}(ξ_j,η_j).
```

Weighted product:

```text
g_p(ξ,η) = Σ_a w_a ξ_a η_a
```

with positive per-ambient-axis weights.

### 10.6 Euclidean→Riemannian Gradient

```text
grad_R,j = P_{p_j}^{M_j}(∇_{E,j}).
```

### 10.7 Euclidean→Riemannian Hessian and W

Blockwise Weingarten:

```text
W_p(ξ,v)_j = W_{p_j}^{M_j}(ξ_j,v_j).
```

Full product Hessian:

```text
Hess_R[ξ] = P_p(∇²_E · ξ - W_p(ξ,∇_E)).
```

The product tangent projection and curvature are blockwise. The objective
Hessian need not be blockwise.

## 11. Product Manifold Composition

A product manifold should only own slicing and composition. Each component
owns its own geometry:

```text
ambient_dim
intrinsic_dim
project_point
project_tangent
retract
vector_transport
inner_product
tangent_basis
weingarten
```

The product validates:

```text
Σ component.ambient_dim() == row ambient dimension
Σ component.dim() == row intrinsic dimension
weights.len() == ambient dimension, if weights are supplied.
```

The product tangent basis should be:

```text
Q = blockdiag(Q_1,...,Q_r).
```

This preserves sparsity, avoids mixing incompatible component units, and
lets interval active sets reduce local tangent dimension.

Composition rules:

```text
project_point       blockwise
project_tangent     blockwise
retract             blockwise
vector_transport    blockwise
weingarten          blockwise
inner_product       sum or weighted ambient sum
tangent_basis       block diagonal
```

## 12. Chain Rule from Chart to Riemannian

`input_loc_derivatives.rs` computes derivatives with respect to the input
coordinates the basis evaluator uses. The Riemannian optimizer consumes
those derivatives after chart-to-embedding conversion.

### 12.1 Radial Embedded Basis

If `t_i` is already the embedded point:

```text
g_E = contract upstream gradient through J_i
H_E = contract upstream Hessian through J_i and HΦ_i
g_R = P_{t_i}(g_E)
H_R[ξ] = P_{t_i}(H_Eξ - W_{t_i}(ξ,g_E)).
```

For chord-distance sphere kernels:

```text
∇_E Φ_k = q(r)(t - c)
H_E Φ_k = qI + s(t-c)(t-c)ᵀ
Hess_R Φ_k[ξ] = P_t(H_EΦ_k ξ - <t,∇_EΦ_k>ξ).
```

`sphere_s2_input_loc_grad` already returns the projected first derivative.
For Hessians, retain the unprojected ambient gradient long enough to compute
`<t,∇_EΦ_k>`.

### 12.2 S¹ Chart Basis

For periodic 1-D bases, source derivatives are usually chart derivatives:

```text
θ ∈ ℝ / 2πℤ
p(θ) = (cos θ, sin θ)
Q(p) = dp/dθ = (-sin θ, cos θ).
```

Chart-to-tangent conversion:

```text
grad_R = Q(p) (dF/dθ)
Qᵀ Hess_R[Qα] = (d²F/dθ²)α.
```

Prefer solving in intrinsic angle basis and lifting `ξ = Qη`, rather than
inventing ambient normal Hessian entries.

### 12.3 Tensor Product Chart Basis

For tensor products:

```text
∂Φ/∂u_a
```

comes from `tensor_product_input_loc_grad`. Convert each chart axis to its
component tangent basis:

```text
S¹ factor:       grad_emb += Q(p_a) ∂F/∂θ_a
Interval factor: grad_emb += ∂F/∂u_a
ℝ factor:        grad_emb += ∂F/∂u_a.
```

For Hessians, use tangent coordinates:

```text
H_Q = J_chart→tanᵀ H_chart J_chart→tan
    + second-chart terms when the chart itself is nonlinear.
```

For embedded circle factors, the intrinsic chart is linear in the tangent
basis at the point, so the row Newton solve should use `η` coordinates and
lift only the final step.

### 12.4 Contracted Gradient and Hessian

Existing gradient contraction:

```text
∂L/∂t_{i,a} = Σ_k (∂L/∂Φ_{i,k})(∂Φ_{i,k}/∂t_{i,a}).
```

Required Hessian contraction:

```text
H_E[a,b] = Σ_{k,l} J[k,a] A[k,l] J[l,b]
         + Σ_k u[k] HΦ[k,a,b].
```

Then convert rowwise:

```text
g_R = manifold.project_tangent(p,g_E)
H_R = manifold.riemannian_hessian_matrix(p,g_E,H_E)
```

## 13. Riemannian Newton on Tangent Space

For each row:

```text
p = t_i
g_E ∈ ℝ^m
H_E ∈ ℝ^{m×m}
grad_R = P_p(g_E)
H_R[ξ] = P_p(H_Eξ - W_p(ξ,g_E)).
```

Build:

```text
Q ∈ ℝ^{m×d}
QᵀQ = I
range(Q) = T_pM.
```

Solve:

```text
(Qᵀ H_R Q + μI) η = -Qᵀ grad_R
ξ = Qη
t_new = R_p(ξ).
```

This solve must be intrinsic. Do not solve the structurally singular
ambient system by pinning normal directions and letting normal regularization
alter the tangent step.

Trust-region clipping:

```text
if ||ξ||_g > Δ:
    ξ ← (Δ / ||ξ||_g)ξ.
```

For product manifolds:

```text
Q = blockdiag(Q_1,...,Q_r).
```

For interval boundaries, the active cone may reduce the row tangent
dimension.

## 14. IFT Warm-Start on Manifold

The existing predictor returns:

```text
Δt_i = -H_{tt,i}^{-1}rhs_i.
```

For manifold latents, interpret this as:

```text
Δt_i ∈ T_{t_i}M.
```

Apply it by:

```text
ξ_i = P_{t_i}(Δt_i)
t_i,new = R_{t_i}(ξ_i).
```

If the cache stores intrinsic tangent factors, preferred flow is:

```text
store p_i and Q_i at factorization time
solve for η_i
lift ξ_i = Q_iη_i
reproject ξ_i at current p_i if needed
apply R_{p_i}(ξ_i).
```

If a previous tangent delta is reused after moving the point, transport it:

```text
Δt_i transported = τ_{old→new}(Δt_i)
                 = P_{new}(Δt_i) by default.
```

## 15. Numerical Pitfalls

Sphere charts:

```text
Embedded spheres do not have pole singularities; latitude/longitude charts do.
Keep optimizer state embedded.
Use charts only for API display or chart-native basis evaluation.
```

South pole / chart transition:

```text
If a chart is required, switch charts near poles.
Do not finite-difference longitude at a pole.
Do not measure longitude residuals as Euclidean errors near a pole.
```

Antipodal sphere transport:

```text
τ_exact has denominator 1 + <p,q>.
Use projection transport near q ≈ -p.
Keep sphere trust steps well below π unless line search accepts larger moves.
```

Retraction normalization:

```text
Project ξ to tangent before normalization.
Clip trust radius.
Treat non-finite ||p+ξ|| as an error.
```

Interval boundary:

```text
clamp is nonsmooth at a,b.
Use KKT active-set logic at boundaries.
Avoid tanh chart near boundary unless its Jacobian is monitored.
```

Vector transport:

```text
Projection transport is cheap and stable but not exactly isometric.
Use it for IFT warm starts.
Check norm loss before using it for quasi-Newton curvature pairs.
```

Radial kernel collisions:

```text
Do not map degenerate collisions to zero.
Respect BasisError::DegenerateAtCollision.
Move centers or choose a smoother kernel.
```

Product units:

```text
Use metric weights for mixed S¹/Interval/ℝ products.
Report weights in diagnostics.
Clip trust radius in the weighted metric norm.
```

## 16. Final Rust Trait

The final manifold trait should use embedded coordinates and return errors
instead of silently repairing invalid optimizer states.

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

pub trait Manifold: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn ambient_dim(&self) -> usize;

    fn project_point(
        &self,
        p: ArrayView1<'_, f64>,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;

    fn project_tangent(
        &self,
        p: ArrayView1<'_, f64>,
        v: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;

    fn retract(
        &self,
        p: ArrayView1<'_, f64>,
        xi: ArrayView1<'_, f64>,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;

    fn vector_transport(
        &self,
        from: ArrayView1<'_, f64>,
        to: ArrayView1<'_, f64>,
        xi: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;

    fn inner_product(
        &self,
        p: ArrayView1<'_, f64>,
        xi: ArrayView1<'_, f64>,
        eta: ArrayView1<'_, f64>,
    ) -> Result<f64, ManifoldError>;

    fn tangent_basis(
        &self,
        p: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, ManifoldError>;

    fn weingarten(
        &self,
        p: ArrayView1<'_, f64>,
        xi: ArrayView1<'_, f64>,
        ambient_grad: ArrayView1<'_, f64>,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;

    fn euclidean_to_riemannian_grad(
        &self,
        p: ArrayView1<'_, f64>,
        egrad: ArrayView1<'_, f64>,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError> {
        out.assign(&egrad);
        self.project_tangent(p, out)
    }

    fn euclidean_to_riemannian_hess_vp(
        &self,
        p: ArrayView1<'_, f64>,
        egrad: ArrayView1<'_, f64>,
        ehess_xi: ArrayView1<'_, f64>,
        xi: ArrayView1<'_, f64>,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), ManifoldError>;
}
```

Error type:

```rust
#[derive(Debug, Clone)]
pub enum ManifoldError {
    DimensionMismatch {
        manifold: &'static str,
        expected: usize,
        found: usize,
        role: &'static str,
    },
    InvalidPoint {
        manifold: &'static str,
        reason: String,
    },
    InvalidTangent {
        manifold: &'static str,
        reason: String,
    },
    BoundaryActive {
        manifold: &'static str,
        reason: String,
    },
}
```

Concrete kinds:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldKind {
    Euclidean { dim: usize },
    Circle,
    Sphere { intrinsic_dim: usize },
    Interval { lo: f64, hi: f64 },
    Torus { dim: usize },
    Product { components: Vec<ManifoldKind> },
    ProductWithMetric {
        components: Vec<ManifoldKind>,
        ambient_weights: Vec<f64>,
    },
}

impl ManifoldKind {
    pub fn build(&self) -> Result<Box<dyn Manifold>, ManifoldError>;
    pub fn dim(&self) -> usize;
    pub fn ambient_dim(&self) -> usize;
    pub fn is_euclidean(&self) -> bool;
}
```

Concrete structs:

```rust
#[derive(Debug, Clone)]
pub struct Euclidean { pub dim: usize }

#[derive(Debug, Clone)]
pub struct Circle;

#[derive(Debug, Clone)]
pub struct Sphere { pub intrinsic_dim: usize }

#[derive(Debug, Clone)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    pub boundary_mode: IntervalBoundaryMode,
}

#[derive(Debug, Clone)]
pub enum IntervalBoundaryMode {
    ClosedCone,
    OpenInterior { edge_band: f64 },
}

#[derive(Debug, Clone)]
pub struct Torus { pub dim: usize }

pub struct Product {
    pub components: Vec<Box<dyn Manifold>>,
    pub ambient_weights: Option<Vec<f64>>,
}
```

## 17. Final Newton and Retraction APIs

```rust
#[derive(Debug, Clone)]
pub struct RiemannianNewtonConfig {
    pub damping: f64,
    pub trust_radius: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct RiemannianNewtonOutcome {
    pub new_point: Array1<f64>,
    pub tangent_step: Array1<f64>,
    pub tangent_step_coords: Array1<f64>,
    pub projected_gradient_norm: f64,
    pub damping_used: f64,
}

pub fn riemannian_newton_step_on_point(
    manifold: &dyn Manifold,
    point: ArrayView1<'_, f64>,
    euclidean_grad: ArrayView1<'_, f64>,
    euclidean_hess: ArrayView2<'_, f64>,
    config: &RiemannianNewtonConfig,
) -> Result<RiemannianNewtonOutcome, ManifoldError>;
```

Required internal sequence:

```text
grad_R = P_p(egrad)
H_R[ξ] = P_p(H_Eξ - W_p(ξ,egrad))
Q = tangent_basis(p)
(QᵀH_RQ + μI)η = -Qᵀgrad_R
ξ = Qη
clip ξ in g_p norm if needed
new_point = R_p(ξ).
```

IFT/retraction helpers:

```rust
pub fn retract_tangent_delta(
    manifold: &dyn Manifold,
    point: ArrayView1<'_, f64>,
    delta: ArrayView1<'_, f64>,
    out: ArrayViewMut1<'_, f64>,
) -> Result<(), ManifoldError>;

pub fn retract_latent_rows(
    manifold: &dyn Manifold,
    point_flat: ArrayView1<'_, f64>,
    delta_flat: ArrayView1<'_, f64>,
    n_rows: usize,
    out_flat: ArrayViewMut1<'_, f64>,
) -> Result<(), ManifoldError>;
```

Derivative contraction extension:

```rust
use ndarray::Array3;

pub struct InputLocationHessianBlocks {
    pub grad_t: Array2<f64>,
    pub hess_t: Array3<f64>,
}

pub fn contract_input_loc_hessian(
    grad_phi: ArrayView2<'_, f64>,
    hess_phi: ArrayView2<'_, f64>,
    design_grad: ArrayView3<'_, f64>,
    design_hess: ArrayView3<'_, f64>,
) -> Result<InputLocationHessianBlocks, BasisError>;
```

The output `hess_t` is row-local ambient `m×m` blocks ready for manifold
conversion.

## 18. Weingarten Summary

```text
ℝ:
  W_p(ξ,v) = 0

Interval[a,b]:
  W_p(ξ,v) = 0
  boundary handled by tangent cone

S¹:
  W_p(ξ,v) = <p,v>ξ

S²:
  W_p(ξ,v) = <p,v>ξ

S^n:
  W_p(ξ,v) = <p,v>ξ

Torus T^d:
  W_p(ξ,v)_j = <p_j,v_j>ξ_j

Product:
  W_p(ξ,v)_j = W_{p_j}^{M_j}(ξ_j,v_j)
```

Again, for sphere-like components `v` must be ambient or normal. Passing
`P_p(∇_E)` removes the curvature correction.

## 19. Implementation Commitments

1. Store optimizer state for `S¹`, `S²`, `S^n`, torus, and products in
   embedded coordinates.
2. Keep chart coordinates only at API boundaries or chart-native basis
   evaluation boundaries.
3. Project every row gradient into the tangent space before solving.
4. Convert every row Hessian action with the Weingarten correction.
5. Solve Newton systems in tangent basis coordinates:

```text
(QᵀH_RQ + μI)η = -Qᵀgrad_R.
```

6. Lift and retract:

```text
ξ = Qη
t_new = R_p(ξ).
```

7. Treat IFT warm-start deltas as tangent vectors and apply retraction.
8. Use projection vector transport by default.
9. Handle interval boundaries as active constraints.
10. Do not add finite-difference derivative paths.
11. Remove unused chart-specific optimizer paths once embedded manifolds own
    the update path.

