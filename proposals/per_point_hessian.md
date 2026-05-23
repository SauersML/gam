# Per-point Hessian blocks for the latent-variable engine

This note derives the row-local Hessian blocks needed by the latent-coordinate
arrow solver.

It is written for the model

```text
z_i ≈ Σ_k Φ_k(t_{ik}) β_k
W_i = U_i U_i^T
r_i = z_i - Σ_k Φ_k(t_{ik}) β_k
F = 1/2 Σ_i ||r_i||^2_{W_i}
  + 1/2 Σ_k tr(β_k^T S_k β_k)
  + Σ_i P_iso
  + P_sparse
```

The target implementation sites are the current latent-coordinate machinery:

```text
src/solver/latent_inner.rs
src/solver/arrow_schur.rs
src/solver/persistent_warm_start.rs
src/terms/input_loc_derivatives.rs
src/terms/latent_coord.rs
src/terms/analytic_penalties.rs
src/linalg/low_rank_weight.rs
```

The current arrow solver expects one row-local block per observation:

```text
ArrowRowBlock {
    htt:     d x d,
    htbeta:  d x K_flat,
    gt:      d,
}
```

For the derivation below, `K_flat` means the flattened decoder coefficient
space.  When we discuss one latent component `k`, the local coefficient block
is `β_k ∈ R^{b_k x p'}` and the cross block is first derived as a
`(d_k, b_k, p')` tensor before being flattened into the row's `htbeta` slab.

## 0. Index conventions

Observation index:

```text
i = 0..n-1
```

Latent component / decoder block:

```text
k, m = 0..K_terms-1
```

Latent coordinate axis inside component `k`:

```text
a, b, c = 0..d_k-1
```

Basis column inside block `k`:

```text
α, γ = 0..b_k-1
```

Output coordinate:

```text
s, v, u = 0..p'-1
```

Low-rank weight axis:

```text
ℓ = 0..q_i-1
```

Basis row:

```text
φ_{ikα} = Φ_k(t_{ik})_α
```

First input-location derivative:

```text
J_{ikαa} = ∂φ_{ikα} / ∂t_{ik,a}
```

Second input-location derivative:

```text
H^Φ_{ikαab} = ∂²φ_{ikα} / ∂t_{ik,a} ∂t_{ik,b}
```

Decoder coefficient:

```text
β_{kαs}
```

Prediction:

```text
η_{is} = Σ_m Σ_α φ_{imα} β_{mαs}
```

Residual:

```text
r_{is} = z_{is} - η_{is}
```

Low-rank weight:

```text
W_{isv} = Σ_ℓ U_{isℓ} U_{ivℓ}
```

Weighted residual:

```text
h_{is} = (W_i r_i)_s
       = Σ_v W_{isv} r_{iv}
       = Σ_ℓ U_{isℓ} c_{iℓ}

c_{iℓ} = Σ_v U_{ivℓ} r_{iv}
       = (U_i^T r_i)_ℓ
```

Decoder tangent for latent block `k`, axis `a`:

```text
A_{ik,a,s} = ∂η_{is} / ∂t_{ik,a}
            = Σ_α J_{ikαa} β_{kαs}
```

Low-rank projected decoder tangent:

```text
B_{ik,a,ℓ} = Σ_s U_{isℓ} A_{ik,a,s}
           = (U_i^T A_{ik,a})_ℓ
```

Weighted decoder tangent:

```text
gA_{ik,a,s} = (W_i A_{ik,a})_s
            = Σ_v W_{isv} A_{ik,a,v}
            = Σ_ℓ U_{isℓ} B_{ik,a,ℓ}
```

The sign convention is important:

```text
r = z - η
```

So every residual-curvature term carries a minus sign.

## 1. Per-point Hessian block `H_{t_i t_i}`

For one observation `i` and one latent component `k`, the data-fit part is

```text
F_i = 1/2 r_i^T W_i r_i.
```

The gradient with respect to a latent coordinate `t_{ik,a}` is

```text
∂F_i / ∂t_{ik,a}
  = - A_{ik,a,s} W_{isv} r_{iv}
  = - A_{ik,a,s} h_{is}.
```

In Einstein notation, repeated output indices are summed:

```text
g^data_{ik,a}
  = - J_{ikαa} β_{kαs} W_{isv} r_{iv}.
```

The Hessian is the derivative of this gradient.

It splits into:

```text
H^data_{ik,ab}
  = H^GN_{ik,ab} + H^curv_{ik,ab}
```

where:

```text
H^GN_{ik,ab}
  = A_{ik,a,s} W_{isv} A_{ik,b,v}
```

and

```text
H^curv_{ik,ab}
  = - H^Φ_{ikαab} β_{kαs} W_{isv} r_{iv}.
```

Equivalently:

```text
H^GN_{ik,ab}
  = J_{ikαa} β_{kαs} W_{isv} β_{kγv} J_{ikγb}
```

and

```text
H^curv_{ik,ab}
  = - H^Φ_{ikαab} β_{kαs} h_{is}.
```

This is the requested matrix expression:

```text
H^GN_{t_{ik}t_{ik}}
  = J_k(t_{ik})^T β_k W_i β_k^T J_k(t_{ik})
```

with `J_k(t_{ik})` viewed as `b_k x d_k`.

The curvature term is:

```text
H^curv_{t_{ik}t_{ik}}
  = - Σ_α H^Φ_{ikα} [β_k W_i r_i]_α
```

where:

```text
[β_k W_i r_i]_α = β_{kαs} W_{isv} r_{iv}.
```

In index form:

```text
H^curv_{ik,ab}
  = - Σ_α H^Φ_{ikαab} [β_k h_i]_α.
```

### 1(a). Gauss-Newton part

The Gauss-Newton part is positive semidefinite when `W_i` is PSD.

Use the low-rank factor immediately:

```text
H^GN_{ik,ab}
  = A_{ik,a,s} U_{isℓ} U_{ivℓ} A_{ik,b,v}
  = B_{ik,a,ℓ} B_{ik,b,ℓ}.
```

Thus the per-row implementation should compute:

```text
B[a, ℓ] = Σ_s A[a, s] U[s, ℓ]
H_gn[a, b] += Σ_ℓ B[a, ℓ] B[b, ℓ]
```

It must not compute or store `W_i`.

If `q_i = 0`, the row has no data-fit curvature from the low-rank weight.

If the engine later supports `W_i = D_i + U_i U_i^T`, add the diagonal part
as:

```text
H^GN_diag_{ab} = Σ_s A_{a,s} D_{s} A_{b,s}.
```

For the model requested here, `W_i = U_i U_i^T`, so the diagonal term is not
part of the core formula.

### 1(b). Curvature part

The exact second-order residual curvature is:

```text
H^curv_{ik,ab}
  = - H^Φ_{ikαab} β_{kαs} h_{is}.
```

Use the weighted residual `h = W r` through `U`:

```text
c[ℓ] = Σ_s U[s, ℓ] r[s]
h[s] = Σ_ℓ U[s, ℓ] c[ℓ]
β_h[α] = Σ_s β[α, s] h[s]
H_curv[a, b] -= Σ_α Hphi[α, a, b] β_h[α]
```

Again, `W` is never materialized.

This term is not PSD.  It is zero at exact residual zero, and it is often
dropped in a pure Gauss-Newton approximation.

For this engine, keep both paths explicit:

```text
include_residual_curvature: bool
```

The default exact Hessian path should include it when second derivatives are
available.

If the basis cannot supply `H^Φ`, the assembler should not silently fake the
curvature term.  It should either assemble a documented Gauss-Newton Hessian
or return an error for callers that requested the exact Hessian.

### 1(c). Isometry-penalty Hessian

The isometry penalty currently lives in `src/terms/analytic_penalties.rs`.

Its documented form is:

```text
P_iso
  = 1/2 μ Σ_i ||G_i - G_i^ref||_F²
```

where:

```text
G_i = Jη_i^T W_i Jη_i
```

and `Jη_i` is the decoder Jacobian with respect to the latent row.

For one latent component `k`, use:

```text
M_i = U_i^T Jη_i
G_i = M_i^T M_i.
```

Define the metric residual:

```text
E_{iab} = G_{iab} - G^ref_{iab}.
```

For a coordinate `t_{ik,c}`:

```text
∂G_{iab} / ∂t_{ik,c}
  = (∂Jη_{i,:,a}/∂t_{ik,c})^T W_i Jη_{i,:,b}
  + Jη_{i,:,a}^T W_i (∂Jη_{i,:,b}/∂t_{ik,c}).
```

The isometry gradient is:

```text
∂P_iso / ∂t_{ik,c}
  = μ E_{iab} ∂G_{iab} / ∂t_{ik,c}.
```

The exact isometry Hessian is:

```text
∂²P_iso / ∂t_{ik,c} ∂t_{ik,d}
  = μ (∂G_{iab}/∂t_{ik,c})(∂G_{iab}/∂t_{ik,d})
  + μ E_{iab} ∂²G_{iab}/∂t_{ik,c}∂t_{ik,d}.
```

The first term is the metric-residual Gauss-Newton term.

The second term is the residual-curvature term for the metric residual.

The current `IsometryPenalty::hvp` documentation names this structure:

```text
B_{ab,cd}
  = K_{a,cd}^T W J_b
  + H_{a,c}^T W H_{b,d}
  + H_{a,d}^T W H_{b,c}
  + J_a^T W K_{b,cd}.
```

Here:

```text
H_{a,c} = ∂J_a / ∂t_c
K_{a,cd} = ∂²J_a / ∂t_c∂t_d
```

The row-local Hessian contribution is therefore:

```text
H^iso_{ik,cd}
  = μ D_{iab,c} D_{iab,d}
  + μ E_{iab} B_{iab,cd}
```

with:

```text
D_{iab,c} = ∂G_{iab}/∂t_{ik,c}.
```

If only the Gauss-Newton isometry Hessian is requested:

```text
H^iso,GN_{ik,cd}
  = μ D_{iab,c} D_{iab,d}.
```

The final row-local block is:

```text
H_{t_{ik}t_{ik}}
  = H^GN_{ik}
  + H^curv_{ik}
  + H^iso_{ik}
  + H^sparse/ard_{ik}
  + ridge_t I.
```

`H^sparse/ard` is present only when a `Psi`-tier analytic penalty targets the
latent row.  ARD contributes a diagonal block.  Sparse assignment penalties
contribute their own diagonal or local Hessian according to
`AnalyticPenaltyKind`.

The LM ridge is not part of the mathematical Hessian used for IFT.  It is a
solve-time damping term.

### 1(d). Rust pseudocode signature

The row-local data Hessian should be assembled without allocating `W`.

The signature below is intentionally close to the existing ndarray style:

```rust
pub fn assemble_per_point_hessian_block(
    phi_jacobian: ndarray::ArrayView2<'_, f64>,      // (b_k, d_k), J[alpha, a]
    phi_hessian: ndarray::ArrayView3<'_, f64>,       // (b_k, d_k, d_k)
    beta: ndarray::ArrayView2<'_, f64>,              // (b_k, p_out)
    residual: ndarray::ArrayView1<'_, f64>,          // (p_out,)
    weight_u: ndarray::ArrayView2<'_, f64>,          // (p_out, q_i)
    include_residual_curvature: bool,
    out: ndarray::ArrayViewMut2<'_, f64>,            // (d_k, d_k), incremented
) -> Result<(), String>;
```

The isometry contribution should be a separate call.  It belongs to the
analytic-penalty layer, but the arrow assembler needs a row-block entry point:

```rust
pub fn add_isometry_hessian_block_for_row(
    penalty: &crate::terms::analytic_penalties::IsometryPenalty,
    target_t_flat: ndarray::ArrayView1<'_, f64>,
    rho_iso: ndarray::ArrayView1<'_, f64>,
    row: usize,
    out: ndarray::ArrayViewMut2<'_, f64>,            // (d_k, d_k), incremented
) -> Result<(), String>;
```

The data part can be implemented with the following contraction order:

```rust
// A[a, s] = sum_alpha J[alpha, a] * beta[alpha, s]
// B[a, ell] = sum_s A[a, s] * U[s, ell]
// H[a, b] += sum_ell B[a, ell] * B[b, ell]
// c[ell] = sum_s U[s, ell] * residual[s]
// h[s] = sum_ell U[s, ell] * c[ell]
// beta_h[alpha] = sum_s beta[alpha, s] * h[s]
// H[a, b] -= sum_alpha Hphi[alpha, a, b] * beta_h[alpha]
```

The output must be symmetrized after accumulation if floating-point loop order
does not write symmetric entries identically.

## 2. Cross term `H_{t_i β_k}`

For one row and one latent component, the cross derivative is:

```text
H_{t_{ik,a}, β_{mγv}}
  = ∂²F_i / ∂t_{ik,a} ∂β_{mγv}.
```

Start from:

```text
g_{t_{ik,a}}
  = - A_{ik,a,s} h_{is}.
```

Differentiate with respect to `β_{mγv}`.

There are two effects:

1. `β_{mγv}` changes the residual.
2. If `m = k`, `β_{kγv}` also changes the tangent `A_{ik,a}`.

The residual effect is:

```text
∂h_{is} / ∂β_{mγv}
  = W_{isu} ∂r_{iu}/∂β_{mγv}
  = - W_{isv} φ_{imγ}.
```

The tangent effect is:

```text
∂A_{ik,a,s} / ∂β_{mγv}
  = 1_{m=k} J_{ikγa} 1_{s=v}.
```

Therefore:

```text
H_{t_{ik,a}, β_{mγv}}
  = φ_{imγ} A_{ik,a,s} W_{isv}
    - 1_{m=k} J_{ikγa} h_{iv}.
```

Using the weighted tangent:

```text
gA_{ik,a,v} = A_{ik,a,s} W_{isv},
```

the formula is:

```text
H_{t_{ik,a}, β_{mγv}}
  = φ_{imγ} gA_{ik,a,v}
    - 1_{m=k} J_{ikγa} h_{iv}.
```

For the same decoder block `m = k`, the requested `(d_k, b_k, p')` tensor is:

```text
H_{aγv}
  = φ_{ikγ} gA_{ik,a,v}
    - J_{ikγa} h_{iv}.
```

This is the storage convention:

```text
cross[a, γ, v] = H_{t_{ik,a}, β_{kγv}}
```

Flatten into `ArrowRowBlock.htbeta` as:

```text
col = beta_block_offset[k] + γ * p_out + v
row.htbeta[[a, col]] += cross[a, γ, v]
```

This assumes row-major flattening of `β_k`:

```text
β_k[γ, v] -> γ * p_out + v.
```

If the repository later standardizes a different coefficient layout, this is
the one line to change.  The tensor convention above should remain unchanged.

### 2(a). Low-rank contraction order

Compute:

```text
c[ℓ] = U[:, ℓ]^T r
h[v] = U[v, ℓ] c[ℓ]
```

Compute the tangent:

```text
A[a, s] = J[γ, a] β[γ, s]
```

Project:

```text
B[a, ℓ] = U[s, ℓ] A[a, s]
```

Return to output space only as a weighted vector:

```text
gA[a, v] = U[v, ℓ] B[a, ℓ]
```

Then:

```text
cross[a, γ, v] += φ[γ] * gA[a, v] - J[γ, a] * h[v].
```

No `p' x p'` matrix is materialized.

### 2(b). Cross terms for other decoder blocks

For `m != k`:

```text
H_{t_{ik,a}, β_{mγv}}
  = φ_{imγ} gA_{ik,a,v}.
```

This matters when the arrow solver's shared `β` block contains every decoder
coefficient.  A movement in `t_{ik}` changes the residual, and every decoder
block affects the residual.

In a block-local update that only solves `(t_{ik}, β_k)` and holds other
decoder blocks fixed, the assembler can omit `m != k` columns.  In the current
arrow system, `htbeta` is `d x K_flat`, so the full cross slab should be
assembled whenever the shared solve includes all `β`.

### 2(c). Penalty cross terms

The data-fit cross tensor above is not the whole story when a penalty couples
`t` and `β`.

The smoothness penalty:

```text
1/2 tr(β_k^T S_k β_k)
```

has no `t` derivative if `S_k` is fixed at the current basis construction.

If `S_k` itself depends on `t`, then the derivative belongs to the
design-moving / hyper-coordinate path, not this row-local data block.

The sparse penalty usually targets `β` or assignment logits.  It contributes
to `H_{tβ}` only if the sparse target explicitly contains both a latent row
and decoder coefficients.  The shipped sparse penalties do not do that.

The isometry penalty can couple `t` and `β` because the decoder Jacobian uses
the coefficients.  For exact Newton on the joint `(t, β)` system, add:

```text
H^iso_{t_{ik,a}, β_{mγv}}
  = ∂²P_iso / ∂t_{ik,a} ∂β_{mγv}.
```

For a Gauss-Newton isometry penalty:

```text
P_iso = 1/2 μ E_{iab} E_{iab},
```

the cross term is:

```text
H^iso,GN_{t_a,β_j}
  = μ (∂E_{iab}/∂t_a) (∂E_{iab}/∂β_j).
```

The exact isometry cross term adds:

```text
μ E_{iab} ∂²E_{iab}/∂t_a∂β_j.
```

This should be added through the analytic-penalty layer, not hidden inside
the residual data kernel.

### 2(d). Rust pseudocode signature

The data cross tensor should be available before flattening:

```rust
pub fn assemble_t_beta_cross_tensor_same_block(
    phi: ndarray::ArrayView1<'_, f64>,               // (b_k,)
    phi_jacobian: ndarray::ArrayView2<'_, f64>,      // (b_k, d_k)
    beta: ndarray::ArrayView2<'_, f64>,              // (b_k, p_out)
    residual: ndarray::ArrayView1<'_, f64>,          // (p_out,)
    weight_u: ndarray::ArrayView2<'_, f64>,          // (p_out, q_i)
    out: ndarray::ArrayViewMut3<'_, f64>,            // (d_k, b_k, p_out), incremented
) -> Result<(), String>;
```

The full shared-slab version should accept all basis rows:

```rust
pub fn assemble_t_beta_cross_slab(
    active_phi_jacobian: ndarray::ArrayView2<'_, f64>, // (b_k, d_k)
    active_beta: ndarray::ArrayView2<'_, f64>,         // (b_k, p_out)
    all_phi_rows: &[ndarray::ArrayView1<'_, f64>],     // each (b_m,)
    residual: ndarray::ArrayView1<'_, f64>,            // (p_out,)
    weight_u: ndarray::ArrayView2<'_, f64>,            // (p_out, q_i)
    beta_block_offsets: &[usize],
    active_block: usize,
    out_htbeta: ndarray::ArrayViewMut2<'_, f64>,       // (d_k, total_beta_dim)
) -> Result<(), String>;
```

The same-block tensor can be flattened into the slab with:

```rust
pub fn scatter_t_beta_cross_tensor(
    cross: ndarray::ArrayView3<'_, f64>,             // (d_k, b_k, p_out)
    beta_block_offset: usize,
    out_htbeta: ndarray::ArrayViewMut2<'_, f64>,     // (d_k, total_beta_dim)
) -> Result<(), String>;
```

## 3. Shared `H_{β_k β_k}` block with low-rank `W_i`

The shared coefficient Hessian for two decoder blocks `m` and `k` is:

```text
H_{β_{mγu}, β_{kαv}}
  = Σ_i φ_{imγ} W_{iuv} φ_{ikα}
    + 1_{m=k} S_{kγα} 1_{uv}
    + H^sparse_{β_{mγu},β_{kαv}}.
```

For one block `k`:

```text
H_{β_k β_k}^{data}
  = Σ_i (φ_{ik}^T φ_{ik}) ⊗ W_i.
```

If `W_i` were dense, this would look expensive because `W_i` is `p' x p'`.

But:

```text
W_i = U_i U_i^T
```

so:

```text
(φ_i^T φ_i) ⊗ (U_i U_i^T)
  = (φ_i^T ⊗ U_i) (φ_i ⊗ U_i^T).
```

Equivalently, each row contributes `q_i` rank-one outer products:

```text
H^{data}_i
  = Σ_ℓ x_{iℓ} x_{iℓ}^T
```

where the flattened coefficient-space vector is:

```text
x_{iℓ,kαs} = φ_{ikα} U_{isℓ}.
```

This is the block-Kronecker low-rank structure.

It stays cheap because all hot operations can use matrix-vector products.

Given a coefficient perturbation `δβ`, compute:

```text
δy_{is} = Σ_k φ_{ikα} δβ_{kαs}
```

Project through the low-rank weight:

```text
d_{iℓ} = Σ_s U_{isℓ} δy_{is}
```

Return to output space:

```text
wδy_{is} = Σ_ℓ U_{isℓ} d_{iℓ}
```

Accumulate:

```text
(H_{ββ}^{data} δβ)_{kαs}
  += Σ_i φ_{ikα} wδy_{is}.
```

The penalty part is:

```text
(S_k δβ_k)_{αs} = S_{kαγ} δβ_{kγs}.
```

The sparse penalty part comes from `AnalyticPenaltyKind::hvp` or its diagonal
majorizer.

No `p' x p'` weight is ever materialized.

Dense materialization, when requested for the direct Schur path, should still
use the rank-one structure:

```text
for i:
  for ell:
    x = phi_i ⊗ U_i[:, ell]
    H += x x^T
```

That materializes the coefficient Hessian, not the output-space weight.

For large `K_flat`, prefer the current `ArrowSchurSystem::set_shared_beta_operator`
style: expose `H_{ββ} x` and a diagonal, then use the inexact Schur PCG path.

## 4. IFT warm-start

At an inner optimum:

```text
g_{t_i}(t_i*, β) = 0.
```

Differentiate with respect to the shared coefficient vector:

```text
H_{t_i t_i} ∂t_i*/∂β + H_{t_i β} = 0.
```

Therefore:

```text
∂t_i*/∂β = - H_{t_i t_i}^{-1} H_{t_i β}.
```

For a proposed coefficient shift `δβ`:

```text
δt_i = - H_{t_i t_i}^{-1} H_{t_i β} δβ.
```

This matches the current warm-start comment in
`src/solver/persistent_warm_start.rs`:

```text
Δt_i ≈ -(H_tt^(i))⁻¹ · (H_tβ^(i) Δβ).
```

Use the undamped per-row Cholesky factors for IFT.

The LM ridge is a Newton globalization device.  It is not part of the implicit
function derivative.

If a hyperparameter or penalty shift contributes a direct row-gradient shift
`δg_{t_i}`, include it as:

```text
δt_i = -H_{t_i t_i}^{-1} (H_{t_i β} δβ + δg_{t_i}).
```

This is already the shape of:

```text
ift_warm_start_latent(cache, delta_beta, delta_gt)
```

The implementer should ensure that `ArrowFactorCache` stores:

```text
htt_factors_undamped
htbeta
d
k
```

and that the predictor uses the undamped factors.

## 5. Arrow log-det

The joint Hessian has arrow structure:

```text
H =
[ H_tt    H_tβ ]
[ H_βt    H_ββ ]
```

where:

```text
H_tt = blockdiag(H_{t_1t_1}, ..., H_{t_nt_n}).
```

The block determinant identity gives:

```text
|H| = |H_tt| |Schur_β|.
```

The Schur complement is:

```text
Schur_β
  = H_ββ - H_βt H_tt^{-1} H_tβ
  = H_ββ - Σ_i H_{βt_i} H_{t_it_i}^{-1} H_{t_iβ}.
```

Thus:

```text
log|H|
  = Σ_i log|H_{t_i t_i}|
    + log|Schur_β|.
```

The first term is row-local.

The second term is shared.

This distinction matters for REML derivatives:

```text
∂/∂t_i log|Schur_β|
  = tr(Schur_β^{-1} ∂Schur_β/∂t_i).
```

`Schur_β^{-1}` is dense in the shared coefficient space.

But:

```text
∂Schur_β/∂t_i
```

is a row-local, rank-limited update because only row `i` of the basis and its
derivative slabs move with `t_i`.

So the correct cost story is:

```text
one shared Schur inverse/factorization per outer point
+ N row-local trace contractions.
```

Do not describe the whole REML log-det derivative as independent per-row work.
Only the `Σ_i log|H_{t_i t_i}|` part factorizes trivially.

## 6. Low-rank contraction order

Every operation must contract through `U_i`.

Do not materialize:

```text
W_i ∈ R^{p' x p'}
```

For residual weighting:

```text
c = U_i^T r_i
h = U_i c
```

For the Gauss-Newton `t-t` block:

```text
A = J^T? β  // implemented as A[a, s]
B = A U_i
H_tt += B B^T
```

In index form:

```text
A_{as} = J_{αa} β_{αs}
B_{aℓ} = A_{as} U_{sℓ}
H_{ab} += B_{aℓ} B_{bℓ}
```

For the residual-curvature `t-t` term:

```text
c_ℓ = U_{sℓ} r_s
h_s = U_{sℓ} c_ℓ
βh_α = β_{αs} h_s
H_{ab} -= H^Φ_{αab} βh_α
```

For the `t-β` cross:

```text
B_{aℓ} = A_{as} U_{sℓ}
gA_{av} = U_{vℓ} B_{aℓ}
h_v = U_{vℓ} c_ℓ
cross_{aγv} = φ_γ gA_{av} - J_{γa} h_v.
```

For the `β-β` matvec:

```text
δy_s = φ_α δβ_{αs}
d_ℓ = U_{sℓ} δy_s
wδy_s = U_{sℓ} d_ℓ
out_{αs} += φ_α wδy_s
```

For isometry:

```text
M = U_i^T Jη_i
G = M^T M
```

and all metric derivatives should similarly use `U_i^T` projections of
`J`, `H`, and `K`.

The rule is simple:

```text
large output-space vectors may appear;
large output-space matrices must not.
```

## 7. Numerical stability

### 7(a). Collisions

For radial bases:

```text
r = ||t_i - c_α||.
```

At `r = 0`, formulas involving `φ'(r)/r` and
`(φ''(r) - φ'(r)/r)/r²` need analytic limits.

The existing `basis_input_loc_grad` and `basis_input_loc_hess` route through
`RadialScalarKind::eval_design_triplet`, which already encodes those limits
or returns a `BasisError::DegenerateAtCollision`.

The Hessian assembler should consume those derivative jets.  It should not
rederive ad hoc `1/r` formulas in the arrow assembler.

If the derivative routine returns a collision error, propagate it with row and
basis identifiers.

Do not silently zero a singular derivative.

Finite zero is valid only when the analytic derivative is actually zero.

### 7(b). Rank-deficient `W_i`

`W_i = U_i U_i^T` is PSD and may be rank deficient.

Then the data Gauss-Newton block:

```text
A W A^T
```

is also PSD and may be singular.

This is expected when:

```text
q_i < d_k
```

or when decoder tangents are locally collinear under `U_i`.

Identifiability must come from:

```text
P_iso
ARD / auxiliary priors
other explicit gauge-fixing penalties
```

Do not assume the data term alone can factor `H_{t_i t_i}`.

### 7(c). Ridge regularization

There are two different ridges.

The mathematical row Hessian is:

```text
H_{t_i t_i}.
```

The damped Newton solve uses:

```text
H_{t_i t_i} + λ_t I.
```

The IFT warm-start uses:

```text
H_{t_i t_i}^{-1}
```

not:

```text
(H_{t_i t_i} + λ_t I)^{-1}.
```

If the undamped factorization fails, the implementation should report that
the IFT predictor is unavailable for that point.  The next inner solve can
still use LM damping.

For the shared Schur solve, the damped system is:

```text
Schur_β(λ_t, λ_β)
  = H_ββ + λ_β I
    - Σ_i H_βt_i (H_t_it_i + λ_t I)^{-1} H_t_iβ.
```

For log-det evidence, use the same Hessian definition as the objective path
being evaluated.  Do not mix damped Newton ridges into REML evidence unless
the objective explicitly includes them.

### 7(d). Woodbury identity

When a diagonal-plus-low-rank weight appears:

```text
W = D + U U^T,
```

the row-space Woodbury identity is:

```text
W^{-1}
  = D^{-1}
    - D^{-1} U (I + U^T D^{-1} U)^{-1} U^T D^{-1}.
```

The parameter-space Gram identity is:

```text
(A + X^T U U^T X)^{-1}
  = A^{-1}
    - A^{-1} X^T U
      (I + U^T X A^{-1} X^T U)^{-1}
      U^T X A^{-1}.
```

This is the same shape implemented in `src/linalg/low_rank_weight.rs`.

For the pure `W_i = U_i U_i^T` model, there may be no invertible diagonal
`D`.  Do not use row-space Woodbury unless `D` is actually present and
positive.

For Hessian assembly, Woodbury is usually unnecessary.  The required operation
is applying `W`, not inverting it:

```text
W x = U (U^T x).
```

Use Woodbury only for solves with a diagonal-plus-low-rank Gram where the base
system `A` is already factored.

### 7(e). Symmetry and sign checks

The `t-t` block must be symmetric:

```text
H_{ab} = H_{ba}.
```

The Gauss-Newton part is symmetric by construction.

The curvature term is symmetric only if the input-location Hessian satisfies:

```text
H^Φ_{αab} = H^Φ_{αba}.
```

After accumulation, symmetrize:

```text
H = 0.5 * (H + H^T)
```

for the row-local block.

The cross slab does not need to be symmetric.  Its transpose lives in the
global `H_βt` block by construction.

The sign check for the curvature term is:

```text
r = z - η  =>  curvature is negative.
```

If the code ever switches to:

```text
r = η - z
```

then both the gradient and curvature sign conventions must be audited.

## 8. Implementation mapping

### 8(a). Existing row structure

The assembler should populate:

```text
row.htt[a, b]       += H_{t_i t_i}[a, b]
row.htbeta[a, col]  += H_{t_i β}[a, col]
row.gt[a]           += g_{t_i}[a]
sys.hbb[col, col2]  += H_{ββ}[col, col2]
sys.gb[col]         += g_β[col]
```

For one row and one active latent block:

```text
g_t[a] = - A[a, s] h[s] + g_t^penalty[a].
```

For beta:

```text
g_β[k, α, s] = - Σ_i φ_{ikα} h_{is} + (S_k β_k)_{αs} + g_β^sparse.
```

The same `h = U(U^T r)` should be reused for:

```text
g_t
H_curv
H_tβ
g_β
```

### 8(b). Basis derivative storage

The current `basis_input_loc_hess` stores:

```text
(n_obs, n_centers, d*d)
```

with:

```text
packed = a * d + b.
```

The row-local Hessian assembler can accept either:

```text
(b_k, d_k, d_k)
```

or the packed row view:

```text
(b_k, d_k * d_k).
```

Prefer the unpacked view for the public row function and provide a small
adapter for the packed derivative cache.

### 8(c). Flattened coefficient layout

Use row-major block layout:

```text
β_k[α, s] -> beta_block_offsets[k] + α * p_out + s.
```

This makes a single basis coefficient's output vector contiguous.

It also matches the natural ndarray `Array2` row-major interpretation for a
`b_k x p_out` coefficient matrix.

### 8(d). Error checks

Every assembler function should check:

```text
phi.len() == b_k
phi_jacobian.dim() == (b_k, d_k)
phi_hessian.dim() == (b_k, d_k, d_k)
beta.dim() == (b_k, p_out)
residual.len() == p_out
weight_u.nrows() == p_out
out dimensions match
```

For the full cross slab:

```text
out_htbeta.nrows() == d_k
out_htbeta.ncols() >= beta_block_offsets[last] + b_last * p_out
```

Report row and block identifiers at the call site.

## 9. Exact Rust signatures for implementers

Use these signatures as the implementation contract.

```rust
pub struct PerPointDataHessianOptions {
    pub include_residual_curvature: bool,
    pub symmetrize_output: bool,
}
```

```rust
pub fn assemble_per_point_data_hessian_block(
    phi_jacobian: ndarray::ArrayView2<'_, f64>,
    phi_hessian: ndarray::ArrayView3<'_, f64>,
    beta: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView1<'_, f64>,
    weight_u: ndarray::ArrayView2<'_, f64>,
    options: PerPointDataHessianOptions,
    out: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn assemble_per_point_data_hessian_block_from_packed_hphi(
    phi_jacobian: ndarray::ArrayView2<'_, f64>,
    phi_hessian_packed: ndarray::ArrayView2<'_, f64>,
    beta: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView1<'_, f64>,
    weight_u: ndarray::ArrayView2<'_, f64>,
    options: PerPointDataHessianOptions,
    out: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn assemble_t_beta_cross_tensor_same_block(
    phi: ndarray::ArrayView1<'_, f64>,
    phi_jacobian: ndarray::ArrayView2<'_, f64>,
    beta: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView1<'_, f64>,
    weight_u: ndarray::ArrayView2<'_, f64>,
    out: ndarray::ArrayViewMut3<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn assemble_t_beta_cross_slab(
    active_phi_jacobian: ndarray::ArrayView2<'_, f64>,
    active_beta: ndarray::ArrayView2<'_, f64>,
    all_phi_rows: &[ndarray::ArrayView1<'_, f64>],
    residual: ndarray::ArrayView1<'_, f64>,
    weight_u: ndarray::ArrayView2<'_, f64>,
    beta_block_offsets: &[usize],
    active_block: usize,
    out_htbeta: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn scatter_t_beta_cross_tensor(
    cross: ndarray::ArrayView3<'_, f64>,
    beta_block_offset: usize,
    out_htbeta: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn add_isometry_hessian_block_for_row(
    penalty: &crate::terms::analytic_penalties::IsometryPenalty,
    target_t_flat: ndarray::ArrayView1<'_, f64>,
    rho_iso: ndarray::ArrayView1<'_, f64>,
    row: usize,
    out: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn add_isometry_t_beta_cross_for_row(
    penalty: &crate::terms::analytic_penalties::IsometryPenalty,
    target_t_flat: ndarray::ArrayView1<'_, f64>,
    beta_flat: ndarray::ArrayView1<'_, f64>,
    rho_iso: ndarray::ArrayView1<'_, f64>,
    row: usize,
    beta_block_offsets: &[usize],
    out_htbeta: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn beta_beta_low_rank_matvec(
    phi_rows_by_block: &[ndarray::ArrayView1<'_, f64>],
    delta_beta_blocks: &[ndarray::ArrayView2<'_, f64>],
    weight_u: ndarray::ArrayView2<'_, f64>,
    out_blocks: &mut [ndarray::ArrayViewMut2<'_, f64>],
) -> Result<(), String>;
```

```rust
pub fn add_beta_beta_low_rank_dense_for_row(
    phi_rows_by_block: &[ndarray::ArrayView1<'_, f64>],
    weight_u: ndarray::ArrayView2<'_, f64>,
    beta_block_offsets: &[usize],
    total_beta_dim: usize,
    out_hbb: ndarray::ArrayViewMut2<'_, f64>,
) -> Result<(), String>;
```

```rust
pub fn flatten_beta_index(
    beta_block_offsets: &[usize],
    block: usize,
    basis_col: usize,
    output_col: usize,
    p_out: usize,
) -> usize;
```

```rust
pub fn add_per_point_blocks_to_arrow_row(
    phi: ndarray::ArrayView1<'_, f64>,
    phi_jacobian: ndarray::ArrayView2<'_, f64>,
    phi_hessian: ndarray::ArrayView3<'_, f64>,
    beta: ndarray::ArrayView2<'_, f64>,
    residual: ndarray::ArrayView1<'_, f64>,
    weight_u: ndarray::ArrayView2<'_, f64>,
    beta_block_offsets: &[usize],
    active_block: usize,
    options: PerPointDataHessianOptions,
    row: &mut crate::solver::arrow_schur::ArrowRowBlock,
) -> Result<(), String>;
```

The last signature is the convenience wrapper the latent assembler should
call.  The smaller functions above are the testable kernels.
