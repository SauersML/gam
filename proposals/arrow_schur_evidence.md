# Arrow Schur Evidence, IFT Cascades, and Topology Selection

## 0. Scope

This note derives the arrow-structured inner solve, the implicit-function
cascade across `(β, u, ρ, T)`, the Laplace evidence gradient with respect to
`ρ`, and the discrete topology-selection rule.

The target implementation context is the existing `src/solver/arrow_schur.rs`
module, the REML log-determinant machinery under `src/solver/reml`, the
analytic penalty tiering in `src/terms/analytic_penalties.rs`, and the Python
topology selection helper in `gamfit/select_topology.py`.

Notation:

1. `β ∈ R^K` is the shared coefficient vector.
2. `u = (u_1, ..., u_N)` is the row-local latent field.
3. `u_i ∈ R^d` is the latent coordinate for row `i`.
4. `ρ ∈ R^R` is the vector of log smoothing or analytic-penalty parameters.
5. `T` is a discrete topology or basis family.
6. `F(β, u; ρ, T)` is the penalized objective being minimized.
7. `g = ∇F` and `H = ∇²F` are taken with respect to `(u, β)` unless stated.
8. `S(ρ)` is reserved for the penalty matrix or its pseudo-logdet term.
9. `S_arrow` is avoided as a name so it does not collide with the penalty.
10. `A` denotes the Schur complement when a scalar name is needed.

The current code calls the row-local coordinate `t` in several places.  This
document uses `u` because the requested derivation uses `u`; map `u` to `t`
when reading `ArrowRowBlock`, `ArrowSchurSystem`, and `LatentCoordValues`.

The central structural assumption is the arrow Hessian:

```text
        [ H_uu   H_uβ ]
H   =   [ H_βu   H_ββ ]
```

where `H_uu` is block diagonal over rows:

```text
H_uu = blockdiag(H_uu_1, ..., H_uu_N).
```

The load-bearing reason this is fast is that each `H_uu_i` is only `d × d`.

## 1. Schur Step

### 1.1 Newton System

At a current iterate `(β, u)`, write the Newton system as:

```text
[ H_uu   H_uβ ] [ Δu ]   =   - [ g_u ]
[ H_βu   H_ββ ] [ Δβ ]       [ g_β ]
```

The first block row is:

```text
H_uu Δu + H_uβ Δβ = -g_u.
```

The second block row is:

```text
H_βu Δu + H_ββ Δβ = -g_β.
```

Because `H_uu` is block diagonal, its inverse is never formed globally.

Instead:

```text
H_uu⁻¹ = blockdiag(H_uu_1⁻¹, ..., H_uu_N⁻¹).
```

### 1.2 Eliminate the Local Field

From the first block row:

```text
H_uu Δu = -g_u - H_uβ Δβ.
```

Multiplying by `H_uu⁻¹` gives:

```text
Δu = -H_uu⁻¹(g_u + H_uβ Δβ).
```

Row by row:

```text
Δu_i = -H_uu_i⁻¹ (g_u_i + H_uβ_i Δβ).
```

Substitute this into the second block row:

```text
H_βu[-H_uu⁻¹(g_u + H_uβ Δβ)] + H_ββ Δβ = -g_β.
```

Expand:

```text
-H_βu H_uu⁻¹ g_u - H_βu H_uu⁻¹ H_uβ Δβ + H_ββ Δβ = -g_β.
```

Collect the `Δβ` terms:

```text
(H_ββ - H_βu H_uu⁻¹ H_uβ) Δβ =
-g_β + H_βu H_uu⁻¹ g_u.
```

Define the Schur complement:

```text
S = H_ββ - H_βu H_uu⁻¹ H_uβ
```

and the reduced right-hand side:

```text
r = -g_β + H_βu H_uu⁻¹ g_u.
```

Then solve:

```text
S Δβ = r.
```

Finally back-substitute:

```text
Δu_i = -H_uu_i⁻¹ (g_u_i + H_uβ_i Δβ).
```

These are exactly the requested equations:

```text
S = H_ββ - H_βu H_uu⁻¹ H_uβ
r = -g_β + H_βu H_uu⁻¹ g_u
S Δβ = r
Δu_i = -H_uu_i⁻¹ (g_u_i + H_uβ_i Δβ)
```

### 1.3 Per-Row Expansion

Because `H_uu` is block diagonal:

```text
H_βu H_uu⁻¹ H_uβ =
Σ_i H_βu_i H_uu_i⁻¹ H_uβ_i.
```

Since `H_βu_i = H_uβ_iᵀ` for symmetric Hessians:

```text
S = H_ββ - Σ_i H_uβ_iᵀ H_uu_i⁻¹ H_uβ_i.
```

Similarly:

```text
r = -g_β + Σ_i H_uβ_iᵀ H_uu_i⁻¹ g_u_i.
```

The row contribution to `S` is a rank-at-most-`d` negative update:

```text
S_i = H_uβ_iᵀ H_uu_i⁻¹ H_uβ_i.
```

The row contribution to `r` is:

```text
r_i = H_uβ_iᵀ H_uu_i⁻¹ g_u_i.
```

The direct dense assembly path does:

```text
S ← H_ββ
for i in 1..N:
    factor H_uu_i
    Y_i ← H_uu_i⁻¹ H_uβ_i
    S ← S - H_uβ_iᵀ Y_i
    z_i ← H_uu_i⁻¹ g_u_i
    r ← r + H_uβ_iᵀ z_i
r ← r - g_β
```

The square-root BA path uses a Cholesky factor `L_i L_iᵀ = H_uu_i`:

```text
H_uβ_iᵀ H_uu_i⁻¹ H_uβ_i =
(L_i⁻¹ H_uβ_i)ᵀ (L_i⁻¹ H_uβ_i).
```

This avoids explicit inverse products and is the numerically preferred direct
assembly when row blocks are poorly conditioned.

### 1.4 Shapes

Global shapes:

```text
β        : K
u        : N d
g_β      : K
g_u      : N d
H_ββ     : K × K
H_uu     : (N d) × (N d), block diagonal
H_uβ     : (N d) × K
H_βu     : K × (N d)
S        : K × K
r        : K
Δβ       : K
Δu       : N d
```

Per-row shapes:

```text
u_i        : d
g_u_i      : d
H_uu_i     : d × d
H_uβ_i     : d × K
H_βu_i     : K × d
Y_i        : d × K
z_i        : d
S_i        : K × K
r_i        : K
Δu_i       : d
```

Current Rust mapping:

```text
ArrowSchurSystem.rows[i].htt     = H_uu_i
ArrowSchurSystem.rows[i].htbeta  = H_uβ_i
ArrowSchurSystem.rows[i].gt      = g_u_i
ArrowSchurSystem.hbb             = H_ββ
ArrowSchurSystem.gb              = g_β
```

The returned `delta_t` in the current Rust API is the flat row-major `Δu`.

### 1.5 Complexity

Let:

```text
N = number of rows
d = latent dimension per row
K = number of shared β coefficients
R = number of ρ coordinates
```

For dense direct Schur assembly:

```text
factor each H_uu_i             : O(N d³)
solve H_uu_i⁻¹ g_u_i           : O(N d²)
solve H_uu_i⁻¹ H_uβ_i          : O(N d² K)
rank-d Schur update            : O(N d K²)
factor S                       : O(K³)
solve S Δβ = r                 : O(K²)
back-substitute Δu             : O(N d K + N d²)
```

The dominant dense-direct cost is:

```text
O(N d³ + N d² K + N d K² + K³).
```

When `d` is tiny and fixed, this is often summarized as:

```text
O(N K² + K³).
```

That shorthand is only valid for dense `H_uβ_i`.

If each row touches only `K_i` β coefficients, replace `K` by `K_i` inside the
row-local terms:

```text
Σ_i O(d³ + d² K_i + d K_i²) + O(K³).
```

For matrix-free inexact PCG:

```text
one Schur matvec:
    H_ββ x                         : cost of shared operator
    row products H_uβ_i x           : O(Σ_i d K_i)
    row solves H_uu_i⁻¹(...)        : O(N d²)
    row products H_uβ_iᵀ(...)       : O(Σ_i d K_i)
```

With dense row slabs:

```text
Schur matvec = O(cost(H_ββ x) + N d K + N d²).
```

The inexact solve cost is:

```text
O(N d³) + n_pcg · O(cost(H_ββ x) + N d K + N d²).
```

This avoids storing and factoring the dense `K × K` Schur complement.

### 1.6 Parallelism

The per-row operations are embarrassingly parallel:

1. Factor `H_uu_i`.
2. Solve `H_uu_i⁻¹ g_u_i`.
3. Solve `H_uu_i⁻¹ H_uβ_i`.
4. Produce row-local Schur contributions.
5. Back-substitute `Δu_i`.

The direct dense Schur assembly needs a reduction into a shared `K × K`
matrix:

```text
S = H_ββ - Σ_i S_i.
```

Practical CPU reduction:

1. Partition rows into worker chunks.
2. Build one local `K × K` accumulator per worker.
3. Reduce worker accumulators into `S`.
4. Symmetrize once after reduction.

Practical GPU reduction:

1. Batch factor tiny `d × d` row blocks.
2. Batch triangular solves for `H_uβ_i`.
3. Emit row-local rank-`d` updates.
4. Reduce updates by tiles of the `K × K` matrix.

The current `BatchedBlockSolver` trait already marks this boundary.

### 1.7 Ridge Strategy

Two ridges are needed and they have different meanings:

```text
ridge_u    added to each H_uu_i
ridge_beta added to H_ββ or to the reduced S diagonal
```

The damped row block is:

```text
H_uu_i(λ_u) = H_uu_i + ridge_u I_d.
```

The damped shared block is:

```text
H_ββ(λ_β) = H_ββ + ridge_beta I_K.
```

The damped Schur complement is:

```text
S(λ_u, λ_β) =
H_ββ + ridge_beta I_K
- Σ_i H_uβ_iᵀ [H_uu_i + ridge_u I_d]⁻¹ H_uβ_i.
```

Ridge policy:

1. Try the unregularized Newton step when row and Schur Cholesky succeed.
2. If a row block fails, increase `ridge_u`.
3. If the Schur complement fails, increase `ridge_beta`.
4. If both fail or the accepted objective worsens, increase both.
5. After accepting steps consistently, decrease ridges.

The row ridge changes the eliminated model, so it must not leak into IFT
sensitivities.  The current `ArrowFactorCache` correctly keeps damped factors
for the Newton step and undamped factors for IFT prediction.

The IFT factor must be:

```text
H_uu_i⁻¹
```

not:

```text
(H_uu_i + ridge_u I)⁻¹.
```

The reason is that the IFT differentiates the stationarity equation, not the
trust-region or Levenberg surrogate.

### 1.8 Sign Convention

The system is written as:

```text
H Δ = -g.
```

Therefore:

```text
Δβ = S⁻¹(-g_β + H_βu H_uu⁻¹ g_u).
```

and:

```text
Δu_i = -H_uu_i⁻¹(g_u_i + H_uβ_i Δβ).
```

This matches the current `solve_arrow_newton_step` convention.

## 2. IFT Cascade Through Tiers

### 2.1 Stationarity Equations

Let:

```text
θ = (u, β)
```

and define the stationarity equations:

```text
G_u(u, β, ρ, T) = ∂F/∂u = 0
G_β(u, β, ρ, T) = ∂F/∂β = 0
```

Stack them:

```text
G(θ, ρ, T) = [G_u; G_β] = 0.
```

At a local optimum:

```text
θ*(ρ, T) = (u*(ρ, T), β*(ρ, T)).
```

IFT gives:

```text
∂θ*/∂ρ_a = -H⁻¹ ∂G/∂ρ_a.
```

The arrow structure lets us compute this without forming `H⁻¹`.

### 2.2 Local Sensitivity to β

Hold `ρ` and `T` fixed and solve the row-local stationarity:

```text
G_u(u*(β), β, ρ, T) = 0.
```

Differentiate with respect to `β`:

```text
G_{u,u} ∂u*/∂β + G_{u,β} = 0.
```

Since:

```text
G_{u,u} = H_uu
G_{u,β} = H_uβ
```

we get:

```text
∂u*/∂β = -H_uu⁻¹ H_uβ.
```

Row by row:

```text
∂u_i*/∂β = -H_uu_i⁻¹ H_uβ_i.
```

Shape:

```text
∂u_i*/∂β : d × K
∂u*/∂β   : (N d) × K
```

This is the same matrix used in Schur elimination.

### 2.3 Reduced β Stationarity

Define the profiled objective:

```text
F_red(β; ρ, T) = F(β, u*(β, ρ, T); ρ, T).
```

The reduced β stationarity is:

```text
g_red(β, ρ, T) = ∂F_red/∂β = G_β(u*(β, ρ, T), β, ρ, T) = 0.
```

Its derivative with respect to `β` is:

```text
∂g_red/∂β =
H_ββ + H_βu ∂u*/∂β.
```

Substitute:

```text
∂g_red/∂β =
H_ββ - H_βu H_uu⁻¹ H_uβ.
```

Therefore:

```text
∂g_red/∂β = S.
```

The Schur complement is the Hessian of the profiled problem.

### 2.4 β Sensitivity to ρ

At the profiled optimum:

```text
g_red(β*(ρ), ρ, T) = 0.
```

Differentiate:

```text
S ∂β*/∂ρ_a + ∂g_red/∂ρ_a = 0.
```

So:

```text
∂β*/∂ρ_a = -S⁻¹ ∂g_red/∂ρ_a.
```

The requested shorthand:

```text
∂β*/∂ρ = -S⁻¹ ∂g_β/∂ρ
```

is exact when `∂g_β/∂ρ` means the profiled derivative:

```text
∂g_red/∂ρ =
G_{β,ρ} + H_βu ∂u*/∂ρ|β.
```

If `u` has no direct `ρ` dependence at fixed `β`, then:

```text
∂g_red/∂ρ = G_{β,ρ}.
```

If `u` has direct `ρ` penalties or `ρ` moves the basis, then the full term is
required.

### 2.5 Direct u Sensitivity to ρ at Fixed β

At fixed `β`, row-local stationarity gives:

```text
G_u(u*(β, ρ), β, ρ, T) = 0.
```

Differentiate with respect to `ρ_a`:

```text
H_uu ∂u*/∂ρ_a|β + G_{u,ρ_a} = 0.
```

Thus:

```text
∂u*/∂ρ_a|β = -H_uu⁻¹ G_{u,ρ_a}.
```

Row by row:

```text
∂u_i*/∂ρ_a|β = -H_uu_i⁻¹ G_{u_i,ρ_a}.
```

This is the second warm-start path exposed by the current factor cache:

```text
predict_delta_t_from_delta_gt
```

where `δg_t` is the gradient perturbation induced by `δρ`.

### 2.6 Full u Sensitivity to ρ

The full derivative of `u*(ρ)` includes the β movement:

```text
∂u*/∂ρ_a =
∂u*/∂ρ_a|β + ∂u*/∂β · ∂β*/∂ρ_a.
```

Substitute:

```text
∂u*/∂ρ_a =
-H_uu⁻¹ G_{u,ρ_a}
-H_uu⁻¹ H_uβ ∂β*/∂ρ_a.
```

Using:

```text
∂β*/∂ρ_a = -S⁻¹ ∂g_red/∂ρ_a
```

gives:

```text
∂u*/∂ρ_a =
-H_uu⁻¹ G_{u,ρ_a}
+ H_uu⁻¹ H_uβ S⁻¹ ∂g_red/∂ρ_a.
```

This is the full IFT cascade through `u`.

### 2.7 Three-Tier Chain

The engine has three continuous tiers:

```text
β tier: shared coefficients and decoder weights
u tier: row-local latent coordinates or ψ-like fields
ρ tier: log smoothing, sparsity, ARD, and analytic penalty strengths
```

For any differentiable scalar diagnostic:

```text
Q(β*, u*, ρ, T)
```

the full derivative with respect to `ρ_a` is:

```text
dQ/dρ_a =
Q_{ρ_a}
+ Q_βᵀ ∂β*/∂ρ_a
+ Q_uᵀ ∂u*/∂ρ_a.
```

Substitute the cascade:

```text
dQ/dρ_a =
Q_{ρ_a}
- Q_βᵀ S⁻¹ ∂g_red/∂ρ_a
- Q_uᵀ H_uu⁻¹ G_{u,ρ_a}
+ Q_uᵀ H_uu⁻¹ H_uβ S⁻¹ ∂g_red/∂ρ_a.
```

At an exact optimum for `F`, the envelope term for `F` itself simplifies
because:

```text
F_β = 0
F_u = 0.
```

But the log-determinant terms do not generally simplify this way because they
depend on Hessian derivatives, not only on first derivatives of `F`.

### 2.8 Topology Tier

The topology `T` is discrete.

For each fixed `T`, there is a separate continuous problem:

```text
(β_T*, u_T*, ρ_T*) = argmin_{β,u,ρ} V(ρ, T)
```

or, depending on sign convention:

```text
(β_T*, u_T*, ρ_T*) = argmax_{β,u,ρ} Evidence(ρ, T).
```

No derivative `∂/∂T` is needed.

The cascade is:

```text
T
  -> basis family and manifold constraints
  -> Φ_T(u), penalties S_T(ρ), analytic penalty registry
  -> inner optimum (β_T*, u_T*)
  -> outer optimum ρ_T*
  -> scalar evidence V(ρ_T*, T)
```

Topology selection is the finite comparison:

```text
T* = argmax_T V(ρ_T*, T).
```

## 3. Laplace Evidence and ∂V/∂ρ

### 3.1 Objective and Evidence

Use the requested evidence form:

```text
V(ρ, T) =
F(β*, u*; ρ, T)
+ (1/2) log|H|
- (1/2) log|S(ρ)|+
```

Here:

1. `H = ∇²_{(u,β)} F(β*, u*; ρ, T)`.
2. `S(ρ)` in `log|S(ρ)|+` is the penalty matrix, not the arrow Schur matrix.
3. `|.|+` is the pseudo-determinant over the positive eigenspace.
4. Constants independent of `(ρ, T)` are omitted.

This sign convention treats `V` as a minimized negative log evidence.  If the
public API ranks larger scores as better, export `Evidence = -V` or rename the
scalar consistently.  The topology section below uses the user's requested
`argmax_T V`; therefore implementation must either store the maximizing
version or negate this formula before ranking.

### 3.2 Envelope Derivative of F

At the inner optimum:

```text
F_β(β*, u*; ρ, T) = 0
F_u(β*, u*; ρ, T) = 0.
```

Therefore:

```text
d/dρ_a F(β*(ρ), u*(ρ); ρ, T) =
F_{ρ_a}(β*, u*; ρ, T).
```

This is the envelope theorem.

For canonical quadratic penalties:

```text
P(β; ρ) = (1/2) Σ_a exp(ρ_a) βᵀ S_a β.
```

Then:

```text
F_{ρ_a} = (1/2) exp(ρ_a) βᵀ S_a β.
```

For analytic penalties, use the penalty's own:

```text
grad_rho(target, rho)
```

and route by tier.

### 3.3 Derivative of log|H|

For nonsingular `H`:

```text
d/dρ_a log|H| = tr(H⁻¹ dH/dρ_a total).
```

The total derivative includes direct `ρ` dependence and movement of the inner
optimum:

```text
dH/dρ_a total =
H_{ρ_a}
+ D_β H [∂β*/∂ρ_a]
+ D_u H [∂u*/∂ρ_a].
```

Thus:

```text
d/dρ_a (1/2 log|H|)
= (1/2) tr(H⁻¹ H_{ρ_a})
+ (1/2) tr(H⁻¹ D_β H [∂β*/∂ρ_a])
+ (1/2) tr(H⁻¹ D_u H [∂u*/∂ρ_a]).
```

If a Gauss-Newton Hessian is treated as frozen with respect to `(β, u)`, the
last two terms are intentionally dropped.  That is an approximation and should
be named as such.

For exact Laplace evidence, keep the terms.

### 3.4 Arrow Log-Det Factorization

For the arrow Hessian:

```text
H = [ H_uu   H_uβ ]
    [ H_βu   H_ββ ]
```

with invertible `H_uu`, the determinant factors as:

```text
|H| = |H_uu| · |A|
```

where:

```text
A = H_ββ - H_βu H_uu⁻¹ H_uβ.
```

Therefore:

```text
log|H| = Σ_i log|H_uu_i| + log|A|.
```

This is the function `arrow_log_det` should expose.

If ridges are present, the log determinant is of the ridged surrogate:

```text
log|H_ridged| =
Σ_i log|H_uu_i + ridge_u I|
+ log|A(ridge_u, ridge_beta)|.
```

Evidence should normally use the undamped optimum Hessian.  Ridges are solver
stabilizers, not prior mass, unless explicitly modeled.

### 3.5 Arrow Trace Formula

For any parameter `α`, the derivative of the arrow log determinant can be
computed as:

```text
∂α log|H| =
Σ_i tr(H_uu_i⁻¹ ∂α H_uu_i)
+ tr(A⁻¹ ∂α A).
```

The Schur derivative is:

```text
∂α A =
∂α H_ββ
- ∂α H_βu H_uu⁻¹ H_uβ
- H_βu H_uu⁻¹ ∂α H_uβ
+ H_βu H_uu⁻¹ (∂α H_uu) H_uu⁻¹ H_uβ.
```

Row by row:

```text
∂α A =
∂α H_ββ
- Σ_i [
    ∂α H_βu_i H_uu_i⁻¹ H_uβ_i
  + H_βu_i H_uu_i⁻¹ ∂α H_uβ_i
  - H_βu_i H_uu_i⁻¹ (∂α H_uu_i) H_uu_i⁻¹ H_uβ_i
].
```

When the Hessian is symmetric:

```text
∂α H_βu_i = (∂α H_uβ_i)ᵀ.
```

This formula is useful for `α = ρ_a`, `α = β_j`, and `α = u_{i,c}`.

### 3.6 Penalty Pseudo-Logdet

Let the penalty matrix be:

```text
S_pen(ρ) = Σ_a λ_a S_a,     λ_a = exp(ρ_a).
```

Assume the nullspace is fixed over `ρ`.

Let:

```text
S_pen = U_+ Λ_+ U_+ᵀ
S_pen⁺ = U_+ Λ_+⁻¹ U_+ᵀ.
```

Then:

```text
log|S_pen(ρ)|+ = Σ_{j: λ_j>0} log λ_j.
```

The first derivative is:

```text
∂/∂ρ_a log|S_pen(ρ)|+
= tr(S_pen⁺ ∂S_pen/∂ρ_a).
```

For log-scale smoothing:

```text
∂S_pen/∂ρ_a = exp(ρ_a) S_a.
```

Therefore:

```text
∂/∂ρ_a log|S_pen(ρ)|+
= exp(ρ_a) tr(S_pen⁺ S_a).
```

This matches the existing `PenaltyPseudologdet` design in
`src/solver/reml/penalty_logdet.rs`.

If the penalty matrix itself moves with a basis parameter `ψ`, use:

```text
∂ψ log|S_pen|+ = tr(S_pen⁺ S_{pen,ψ})
```

plus the moving-nullspace correction for second derivatives.  First
derivatives only need a stable rank and a consistent positive eigenspace.

### 3.7 Full ∂V/∂ρ

For each coordinate `ρ_a`:

```text
∂V/∂ρ_a =
F_{ρ_a}
+ (1/2) tr(H⁻¹ dH/dρ_a total)
- (1/2) tr(S_pen⁺ ∂S_pen/∂ρ_a).
```

Expanding the total Hessian derivative:

```text
∂V/∂ρ_a =
F_{ρ_a}
+ (1/2) tr(H⁻¹ H_{ρ_a})
+ (1/2) tr(H⁻¹ D_β H [∂β*/∂ρ_a])
+ (1/2) tr(H⁻¹ D_u H [∂u*/∂ρ_a])
- (1/2) tr(S_pen⁺ S_{pen,ρ_a}).
```

Using the IFT cascade:

```text
∂β*/∂ρ_a = -A⁻¹ q_a
```

where:

```text
q_a = ∂g_red/∂ρ_a.
```

and:

```text
∂u*/∂ρ_a =
-H_uu⁻¹ G_{u,ρ_a}
-H_uu⁻¹ H_uβ ∂β*/∂ρ_a.
```

Substitute:

```text
∂V/∂ρ_a =
F_{ρ_a}
+ (1/2) tr(H⁻¹ H_{ρ_a})
- (1/2) tr(S_pen⁺ S_{pen,ρ_a})
- (1/2) tr(H⁻¹ D_β H [A⁻¹ q_a])
+ (1/2) tr(H⁻¹ D_u H [
      -H_uu⁻¹ G_{u,ρ_a}
      +H_uu⁻¹ H_uβ A⁻¹ q_a
   ]).
```

This is the analytic gradient with the full Laplace correction.

### 3.8 Practical Split

Implementation should split the gradient into named parts:

```text
grad_value[a] =
F_{ρ_a}

grad_h_direct[a] =
0.5 tr(H⁻¹ H_{ρ_a})

grad_h_ift_beta[a] =
0.5 tr(H⁻¹ D_β H [∂β*/∂ρ_a])

grad_h_ift_u[a] =
0.5 tr(H⁻¹ D_u H [∂u*/∂ρ_a])

grad_pen_logdet[a] =
-0.5 tr(S_pen⁺ S_{pen,ρ_a})
```

Then:

```text
grad[a] =
grad_value[a]
+ grad_h_direct[a]
+ grad_h_ift_beta[a]
+ grad_h_ift_u[a]
+ grad_pen_logdet[a].
```

This makes it possible to test exact, Gauss-Newton-frozen, and projected
variants without hiding missing terms.

### 3.9 When the Hessian Is the Penalized Normal Matrix

For Gaussian/PIRLS normal equations:

```text
H_ββ = Xᵀ W X + S_pen(ρ)
```

If `u` is absent:

```text
∂V/∂ρ_a =
0.5 βᵀ S_{pen,ρ_a} β
+ 0.5 tr(H⁻¹ S_{pen,ρ_a})
- 0.5 tr(S_pen⁺ S_{pen,ρ_a}).
```

With `u`, the same expression holds for the direct part, but `H⁻¹` is the
arrow inverse and the IFT movement terms enter if `X`, `W`, or the row Hessian
depends on `(β, u)`.

### 3.10 Arrow Inverse Blocks

Some trace terms can be evaluated through arrow inverse blocks.

For:

```text
A = H_ββ - H_βu H_uu⁻¹ H_uβ
```

the inverse blocks are:

```text
H⁻¹_ββ = A⁻¹
H⁻¹_uβ = -H_uu⁻¹ H_uβ A⁻¹
H⁻¹_βu = -A⁻¹ H_βu H_uu⁻¹
H⁻¹_uu = H_uu⁻¹ + H_uu⁻¹ H_uβ A⁻¹ H_βu H_uu⁻¹.
```

This is why a dense `A⁻¹` or Schur factor is enough for many exact traces.

For large `K`, exact traces need either:

1. selected solves against structured derivative matrices, or
2. stochastic trace estimation with fixed probes, or
3. a low-rank derivative representation.

No dense `H⁻¹` should be formed.

## 4. Topology Selection

### 4.1 Candidate Set

The requested discrete basis set is:

```text
T ∈ {periodic, flat, sphere, torus}.
```

Map these to implementation candidates:

```text
periodic -> cyclic 1D spline or circle topology
flat     -> Euclidean Duchon/Matern/thin-plate patch
sphere   -> spherical Wahba/Sobolev basis on S²
torus    -> mixed-periodicity Duchon with periodic axes
```

The existing Python helper currently defaults to Circle, Sphere, Torus,
Cylinder, and EuclideanPatch.  For this proposal, the exact requested set is
the four-way finite set above.

### 4.2 Evidence per Topology

For each topology `T`:

```text
1. Build Φ_T and penalties S_T.
2. Optimize `(β, u)` for fixed `(ρ, T)`.
3. Optimize `ρ` using analytic ∂V/∂ρ.
4. Evaluate `V(ρ_T*, T)`.
```

The scalar comparison is:

```text
score_T = V(ρ_T*, T).
```

Then:

```text
T* = argmax_T V(ρ_T*, T)
```

over:

```text
{periodic, flat, sphere, torus}.
```

### 4.3 Sign Discipline

The formula in this note is:

```text
V = F + 0.5 log|H| - 0.5 log|S_pen|+.
```

This is a negative log-evidence form when `F` is a negative penalized log
posterior.

If the selector uses `argmax`, then one of these must be true:

```text
public_score = -V
```

or:

```text
F = log posterior objective to maximize
```

The code must not mix these conventions.

A robust API should name the scalar explicitly:

```text
negative_log_evidence
```

or:

```text
log_evidence
```

Then `select_topology` ranks according to that name.

### 4.4 Comparable Fits

Evidence is comparable across `T` only when:

1. the response likelihood is the same;
2. the response transformation is the same;
3. observation weights are the same;
4. offsets are the same;
5. prior normalizations are included consistently;
6. basis dimensions are either fixed by the candidate definition or penalized
   through the same Laplace/Occam convention;
7. failures and singular fits are excluded, not silently given arbitrary
   fallback scores.

If a candidate topology changes the coordinate dimension, its `u` dimension
and penalty rank change.  That is acceptable only if the log-determinant and
prior-normalization terms are included consistently.

### 4.5 Candidate Construction Contract

The topology selector should materialize:

```text
enum TopologyCandidate {
    Periodic,
    Flat,
    Sphere,
    Torus,
}
```

Each candidate returns:

```text
basis_spec(T)
manifold_spec(T)
penalty_spec(T)
initial_u(T)
```

For `periodic`:

```text
manifold = S¹ or periodic interval
basis    = cyclic B-spline / periodic Duchon
```

For `flat`:

```text
manifold = Rᵈ
basis    = Euclidean Duchon/Matern/thin-plate
```

For `sphere`:

```text
manifold = S² embedded in R³
basis    = spherical Sobolev/Wahba
```

For `torus`:

```text
manifold = S¹ × S¹
basis    = mixed-periodicity Duchon
```

### 4.6 Selection Rule

The final rule is:

```text
T_hat = argmax_{T ∈ {periodic, flat, sphere, torus}} V(ρ_T*, T).
```

If two scores are numerically tied:

```text
|V(T_a) - V(T_b)| <= tolerance
```

break ties by the simpler topology:

```text
flat < periodic < sphere < torus
```

or expose the tie instead of forcing a winner.

No continuous interpolation between topologies is part of this contract.

## 5. Integration Contract

### 5.1 Existing Core Entry Point

The existing cached Newton entry point is:

```rust
pub fn solve_arrow_newton_step(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<(Array1<f64>, Array1<f64>, ArrowFactorCache), ArrowSchurError>
```

For the `u` notation in this document:

```text
ridge_t    = ridge_u
delta_t    = Δu
delta_beta = Δβ
```

This signature should remain the narrow solve primitive.

### 5.2 Proposed `arrow_log_det`

The log-determinant primitive should compute:

```text
log|H| = Σ_i log|H_uu_i| + log|A|
```

using the same row factors and Schur factor as the Newton path when possible.

Exact Rust signature:

```rust
pub fn arrow_log_det(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> Result<ArrowLogDet, ArrowSchurError>
```

Return type:

```rust
#[derive(Debug, Clone)]
pub struct ArrowLogDet {
    pub log_det_h: f64,
    pub log_det_huu: f64,
    pub log_det_schur: f64,
    pub factor_cache: ArrowFactorCache,
}
```

Contract:

1. `log_det_huu = Σ_i log|H_uu_i + ridge_t I|`.
2. `log_det_schur = log|A(ridge_t, ridge_beta)|`.
3. `log_det_h = log_det_huu + log_det_schur`.
4. For evidence, callers should pass zero ridges unless the ridge is part of
   the modeled objective.
5. `factor_cache` is returned so downstream IFT and gradient code reuses the
   same factors.

For `InexactPCG`, exact `log_det_schur` is unavailable unless a dense Schur
factor is also requested.  The function must return an error unless an
explicit stochastic or Lanczos estimator mode is added to the signature.

No silent approximation should occur.

### 5.3 Proposed `arrow_ift_warm_start`

This primitive predicts new `(β, u)` after a small `ρ` step.

Exact Rust signature:

```rust
pub fn arrow_ift_warm_start(
    cache: &ArrowFactorCache,
    schur_rhs_rho: ndarray::ArrayView2<'_, f64>,
    gu_rho: ndarray::ArrayView2<'_, f64>,
    delta_rho: ndarray::ArrayView1<'_, f64>,
) -> Result<ArrowIftWarmStart, ArrowSchurError>
```

Return type:

```rust
#[derive(Debug, Clone)]
pub struct ArrowIftWarmStart {
    pub delta_beta: Array1<f64>,
    pub delta_u: Array1<f64>,
    pub beta_rho: Array2<f64>,
    pub u_rho: Array2<f64>,
}
```

Shape contract:

```text
schur_rhs_rho : K × R, columns q_a = ∂g_red/∂ρ_a
gu_rho        : (N d) × R, columns G_{u,ρ_a}
delta_rho     : R
beta_rho      : K × R
u_rho         : (N d) × R
delta_beta    : K
delta_u       : N d
```

Math contract:

```text
β_ρ[:, a] = -A⁻¹ q_a
u_ρ[:, a] = -H_uu⁻¹ gu_rho[:, a] - H_uu⁻¹ H_uβ β_ρ[:, a]
δβ = β_ρ δρ
δu = u_ρ δρ
```

Operational contract:

1. Requires `cache.schur_factor.is_some()`.
2. Uses `cache.htt_factors_undamped` for `H_uu⁻¹`.
3. Does not use the damped row factors for IFT.
4. Does not rebuild the design.
5. Does not mutate the cache.

If large-`K` PCG mode is needed, add a separate matrix-free IFT signature that
accepts a Schur solve closure.  Do not overload this exact dense-factor
contract with hidden iterative behavior.

### 5.4 Proposed `arrow_evidence_grad`

This primitive evaluates the analytic `ρ` gradient of:

```text
V(ρ, T) =
F(β*, u*; ρ, T)
+ 0.5 log|H|
- 0.5 log|S_pen(ρ)|+.
```

Exact Rust signature:

```rust
pub fn arrow_evidence_grad(
    value_rho: ndarray::ArrayView1<'_, f64>,
    hessian_rho: &[ArrowHessianDerivative],
    schur_rhs_rho: ndarray::ArrayView2<'_, f64>,
    gu_rho: ndarray::ArrayView2<'_, f64>,
    penalty_logdet: &PenaltyLogdetDerivs,
    logdet: &ArrowLogDet,
) -> Result<ArrowEvidenceGradient, ArrowSchurError>
```

Supporting types:

```rust
#[derive(Debug, Clone)]
pub struct ArrowHessianDerivative {
    pub hbb: Array2<f64>,
    pub rows: Vec<ArrowRowHessianDerivative>,
}
```

```rust
#[derive(Debug, Clone)]
pub struct ArrowRowHessianDerivative {
    pub huu: Array2<f64>,
    pub huβ: Array2<f64>,
}
```

```rust
#[derive(Debug, Clone)]
pub struct ArrowEvidenceGradient {
    pub gradient: Array1<f64>,
    pub value_part: Array1<f64>,
    pub logdet_h_part: Array1<f64>,
    pub logdet_penalty_part: Array1<f64>,
    pub ift_beta: Array2<f64>,
    pub ift_u: Array2<f64>,
}
```

Contract:

1. `value_rho[a] = F_{ρ_a}` at the inner optimum.
2. `hessian_rho[a]` stores the direct derivative `H_{ρ_a}`.
3. `schur_rhs_rho[:, a] = q_a = ∂g_red/∂ρ_a`.
4. `gu_rho[:, a] = G_{u,ρ_a}`.
5. `penalty_logdet.first[a] = ∂ρ_a log|S_pen|+`.
6. `logdet.factor_cache` supplies row and Schur factors.
7. The function computes `ift_beta` and `ift_u` through `arrow_ift_warm_start`.
8. The direct trace uses arrow determinant derivatives.
9. Optional third-derivative movement terms require extending
   `ArrowHessianDerivative` to directional Hessian derivatives.

If exact `D_β H` and `D_u H` callbacks are not supplied, the first version
should explicitly implement the frozen-Hessian gradient:

```text
∂V/∂ρ_a =
F_{ρ_a}
+ 0.5 tr(H⁻¹ H_{ρ_a})
- 0.5 ∂ρ_a log|S_pen|+.
```

and name the mode:

```rust
pub enum ArrowEvidenceGradientMode {
    FrozenHessian,
    ExactLaplace,
}
```

A missing exact mode should be an error when `ExactLaplace` is requested.

### 5.5 Proposed Topology Selection Signature

For Rust-side topology selection:

```rust
pub fn select_topology_by_arrow_evidence(
    candidates: &[TopologyCandidate],
    options: &ArrowEvidenceOptions,
) -> Result<TopologySelection, ArrowEvidenceError>
```

Types:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyCandidate {
    Periodic,
    Flat,
    Sphere,
    Torus,
}
```

```rust
#[derive(Debug, Clone)]
pub struct TopologySelection {
    pub winner: TopologyCandidate,
    pub ranking: Vec<TopologyEvidence>,
}
```

```rust
#[derive(Debug, Clone)]
pub struct TopologyEvidence {
    pub topology: TopologyCandidate,
    pub rho: Array1<f64>,
    pub score: f64,
    pub converged: bool,
}
```

The selector should reject non-finite scores.

It should not install fallback topologies.

## 6. Numerical Pitfalls

### 6.1 Singular Row Blocks

`H_uu_i` can be singular when:

1. the latent coordinate is unidentifiable for row `i`;
2. the decoder derivative vanishes;
3. the manifold gauge is not fixed;
4. an analytic penalty contributes too little curvature;
5. the row has zero or near-zero working weight.

Mitigation:

1. use an isometry or ARD/gauge penalty where mathematically required;
2. add `ridge_u` for the Newton step;
3. keep undamped factors for IFT only after the true Hessian is nonsingular;
4. fail loudly if undamped IFT factors cannot be built.

### 6.2 Indefinite Schur Complement

The Schur complement can fail even when every row block factors.

Causes:

1. global β gauge modes;
2. insufficient β penalty rank;
3. collinear basis columns;
4. an invalid negative curvature contribution;
5. inconsistent row/block assembly signs.

Mitigation:

1. apply identifiability constraints before solving;
2. keep penalty nullspace accounting exact;
3. use `ridge_beta` only as solver damping;
4. report the failing factorization stage separately.

### 6.3 Pseudo-Logdet Rank Changes

The pseudo-logdet derivative assumes stable rank.

Rank can change when:

1. a smoothing parameter underflows;
2. a topology creates duplicate basis columns;
3. periodic endpoints are not collapsed;
4. a sphere or torus basis has redundant centers;
5. a penalty nullspace moves across the eigenvalue threshold.

Mitigation:

1. use structural nullity where available;
2. use fixed positive-eigenspace thresholds per evaluation;
3. exclude degenerate topology candidates;
4. do not compare evidence for candidates with accidental rank loss.

### 6.4 Ridges in Evidence

Ridges used for solver stability are not part of the statistical model.

Do not compute final evidence with:

```text
H + ridge I
```

unless the ridge corresponds to an explicit prior included in `F` and in
`S_pen`.

The accepted workflow is:

1. use ridge to find a stable step;
2. converge to the true penalized optimum;
3. evaluate evidence with zero solver ridge;
4. fail if the true Hessian is singular.

### 6.5 Inexact PCG and Log Determinants

PCG solves do not provide log determinants.

Therefore:

1. `solve_arrow_newton_step` can use `InexactPCG`;
2. `arrow_log_det` cannot return exact `log|A|` from PCG alone;
3. evidence code must either request dense Schur factorization or an explicit
   logdet estimator;
4. estimators must expose their stochastic error and fixed probe seed.

No silent conversion from exact evidence to approximate evidence should occur.

### 6.6 Third Derivatives

Exact Laplace gradients require derivatives of the Hessian along IFT
directions:

```text
D_β H [β_ρ]
D_u H [u_ρ]
```

These are third derivatives of `F`.

If the family only exposes Gauss-Newton Hessian blocks, the exact derivative
is not available.

The implementation must distinguish:

```text
FrozenHessian evidence gradient
```

from:

```text
ExactLaplace evidence gradient
```

They are not the same quantity.

### 6.7 Moving Basis Parameters

When `ρ` moves basis geometry, not only penalty weights:

```text
Φ = Φ(u; ρ, T)
S_pen = S_pen(ρ, T)
```

then `H_{ρ_a}`, `G_{u,ρ_a}`, and `G_{β,ρ_a}` receive design-derivative
terms.

Ignoring those terms biases topology selection toward bases whose geometry is
less exposed to the derivative path.

### 6.8 Manifold Retraction

For sphere and torus candidates, `u` updates live in tangent coordinates but
must be applied through a retraction:

```text
u_i_new = Retr_{u_i}(Δu_i).
```

The Schur algebra operates in the tangent block.

The evidence Hessian must correspond to the same local chart used by the
stationarity equations.

Mixing ambient Euclidean updates with manifold-constrained Hessians produces
incorrect curvature and invalid log determinants.

### 6.9 Sign Errors

The most common implementation error is flipping the reduced RHS sign.

The correct convention for minimizing `F` is:

```text
S Δβ = -g_β + H_βu H_uu⁻¹ g_u.
```

and:

```text
Δu_i = -H_uu_i⁻¹(g_u_i + H_uβ_i Δβ).
```

A simple dense reference check should compare the arrow step against the full
bordered solve for small `(N, d, K)`.

### 6.10 Symmetry Drift

Floating-point row reductions can produce:

```text
S ≠ Sᵀ
```

by tiny amounts.

Before Cholesky:

```text
S ← (S + Sᵀ)/2.
```

This is not a fallback; it is enforcing the mathematical symmetry of the
assembled Hessian.

### 6.11 Topology Candidate Failures

A topology candidate may fail because its basis is invalid for the supplied
data.

Examples:

1. sphere input not on or near `S²`;
2. torus periods missing or invalid;
3. periodic endpoints duplicated;
4. flat basis centers too few for the nullspace;
5. candidate dimension incompatible with features.

Selection should record:

```text
candidate excluded: reason
```

and rank only finite valid evidence values.

It should not silently replace a failed candidate with a flat basis.

### 6.12 Dead Terms and Compatibility

The implementation should not keep unused compatibility paths.

If topology selection uses:

```text
{periodic, flat, sphere, torus}
```

then the Rust topology enum for this feature should expose those four values
and no unused candidate variant.

Additional public Python candidates can remain in Python if they are part of a
separate existing API, but the new arrow-evidence selector should keep its
contract narrow.

## 7. Summary Formula Sheet

Schur complement:

```text
S = H_ββ - H_βu H_uu⁻¹ H_uβ
```

Reduced RHS:

```text
r = -g_β + H_βu H_uu⁻¹ g_u
```

Reduced solve:

```text
S Δβ = r
```

Back-substitution:

```text
Δu_i = -H_uu_i⁻¹(g_u_i + H_uβ_i Δβ)
```

IFT local β sensitivity:

```text
∂u*/∂β = -H_uu⁻¹ H_uβ
```

IFT β sensitivity:

```text
∂β*/∂ρ = -S⁻¹ ∂g_red/∂ρ
```

IFT full u sensitivity:

```text
∂u*/∂ρ =
-H_uu⁻¹ G_{u,ρ}
-H_uu⁻¹ H_uβ ∂β*/∂ρ
```

Arrow determinant:

```text
log|H| = Σ_i log|H_uu_i| + log|S|
```

Laplace evidence:

```text
V(ρ, T) =
F(β*, u*; ρ, T)
+ 0.5 log|H|
- 0.5 log|S_pen(ρ)|+
```

Analytic gradient:

```text
∂V/∂ρ_a =
F_{ρ_a}
+ 0.5 tr(H⁻¹ dH/dρ_a total)
- 0.5 tr(S_pen⁺ S_{pen,ρ_a})
```

Topology selection:

```text
T_hat = argmax_{T ∈ {periodic, flat, sphere, torus}} V(ρ_T*, T).
```
