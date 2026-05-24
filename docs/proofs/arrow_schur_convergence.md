# Arrow-Schur Inner Newton Convergence

## Claim Audited

Let

```text
F(t, beta; rho) =
    ell(Phi(t) beta) + P_beta(beta; rho) + P_t(t; rho)
```

be the composed inner objective for the `(beta, psi, rho)` engine after fixing
`rho`, with per-row latent coordinates `t = (t_1, ..., t_N)`. The arrow-Schur
inner iteration forms the bordered Newton system

```text
[ H_tt   H_t beta ] [ d_t    ] = -[ g_t    ]
[ H_beta t H_bb   ] [ d_beta ]    [ g_beta ]
```

where `H_tt` is row-block diagonal, eliminates `d_t`, solves the Schur system
for `d_beta`, back-substitutes `d_t`, and applies the full step
`(t, beta) <- (t, beta) + (d_t, d_beta)`.

The proposed convergence statement was:

> If the latent block is compact, the analytic penalty gradients are Lipschitz,
> and the Schur complement has bounded condition number, then the arrow-Schur
> inner Newton iteration converges to a critical point even when `P_t` includes
> non-convex analytic penalties such as `BlockOrthogonality`,
> `MechanismSparsity`, `IvaeRidgeMeanGauge`, or row-precision regions with
> non-positive-definite `Lambda`.

This statement is false for the undamped full-step iteration.

## Counterexample

Take the smallest arrow system:

```text
N = 1, d = 1, K = 0.
```

There is no shared `beta` block and no Schur solve. The arrow-Schur step is
ordinary scalar Newton on the latent coordinate. Define the analytic objective

```text
F(t) = 1/4 t^4 - t^2 + 2t,
g(t) = F'(t) = t^3 - 2t + 2,
H(t) = F''(t) = 3t^2 - 2.
```

This is a polynomial penalty term, hence analytic. On the compact latent set
`C = [0, 1]`, `g` is Lipschitz because `H` is continuous and bounded:

```text
|H(t)| <= 2,  t in [0, 1].
```

The `K = 0` Schur complement is the empty matrix, so its condition number is
vacuously one. The Newton map is

```text
N(t) = t - g(t) / H(t).
```

At the two iterates

```text
g(0) = 2,     H(0) = -2,  N(0) = 1,
g(1) = 1,     H(1) = 1,   N(1) = 0.
```

Thus the full-step inner Newton iteration cycles:

```text
0 -> 1 -> 0 -> 1 -> ...
```

Neither point is critical, since `g(0) = 2` and `g(1) = 1`. The sequence is
bounded in the compact latent block and satisfies the stated Lipschitz and
Schur-conditioning assumptions, yet it does not converge to a critical point.

The failure is not caused by Schur algebra. Schur elimination is an exact block
factorization when the eliminated row blocks are invertible; it preserves the
Newton direction. The missing assumption is a globalization mechanism that
forces descent for the actual nonlinear objective.

## Correction

Use adaptive proximal Levenberg damping and Armijo acceptance. At an iterate
`x = (t, beta)`, choose `mu >= 0` and solve the arrow system with

```text
H_mu = H + mu I.
```

The candidate step `p_mu` must satisfy

```text
F(x + p_mu) <= F(x) + c1 grad F(x)^T p_mu,
0 < c1 < 1.
```

If the row block or Schur complement is not positive definite, or if the
Armijo inequality fails, increase `mu <- gamma mu` with `gamma > 1` and retry.

For sufficiently large `mu`,

```text
p_mu = -(H + mu I)^(-1) grad F(x)
     = -mu^(-1) grad F(x) + O(mu^(-2) ||H|| ||grad F(x)||).
```

Therefore

```text
grad F(x)^T p_mu
  = -mu^(-1) ||grad F(x)||^2
    + O(mu^(-2) ||H|| ||grad F(x)||^2).
```

When `mu > 2 ||H||`, this gives

```text
grad F(x)^T p_mu <= -1/(2mu) ||grad F(x)||^2.
```

If `grad F` is Lipschitz with constant `L` on the compact level set, the
descent lemma gives

```text
F(x + p_mu)
  <= F(x) + grad F(x)^T p_mu + L/2 ||p_mu||^2.
```

Since `||p_mu|| <= 2 ||grad F(x)|| / mu` for `mu > 2 ||H||`, the last term is
at most

```text
2L ||grad F(x)||^2 / mu^2.
```

For large enough `mu`, this is no larger than
`(1 - c1)(-grad F(x)^T p_mu)`, so Armijo acceptance occurs after finitely many
damping increases whenever `grad F(x) != 0`.

## Convergence Statement That Survives

Assume:

1. The accepted iterates remain in a compact level set.
2. `F` is continuously differentiable and `grad F` is Lipschitz on that level
   set.
3. The arrow row blocks and Schur complement are solved exactly after adding
   the accepted proximal shift `mu I`, or the inexact solve satisfies the
   standard relative residual forcing condition.
4. `F` is bounded below on the level set.
5. The adaptive damping loop uses `mu <- gamma mu`, `gamma > 1`, and accepts
   only Armijo steps with `0 < c1 < 1`.

Then every accumulation point of the accepted iterates is first-order critical,
and

```text
liminf_k ||grad F(x_k)|| = 0.
```

With the additional Kurdyka-Lojasiewicz property, which holds for real
analytic objectives on compact semianalytic sets, the whole accepted sequence
converges to one critical point.

The longest open derivation is the KL upgrade from subsequential stationarity
to full-sequence convergence for the product latent manifold with retractions,
because the implementation mixes Euclidean beta coordinates, possible
Riemannian latent retractions, and analytic penalties that may be only
operator-represented in the solver. The missing lemma is:

> The retracted arrow-Schur Armijo iteration for the implemented product
> latent manifolds is a descent method for a definable analytic objective with
> a relative-error bound between the accepted proximal direction and the
> Riemannian gradient mapping.

