# Scaling `gamfit.torch.fit` — joint vs shared-scale block REML

This page documents the `mode` argument to `gamfit.torch.fit()` and
when to use which.

## TL;DR

| Use case | F (smooths) | D (outputs) | recommended mode |
| --- | --- | --- | --- |
| Tabular GAM, formula-style | 1–10 | 1 | `"joint"` (or just use `gamfit.fit(data, formula=...)`) |
| Multi-smooth small | ≤ 64 | 1 | `"joint"` |
| Multi-smooth medium | ≤ 64 | > 1 (multi-output) | `"independent"` |
| SAE / large-scale | > 64 | any | `"independent"` |
| Don't know | any | any | `"auto"` (default) |

`mode="auto"` picks `"joint"` if `F ≤ 64` and `D == 1`, else `"independent"`.

## Why two modes

### Joint additive REML
Assembles `Z = [diag(a_1) X_1 | ... | diag(a_F) X_F]` and Choleskies the
joint `(Σ_k M_k) × (Σ_k M_k)` inner Hessian. Per-smooth `λ_k` are
jointly selected by REML's outer Newton iteration with the F×F outer
Hessian factored together. Statistically the most efficient — atoms see
each other through the joint design.

Cost: `O((Σ_k M_k)³)` per inner Cholesky, `O(F³)` per outer Newton step.
Feasible for `F ≲ 64`. Infeasible past `F ≳ 1000` — the inner Hessian
becomes a `(Σ_k M_k)²` dense matrix.

### Shared-scale block-orthogonal REML
For each smooth k, build the by-modulated block and solve its coefficient
problem locally, but select the `λ_k` values against one additive residual
quadratic shared across all blocks:

```text
q_d = y_d' W y_d - sum_k b_{kd}' K_k^{-1} b_{kd}
```

This is the exact additive REML objective when the modulated block designs
are W-orthogonal. It fixes the old private-scale independent loop, where
each atom fit the full response with its own residual quadratic and the
scores were summed.

Cost: `O(Σ_k M_k³)` — linear in F for fixed per-smooth width. Multi-output
`D > 1` is supported with one λ per smooth and one profiled residual scale
per output column.

Mathematical caveat: this path assumes the realized by-modulated block
cross-Grams are negligible. Under **TopK sparse gating in an SAE** this is
often a good approximation, and when supports are disjoint it is exact. For
genuinely overlapping smooths over the same predictor space (e.g. a
multi-term GAM on the same covariate), the joint fit is statistically more
efficient, but only computable at moderate F.

### Auto threshold

`mode="auto"` routes to `"joint"` when both:

- `F ≤ 64` (the `_AUTO_MODE_F_THRESHOLD` constant), and
- `D == 1` (the multi-block backward currently rejects D > 1).

Otherwise routes to `"independent"`, whose implementation is the
shared-scale block-orthogonal estimator. The threshold is conservative:
joint is feasible up to ~F=256 in many setups, but the user benefits more
from linear block scaling and multi-output support past F=64.

## Large-F use: sparse-atom training at F=100K

At very large F with TopK-gated per-atom designs, `mode="independent"`
uses the shared-scale block estimator: the coefficient problems decouple,
the residual scale remains global, autograd flows through the differentiable
block solves, and memory stays linear in F. A representative architecture is:

```python
codes = encoder(x)               # (B, F)
positions = codes                # (B, F)
amps = topk_amps(codes)          # (B, F), sparse
result = gamfit.torch.fit(
    points=[positions[:, k:k+1] for k in range(F)],
    response=x_centered,          # (B, D)
    smooths=[
        Duchon(centers=self.centers, m=2, by=amps[:, k])
        for k in range(F)
    ],
    mode="independent",
)
recon = result.fitted + b_dec    # (B, D)
loss = ((recon - x) ** 2).mean()
loss.backward()                  # gradient flows through TopK amplitudes
```

Autograd is preserved through the shared-scale block solves to the modulated
designs and differentiable inputs. With the current Duchon torch basis,
`points` are structural (the basis is forward-only with respect to
positions), so encoder gradients in the example above flow through the TopK
`by` amplitudes; use a differentiable basis path if you need coordinate
gradients.

## Future work: sparse joint REML

The next algorithmic step at F=100K with real coactivation overlap is
**coactivation-graph joint REML**. Under K=8 active per token with F=100K,
the joint inner Hessian is sparse in the active-pair graph. Sparse/block
Cholesky on that structure would recover joint competition inside
coactivation components while keeping the profiled residual scale global.

gamfit's exact joint Torch REML driver currently uses a dense Cholesky and
doesn't exploit this sparsity. Until sparse-aware joint REML lands,
`mode="independent"` is the scalable shared-scale orthogonal approximation.

## Non-Gaussian LAML

Non-Gaussian families are not currently wired in `gamfit.torch`. Only
the Gaussian-identity closed-form REML primitives have forward and
backward support. Non-Gaussian families (Binomial/Poisson/Gamma) and
the corresponding analytic backward (IRLS curvature derivative terms
`t = ℓ_ηηη`, `κ = ℓ_ηηηη`, c-vector per the math memo) are deferred.
