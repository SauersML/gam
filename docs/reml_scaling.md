# Scaling `gamfit.torch.fit` — joint vs independent additive REML

This page documents the `mode` argument to `gamfit.torch.fit()` and
when to use which.

## TL;DR

| Use case | F (smooths) | D (outputs) | recommended mode |
| --- | --- | --- | --- |
| Tabular GAM, formula-style | 1–10 | 1 | `"joint"` (or just use `gamfit.fit(formula=...)`) |
| Multi-smooth small | ≤ 64 | 1 | `"joint"` |
| Multi-smooth medium | ≤ 64 | > 1 (multi-output) | `"independent"` |
| SAE / large-scale | > 64 | any | `"independent"` |
| Don't know | any | any | `"auto"` (default) |

`mode="auto"` picks `"joint"` if `F ≤ 64` and `D == 1`, else `"independent"`.

## Why two modes

### Joint additive REML
Assembles `Z = [diag(a_1) X_1 | ... | diag(a_F) X_F]` and Choleskies the
joint `(F·M_k) × (F·M_k)` inner Hessian. Per-smooth `λ_k` are jointly
selected by REML's outer Newton iteration with the F×F outer Hessian
factored together. Statistically the most efficient — atoms see each
other through the joint design.

Cost: `O((F · M_k)³)` per inner Cholesky, `O(F³)` per outer Newton step.
Feasible for `F ≲ 64`. Infeasible past `F ≳ 1000` — the inner Hessian
becomes a `(F · M_k)²` dense matrix.

### Per-atom independent REML
For each smooth k, run a separate single-smooth REML fit. Each atom gets
its own `λ_k` chosen independently of the others. Per-atom fitted
contributions are summed.

Cost: `O(F · M_k³)` — linear in F. Multi-output `D > 1` supported natively
because the underlying single-smooth REML primitive supports it.

Mathematical caveat: per-atom λ_k are treated as independent. Under
**TopK sparse gating in an SAE** — where most atoms are zero per row, so
per-atom designs are effectively orthogonal in expectation — this is the
*correct* algorithm. For genuinely overlapping smooths over the same
predictor space (e.g. a multi-term GAM on the same covariate), the joint
fit is statistically more efficient, but only computable at moderate F.

### Auto threshold

`mode="auto"` routes to `"joint"` when both:

- `F ≤ 64` (the `_AUTO_MODE_F_THRESHOLD` constant), and
- `D == 1` (the multi-block backward currently rejects D > 1).

Otherwise routes to `"independent"`. The threshold is conservative —
joint is feasible up to ~F=256 in many setups, but the user benefits
more from independence (and from multi-output support) past F=64.

## Production use: SAE training at F=100K

The Manifold-SAE production training path uses `mode="independent"`
unconditionally. The architecture is:

```python
positions = encoder(x)           # (B, F)
amps = topk_amps(encoder(x))     # (B, F), sparse
result = gamfit.torch.fit(
    points=[positions[:, k:k+1].unsqueeze(-1) for k in range(F)],
    response=x_centered,          # (B, D)
    smooths=[
        Duchon(centers=self.centers, m=2, by=amps[:, k])
        for k in range(F)
    ],
    mode="independent",
)
recon = result.fitted + b_dec    # (B, D)
loss = ((recon - x) ** 2).mean()
loss.backward()                  # autograd flows back to encoder
```

Autograd is preserved through every per-atom REML's analytic VJP. The
encoder gradient is the sum of F single-smooth backward contributions
through TopK-gated per-atom designs.

## Empirical timing (CPU, MacOS arm64, single process)

These numbers are from a synthetic benchmark on the developer machine;
expect cluster CPUs to be 2-5x faster.

| F     | D    | N    | mode          | dt (s)  |
| ---   | ---  | ---  | ---           | ---     |
| 8     | 16   | 64   | independent   | 0.01    |
| 80    | 1    | 100  | auto → independent | 0.01 |
| 1024  | 64   | 256  | independent   | 0.35    |
| 4096  | 64   | 256  | independent   | 1.36    |

Cost scales linearly in F (as predicted by O(F · M_k³)).

## Future work: sparse joint REML

The truly right algorithm at F=100K with TopK gating is **sparse joint
REML** — under K=8 active per token with F=100K, the joint inner Hessian
is 99.99% block-diagonal in expectation. A sparse Cholesky on this
structure would give back the per-smooth-λ joint statistics (improving
over independent's per-atom independence assumption) while remaining
linear in F.

gamfit's REML driver currently uses a dense Cholesky and doesn't exploit
the sparsity. Adding sparse-aware joint REML is a future workstream;
until then, `mode="independent"` is the right answer at SAE scale.

## Non-Gaussian LAML

Non-Gaussian families are not currently wired in `gamfit.torch`. Only
the Gaussian-identity closed-form REML primitives have forward and
backward support. Non-Gaussian families (Binomial/Poisson/Gamma) and
the corresponding analytic backward (IRLS curvature derivative terms
`t = ℓ_ηηη`, `κ = ℓ_ηηηη`, c-vector per the math memo) are deferred.
