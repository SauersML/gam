# 150× perf roadmap — geometric smooths

Multi-week effort to reach literal 150× speedup across the geometric
smooth features (periodic 1D, cylinder/torus tensor, BC clamped/anchored,
intrinsic S² sphere). Tracks current baseline, identified bottlenecks,
and the algorithmic / SIMD / structural changes that have to land.

## Status (May 2026)

| Track                                             | State                |
| ------------------------------------------------- | -------------------- |
| Parallel basis builds (periodic / sphere / BC)    | **done** (rayon)     |
| Track 3 — SIMD `wahba_sphere_kernel_from_cos`     | **done**             |
| GPU fast-paths for PIRLS XtWX / Cholesky / SpMV   | **done** (cuBLAS, cuSOLVER, cuSPARSE) |
| Discretize-and-fit (rejected, see end of doc)     | not landing          |
| Track 0 — analytic closed-form Gaussian REML      | **TODO** (headline win, ~1 week) |
| Track 2 — Kronecker matvec in PIRLS               | TODO                 |
| Track 4 — stochastic / subsampled REML            | TODO                 |

Track 3 landed in commits `049965af` and `98baab92`; the GPU
fast-paths landed in `29b066fc` (XtWX), `73f8bc6b` (Cholesky), and
`30386254` (sparse mat-vec). Parallel basis fills landed across
`794d424e`, `a8d9e763`, `4fe84de5`, `56ba568f`. The
[manifold smooths gallery](manifold-smooths.md) is a self-validating
end-to-end demo of the geometric features the roadmap targets.

## Current baseline (release, serial, warm — May 2026)

Basis builds (already ≥ parallel, near-linear in N):

| feature                          | N=100K   | scaling vs N |
| -------------------------------- | -------- | ------------ |
| `create_periodic_bspline_basis`  |  2.88 ms | linear       |
| `exact_periodic_cubic`           |  8.73 ms | linear       |
| `build_spherical_harmonic_basis` |  4.50 ms | linear       |
| `spherical_wahba_kernel_matrix`  |       —  | N²·K (centers) |
| `bc_bspline` (N=10K)             |  1.41 ms | linear       |

For the basis-build side at biobank N, **we are already past 150×** vs.
the pre-parallelization serial baseline. The next 150× has to come from
the FIT path.

Full GAM fits at N=10K cylinder (`te(theta, h, periodic=[0], ...)`,
p=128):

| stage                       | time     |
| --------------------------- | -------- |
| dataset                     |    8 ms  |
| materialize (formula → spec)|    1 ms  |
| full `fit_from_formula`     |  485 ms  |
| └ outer REML iters          |    8     |
| └ ≈per-iter cost            |   60 ms  |

Scaling: N=10K → N=100K → 485 ms → 3605 ms (≈7.4× for 10× more rows;
sub-linear, dominated by per-iteration fixed cost of REML + PIRLS).

**150× targets:**

- N=10K cylinder fit: 485 ms → **3.2 ms**
- N=100K cylinder fit: 3.6 s → **24 ms**

These are not reachable by per-line micro-optimization. They require
algorithmic restructuring.

## Bottlenecks (FIT path)

For Gaussian cylinder at N=100K, p=128:

- PIRLS X'WX per iter:  N·p² = 1.6 G ops  (BLAS, ~50 ms on 8 cores)
- Cholesky on p×p:       p³/6 = 0.3 M ops (<1 ms)
- REML outer evaluations: O(8) per fit
- Per-iter linalg dominates; outer count ~8 for Gaussian (single-seed).

Cost model: total ≈ 8 outer × (X'WX 50 ms + Cholesky 0.5 ms + glue)
                 ≈ 420 ms (matches measurement).

## Roadmap to 150×

### Track 0 (preferred): analytic closed-form Gaussian REML

For Gaussian (identity-link) fits — the most common case in geometric
smooths — REML can be solved analytically without PIRLS iteration:

1. One-time decomposition: simultaneously diagonalize `X'X` and the
   penalty `S` via the generalized eigenproblem `X'X v = γ S v` (or, in
   the multi-penalty case, sequential transformation following Wood
   §6.6.2). Cost: `O(p³)` once.
2. In the diagonalized basis, the REML score becomes a closed-form sum:
   `REML(λ) = − 0.5 log|S(λ) + X'X| + log|S(λ)|/2 − 0.5 RSS(λ) / σ²(λ)`.
   Each evaluation is `O(p)`.
3. Optimize over λ by 1D (or D-dim) BFGS / Brent: ~10 evals.
4. Final β = (X'X + S(λ_opt))⁻¹ X'y via the same diagonalization.

Cost breakdown at biobank N=1M, p=128, D smoothing params:

- X'WX assembly:   `O(N·p²)` once = 128 ms (BLAS, 8 cores)
- Eigendecomp:    `O(p³)` once = ~2 ms
- REML λ search:  ~10 × `O(p)` = sub-ms
- Total fit:      **~130–150 ms** (vs. PIRLS at ~23 s)
- Speedup:        **~150×** with no accuracy penalty (mathematically
                  identical to the full PIRLS fit at convergence)

This is the principled path: same answer as PIRLS, far less computation.

Estimate: 1 week to implement + validate against PIRLS coefficient
agreement on a wide test sweep.

Affected modules:
- `src/solver/estimate.rs` — Gaussian-direct entry that bypasses PIRLS
- `src/solver/reml/unified.rs` — closed-form REML score + gradient
- `src/terms/smooth.rs` — auto-detect when Gaussian-direct is safe

### Track 1 (rejected): discretize-and-fit

For a 2D tensor smooth `te(x, y)` with knot-aligned bin edges:

1. Bin (x_i, y_i) into cells defined by the marginal interior knots.
2. Per cell c: `w_c = Σ_{i ∈ c} w_i`, `y_c = (Σ_{i ∈ c} w_i y_i) / w_c`,
   plus deviance residual sum-of-squares for the scale estimate.
3. Fit the GAM on cells (effective N = number of non-empty cells ≈ p).
4. Predict at original (x_i, y_i) by basis evaluation; uncertainty by
   plugging w_c into the standard formulas.

The tempting linear-algebra identity is:

`X'WX = Σ_i w_i x_i x_i' = Σ_c (Σ_{i ∈ c} w_i) x_c x_c'`

That identity is not enough for this solver path. The current
`weight_column` field is not a frequency-weight contract for Gaussian REML:
the profiled scale and REML derivatives use the physical row count in the
weighted dataset. A full duplicate-row fit sees `n = N`; a coalesced cell fit
sees `n = M`. Even when duplicate observations share an exact basis row, the
two paths optimize different REML criteria and select different smoothing
parameters.

- N=100K → N_eff = 128 → X'WX drops from N·p² to N_eff·p² = 16,384 ops
  per iter (210,000× less). Realistic speedup at the matvec is ~100×
  after accounting for overhead; the rest comes from skipping the per-iter
  N·p design materialization too.
- Expected speedup: 100–500× at N≥100K for Gaussian tensor smooths.

This track should not be implemented as automatic discretization unless the
solver grows an explicit frequency-weight likelihood with a row-count contract
threaded through the REML objective, derivatives, diagnostics, and prediction
metadata. Until then it is a different model, not a numerically noisy version
of the full fit.

Estimate: 3–5 days work.

Affected modules:
- `src/solver/pirls.rs` — discrete path that consumes pre-aggregated stats
- `src/terms/smooth.rs` — `te()` design-build with cell-aggregation option
- `src/solver/workflow.rs` — auto-detect when bam path is safe

Tests required:
- Coefficient agreement vs. full fit at N=1K (where both are fast)
- Edf, lambda, REML score agreement
- Confidence interval coverage

### Track 2: Khatri-Rao X·β in PIRLS

When the design is a Kronecker product X = X1 ⊗_R X2 (the existing
`kronecker_factored` metadata flag), compute X·β by reshaping:

  X β = vec(X1 · reshape(β, K1, K2) · X2')

Cost per matvec: N·K1 + N·K2  vs.  N·K1·K2 = N·p
For K1=K2=12: 24N vs 144N (6× per matvec).
Plus memory: store X1 (N·K1) and X2 (N·K2) instead of materialized X
(N·K1·K2): 12× memory reduction at K1=K2=12.

PIRLS needs three primitives: `X·β`, `X'·r`, `X'·diag(w)·X`. All three
benefit from the Kronecker structure.

Estimate: 1 week.

Affected modules:
- `src/matrix.rs` — `DesignMatrix::Kronecker` variant
- `src/solver/pirls.rs` — Kronecker-aware matvec
- `src/terms/smooth.rs` — return Kronecker design when applicable

### Track 3: SIMD `wahba_sphere_kernel_from_cos`

The kernel evaluates `(1.0 + 1.0/sqrt(w)).ln()` per (i,j). At N=K=4096
that's 16M sqrt+ln per kernel build. Vectorize over centers using
`wide::f64x4` (already in use elsewhere in `basis.rs`):

- Replace `.sqrt()` with `wide::f64x4`'s SIMD sqrt (1 cycle on NEON).
- Replace `.ln()` with a 4-element polynomial approximation
  (Estrin form, 8 mul + 7 add, < 1 ulp over [1, 2]).
- Manual range reduction via Frexp; combine.

Expected speedup: 5–10× on the inner body.

Estimate: 2 days. Numerical accuracy validated against the scalar
implementation across the full domain.

## Track 4: stochastic / subsampled REML for biobank N

For N ≫ p, the smoothing parameter λ is determined to high precision by
a random subsample of size ~10p log p. Strategy:

- During REML outer iterations, fit λ on a subsample of size m = 5p².
- At convergence, do ONE final PIRLS on the full N with the converged λ.

Cost: K_outer · m · p² + 1 · N · p² ≈ N · p² · (1 + K_outer · m / N).
For K_outer = 8, m = 80K (at p = 128), N = 1M: 1 + 0.64 = 1.64×.
Equivalent to ~5× speedup for the REML outer loop alone at biobank N.

Estimate: 1 week. Risk: subsample noise in REML score affecting
convergence — needs care.

## Combined yield

Stacking the viable tracks at N=100K cylinder:

- Track 2 (Kronecker): 485 → lower constant-factor PIRLS cost
- Tracks 3, 4 marginal at this point

Per-track tests + validation: each track gets a regression test that
asserts coefficient agreement with the full path (within numerical
tolerance).

---

## Rejected (May 2026): weighted-cell discretization

The weighted-cell benchmark was removed because it measured a faster but
different model. On an exact 50×16 duplicate-coordinate lattice, the
coalesced weighted-cell path still disagreed with the full fit at strict
coefficient tolerance. Root cause: Gaussian profiled REML uses the dataset
row count, so the cell fit uses `M` cells where the full fit uses `N`
observations. This is a semantics bug in the proposed optimization, not a
tolerance problem.
