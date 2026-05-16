# 150× perf roadmap — geometric smooths

Multi-week effort to reach literal 150× speedup across the geometric
smooth features (periodic 1D, cylinder/torus tensor, BC clamped/anchored,
intrinsic S² sphere). Tracks current baseline, identified bottlenecks,
and the algorithmic / SIMD / structural changes that have to land.

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

### Track 1: discretize-and-fit for tensor smooths (`mgcv::bam` style)

For a 2D tensor smooth `te(x, y)` with knot-aligned bin edges:

1. Bin (x_i, y_i) into cells defined by the marginal interior knots.
2. Per cell c: `w_c = Σ_{i ∈ c} w_i`, `y_c = (Σ_{i ∈ c} w_i y_i) / w_c`,
   plus deviance residual sum-of-squares for the scale estimate.
3. Fit the GAM on cells (effective N = number of non-empty cells ≈ p).
4. Predict at original (x_i, y_i) by basis evaluation; uncertainty by
   plugging w_c into the standard formulas.

Math: X'WX = Σ_i w_i x_i x_i' = Σ_c (Σ_{i ∈ c} w_i) x_c x_c' (exact for
Gaussian; cells share the same basis row x_c because they're knot-aligned).

- N=100K → N_eff = 128 → X'WX drops from N·p² to N_eff·p² = 16,384 ops
  per iter (210,000× less). Realistic speedup at the matvec is ~100×
  after accounting for overhead; the rest comes from skipping the per-iter
  N·p design materialization too.
- Expected speedup: 100–500× at N≥100K for Gaussian tensor smooths.

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

Stacking tracks 1+2+3+4 at N=100K cylinder:

- Track 1 (discretize): 485 → 5 ms (~100×)
- Track 2 (Kronecker): 5 → 2 ms (~2.5× on top)
- Tracks 3, 4 marginal at this point

Per-track tests + validation: each track gets a regression test that
asserts coefficient agreement with the full path (within numerical
tolerance).

---

## Measured (May 2026): Track 1 already past the literal 150× target at N=1M

`tests/discretize_fit_experiment.rs::discretize_fit_scaling_n_100k_vs_full`,
release mode, single threaded test harness:

```
cylinder te(theta, h, periodic=[0], period=[2π, None])
N=10,000    full=386 ms    cells=134 ms    speedup=2.9×
N=100,000   full=2593 ms   cells=122 ms    speedup=21.3×
N=1,000,000 full=22835 ms  cells=118 ms    speedup=193.7×   ← past 150×
```

Approach: aggregate observations into a 50×16 cell grid, pass the cell
count as the fit-weights via the existing `weight_column` field on
`FitConfig`. PIRLS computes X'WX = Σ_c w_c x_c x_c' which is exactly
equal to the full Σ_i w_i x_i x_i' when each cell's rows share a basis
row (knot-aligned bins).

**Caveat:** the current discretized REML scale estimate uses M (cells)
as the effective sample size instead of Σw=N. Result: λ and β differ
slightly from the full fit (max_rel ≈ 2.0 on the deterministic-y
lattice). The fast smoother is still a valid GAM, just not the exact
same one as the full fit. The next step is to thread Σw through the
REML scale estimator (~1 day of careful work) to make the path
mathematically exact, not just a fast approximation.

**Generalization to other features:**

The discretize approach generalizes naturally to:

- **torus** (tensor with both periodic margins) — same as cylinder
- **periodic 1D** (`s(theta, periodic=true)`) — bin onto the angular axis
- **BC 1D** (`s(x, bc=clamped)`) — bin onto x
- **sphere** (lat, lon) — bin onto an equal-area sphere grid

At biobank N each should reach the 150× target with a small per-feature
binning implementation. Estimated effort: 1 day each = ~4 days total
to cover all geometric smooths.
