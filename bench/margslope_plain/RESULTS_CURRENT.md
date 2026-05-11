# Plain marginal-slope (no scale_dims) performance — current state

Snapshot date: 2026-05-11. Binary: `target/release/gam` built from main.

Synthetic data: `case ~ probit + sex + duchon(PC1..PC10, centers=40,
order=1, power=2, length_scale=1)` with `logslope =` same Duchon term.

## Plain path (no `--scale-dimensions`)

| n      | wall | fit-phase elapsed | outer_iter | converged |
| -----: | ---: | ----------------: | ---------: | --------- |
|    100 |    ? |              ~5 s |        100 | **fail**¹ |
|    200 |  5 s |             4.9 s |         60 | false²    |
|    300 |  1 s |             0.8 s |          ? | true      |
|    500 |  1 s |             1.1 s |          ? | true      |
|    750 |  2 s |             1.4 s |          ? | true      |
|   1000 |  2 s |             1.9 s |          9 | true      |
|   1500 |  2 s |             2.2 s |          ? | true      |
|   2000 | 17 s |            16.7 s |         12 | true      |
|   3000 |  4 s |             4.1 s |          ? | true      |
|   5000 | 31 s |            31.2 s |         13 | true      |
|  10000 | 38 s |            37.7 s |         11 | true      |
|  20000 | 18 s |            17.9 s |          ? | true      |
|  50000 |285 s |           285.5 s |         11 | true      |

¹ N=100: outer rejects all seeds because inner PIRLS spins 100 cycles
  with `beta_inf=3.4e4, residual=3.3e-4 > tol=5.8e-6` — symptomatic of
  the joint Hessian being near-singular. With 100 rows and dim=81 the
  data Hessian X^T W X + S(λ) is ill-conditioned along whichever
  direction the seed leaves under-penalised, so the Newton step grows
  large and the trust-region cuts it down. PIRLS makes only ~1% per
  cycle progress along the slow direction. **Open root cause**: the
  Operator-mode joint hessian skips the eigenvalue-based ridge
  stabilisation that the Dense path does (`stabilized_joint_solver_diagonal_ridge`
  only stabilises `JointHessianSource::Dense`).

² N=200 converged=false: ARC hits outer_max_iter=60, retries from the
  same ρ, then "ARC retry stalled at iter=60 cost=… |g|=… (prev |g|=…);
  deterministic replay suspected, falling through to degraded plan".
  Cost reaches a stable local minimum (~229) but the gradient at this
  ρ shows |g|≈3500 because the trace term `tr(K · λ_k S_k) ≈ 5774` for
  the dominant axis. The trace term is genuinely large because at this
  ρ region the inner H is ill-conditioned: tiny β-drift between
  consecutive same-ρ retry evals produces |g|=87 → 327 → 2052 → 3566 .
  Cost is still slightly decreasing, so ARC's cost-stagnation gate
  does not fire, but rho doesn't move. **Open root cause**: same
  family-induced operator-mode rank-deficiency story as N=100; here
  PIRLS converges in 16 cycles but the gradient sensitivity at the
  found ρ is dominated by ill-conditioning, not by genuine outer
  stationarity.

## With `--scale-dimensions` (psi-dim = 20)

| n     | budget | result                                  |
| ----: | -----: | --------------------------------------- |
|    50 |  300 s | still running at ~100 s (9 KAPPA evals) |
|   200 |  120 s | **timed out** at 120 s (≤ 7 KAPPA evals) |
|  1000 |  180 s | timed out / TBD                         |

Each KAPPA-PHASE eval at small n takes 5-20 s because the rho-outer
inside it triggers PIRLS spinning 100 cycles (same rank-deficiency
pattern). To meet the goal of N=50/30s with scale_dimensions, the
inner PIRLS at small n needs to converge in O(10) cycles, not 100+.

## Bottlenecks for future improvement

1. **Operator-mode joint-Newton stabilisation** —
   `stabilized_joint_solver_diagonal_ridge` (`custom_family.rs:8685`)
   returns the bare `base_diagonal_ridge` for `JointHessianSource::Operator`.
   The matching Dense branch runs an eigendecomposition and adds a shift
   to lift the minimum eigenvalue above the ridge floor. For dim ≤ 512
   the operator can be materialised cheaply (it already is in the
   dense-fallback inner path); the stabilisation shift could be applied
   to that materialised matrix and added to `diagonal_ridge` before any
   PCG attempt is made. Until then small-n fits accumulate cycles
   solving an ill-conditioned linear system in PCG.

2. **N=1000 → N=2000 per-iter scaling** —
   `DenseSpectralOperator::trace_logdet_operator` (dim=81) takes 0.057 s
   at n=1000 and 0.564 s at n=2000 (≈10× for 2× rows). The hot path is
   `RowKernelDirectionalDerivativeOperator::compute_jf` (a per-row
   `par_chunks_mut` build of the `n × K·rank` projection). Either the
   parallel grain crosses a threshold or cache locality changes; profile
   to confirm.

3. **N=200 ARC retry path** — once the ARC retry triggers and the
   "degraded plan" runs, no further useful work happens. Either the
   retry should be skipped when the prior |g| has not changed
   meaningfully (avoiding the extra eval cost), or the convergence test
   should accept a cost-stagnant point as the optimum-within-numerical-
   precision and stop without falling through.

## Hot-path analysis at N=50K and N=100K

Profile shows the dominant cost is in `RowKernelDirectionalDerivativeOperator::compute_jf`
(unified.rs:10106 calls `op.trace_projected_factor_cached` which calls `compute_jf` on
first use of a new factor value). This builds a `(n_rows, K * rank)` row-major
projection `J · F` where `J_r` is the per-row Jacobian.

Current implementation (`row_kernel.rs:732-759`) loops per row with a strided write:

```rust
jf.as_slice_mut()
    .par_chunks_mut(stride)
    .enumerate()
    .for_each(|(row, jf_row)| {
        for k_col in 0..rank {
            let vec_k = self.kern.jacobian_action(row, f_t.row(k_col).as_slice());
            for k in 0..K {
                jf_row[k * rank + k_col] = vec_k[k];
            }
        }
    });
```

For marginal-slope `jacobian_action(row, slice)` is
`[marginal_design.row(row).dot(slice[marg_range]),
   logslope_design.row(row).dot(slice[logs_range])]` — two row-dot products.

The full kernel is mathematically a **dense matrix-matrix product**:

```
J · F = [ marginal_design · F_marginal_block ;   (concatenated along K axis)
          logslope_design  · F_logslope_block ]
```

A single BLAS-3 GEMM per K-axis (here, two GEMMs) would produce the same data
with single-precision-friendly inner loops, avoiding rayon scheduling overhead
on `K·rank = 162`-element chunks. Measured: at N=100K, each `compute_jf`
takes ~4.5 s (avg over 21 calls); a GEMM at the same shapes should run in
< 0.5 s. The per-eval outer cost would drop from ~48 s to ~10 s, fitting
N=100K in ~120 s instead of the projected ~500 s.

This is the highest-impact optimization left for the plain margslope path
at biobank scale.
