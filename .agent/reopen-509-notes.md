# Reopen #509 — monotone box-reparam over-smooths (second face)

## Status
Issue #509 reopened 2026-06-30. The repo owner's closing comment explicitly
admits the SECOND face of the title ("over-smooths to a flat constant on
already-monotone data") was left unfixed:

> "the monotone fit now completes but collapses to range ≈ 0 (REML drives λ to
>  the ceiling)… shares a root cause with the #500 bc=clamped over-smoothing
>  defect."

A box-reparam ridge-rebuild (term_specs.rs:7610-7651) since killed the GROSS
flat-collapse, but a residual remains: the R-free regression test
`monotone_increasing_shape_smooth_fits_already_monotone_data`
(tests/regressions/smooths/monotone_shape_smooth_aborts_fit.rs) is RED on
origin/main HEAD:

    mono RMSE 0.0066 vs free RMSE 0.0056  (ratio 1.18 > 1.10 gate)

## Empirical root cause (diagnostic harness, n=400, y=x²+N(0,0.05²))
                         lambdas              edf_by_block   reml_score  outer_iters
  unconstrained s(x)   [297.8, 1.7e-3]       [4.71, 1.00]   -612.93     12
  monotone s(x)        [ 20.1, 4.98e-2]      [6.42, 1.00]   -599.25      5  (== seed [3,-3])

The monotone fit parks at the integer SEED ρ=[3,-3] after only 5 outer iters,
never refining λ. Its λ_wiggle=20 (vs the true optimum 298) ⇒ EDF 6.4 (vs 4.7)
⇒ UNDER-smoothed, fitting noise ⇒ worse RMSE. The test's stated hypothesis
("λ→ceiling") is the OPPOSITE of the residual defect.

Working hypothesis: the cumulative-sum coefficient congruence β=Tγ is not
REML-invariant — the penalty pseudo-log-det term log|S_λ|₊ in the REML/LAML
Occam factor is distorted by the non-orthogonal T (cond(T) grows with basis
dim), biasing the outer λ-search and tripping early outer convergence.

## Plan
1. Confirm non-invariance of the REML score under β=Tγ (penalty log-det vs
   |H| mismatch) — instrument log-det terms.
2. Fix root cause so the box-reparam monotone fit's REML optimum matches the
   unconstrained fit on non-binding data (and #500 bc=clamped shares it).
3. Keep binding-constraint behavior (monotone_shape_binding_constraint.rs) and
   convex/concave (order-2) paths green.
4. Re-tighten the regression gate and add invariance regression tests.
