# PR #1462 Review: Faddeeva oracle replaced with GHQ

## Context / Scope

Reviewed `fork/fix/faddeeva-ghq-1459` (`49e772075`) against `origin/main` for PR #1462 / issue #1459. Focus areas: `src/inference/quadrature.rs`, `compute_gauss_hermite_n`, `logit_posterior_mean_exact`, removal of `faddeeva_upper_halfplane`, dead-code/caller checks, and focused tests.

## Verdict

**REQUEST_CHANGES**

The GHQ formula and physicists-Hermite scaling are mathematically correct for `E[sigmoid(mu + sigma Z)]`, the removed Faddeeva helper has no remaining callers, and the focused tests pass. However, the new implementation comment in `logit_posterior_mean_exact` incorrectly claims sigmoid is entire/analytic everywhere and overclaims machine-precision/exactness from fixed 128-point GHQ. The new bias regression test also does not cover the full `{mu} x {sigma}` grid stated in the issue/review request.

## Findings

1. **Must fix: incorrect mathematical claim in oracle docs/comment**  
   `src/inference/quadrature.rs:4297` says the integrand is entire because sigmoid is analytic everywhere. Logistic sigmoid is meromorphic, with complex poles at odd multiples of `i*pi`; after scaling by `sqrt(2)*sigma`, those poles move closer to the real axis as `sigma` grows. The GHQ substitution is correct, but the convergence justification should be reworded to avoid the false entire-function/machine-precision guarantee.

2. **Must fix: regression test does not cover all issue-flagged combinations**  
   `src/inference/quadrature.rs:5006` tests 8 cases: `(1, .02/.5/2)`, `(3, .05/.5/2)`, `(-2, .05/2)`. It does not cover the full `mu in {1, 3, -2}` and `sigma in {0.02, 0.05, 0.5, 2.0}` Cartesian grid requested/flagged; missing cases are `(1,0.05)`, `(3,0.02)`, `(-2,0.02)`, and `(-2,0.5)`.

## Verification Notes

- GHQ variable substitution is correct: with `Z ~ N(0,1)` and `z = sqrt(2) t`, `E[f(mu+sigma Z)] = (1/sqrt(pi)) integral exp(-t^2) f(mu + sqrt(2) sigma t) dt`.
- `compute_gauss_hermite_n` uses the physicists-Hermite Jacobi matrix (`off_diag = sqrt((i+1)/2)`) and weights scaled by `mu0 = sqrt(pi)`, so the PR's final `1/sqrt(pi)` normalization is correct.
- `grep -R faddeeva_upper_halfplane ...` returns no matches on the PR branch.
- `grep -R logit_posterior_mean_exact ...` shows only documentation/comments and tests in `src/inference/quadrature.rs`; no production caller relies on the looser old tolerance.
- `cargo +1.93.0 check --lib --tests 2>&1 | grep -iE "unused|dead.code"` produced no output.
- `cargo +1.93.0 test --lib "test_logit_posterior_mean_exact" 2>&1` passed: 3 passed, 0 failed, 3841 filtered out.

## Recommendation

Reword the GHQ comment to state the real-line integrand is smooth/bounded and empirically accurate over the tested range, not entire or guaranteed machine-precision for all sigma. Expand `test_logit_posterior_mean_exact_no_mu_linear_bias_1459` to include the full 12-case grid from the issue/review request. After those changes, this should be approvable.
