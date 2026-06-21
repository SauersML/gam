# PR #1462 Secondary Review: GHQ follow-up fixes

## Context / Scope

Reviewed `fork/fix/faddeeva-ghq-1459` against `origin/main` on 2026-06-21 after fetching `fork`. This secondary review verified the two prior REQUEST_CHANGES findings, re-checked the GHQ math, searched for remaining Faddeeva callers, checked warning output, ran focused tests, and inspected callers of `logit_posterior_mean_exact`.

## Verdict

**REQUEST_CHANGES**

The implementation math and 12-case regression grid are now correct, but the prior documentation/comment issue is not fully resolved. The rustdoc above `logit_posterior_mean_exact` still documents the removed Faddeeva implementation as the routine's exact contract, and the new inline comment still incorrectly says the composed sigmoid is entire for small nonzero `sigma`.

## Findings

### 1. Stale rustdoc still documents removed Faddeeva exact implementation

- Location: `src/inference/quadrature.rs:4230-4278`
- The public/rustdoc block above `logit_posterior_mean_exact` still presents the old Faddeeva-series formula and concludes: "Therefore this routine is mathematically exact up to numerical truncation and numerical evaluation error of w(z)."
- The function now uses fixed 128-point Gauss-Hermite quadrature, not a Faddeeva evaluator. This keeps the earlier overclaim/exactness concern alive at the API/documentation level.
- Recommended fix: rewrite the rustdoc to describe the current GHQ128 oracle and its tested accuracy envelope, or clearly mark the Faddeeva material as historical/alternative derivation rather than implementation contract.

### 2. New inline comment still has an incorrect analyticity statement

- Location: `src/inference/quadrature.rs:4294-4302`
- The comment correctly mentions that logistic sigmoid is meromorphic with complex poles at odd multiples of `iπ`, and correctly avoids the old machine-precision wording.
- However, it says `sigmoid(μ + √2σt)` is "entire as a function of t only when σ is small." For every nonzero `sigma`, the composition is still meromorphic with poles at `(iπ(2n+1)-μ)/(√2σ)`, not entire. Small `sigma` only moves poles farther from the real axis and improves strip-analytic/GHQ convergence.
- Recommended fix: replace this with wording like: "For nonzero σ the composed integrand is meromorphic in complex t, with poles whose distance from the real axis scales like π/(√2|σ|); GHQ convergence is geometric while those poles remain far from the real axis relative to the node span."

## Verification Notes

- Prior finding #2 is fixed: `test_logit_posterior_mean_exact_no_mu_linear_bias_1459` uses nested loops over 3 `mu` values and 4 `sigma` values, for 12 total cases.
- GHQ formula is correct: with `Z ~ N(0,1)` and `Z = √2 t`, `E[sigmoid(mu + sigma Z)] = (1/√π) ∫ exp(-t²) sigmoid(mu + √2 sigma t) dt`, matching the implementation.
- `compute_gauss_hermite_n` uses physicists-Hermite recurrence coefficients `sqrt((i+1)/2)` and weights scaled by `mu0 = sqrt(pi)`, so the final `1/sqrt(pi)` normalization is correct.
- No live `faddeeva_upper_halfplane` callers remain; only the prior review markdown mentions the symbol.
- `logit_posterior_mean_exact` has no external production callers under `src/` or `tests/`; remaining uses are documentation/comments and in-file tests.
- `cargo +1.93.0 check --lib --tests 2>&1 | grep -iE "unused|dead.code"` produced no output.
- `cargo +1.93.0 test --lib "test_logit_posterior_mean_exact" 2>&1` passed all 3 focused tests, including the new bias test.

## Command Outputs

```text
$ git fetch fork && git log --oneline fork/fix/faddeeva-ghq-1459 -5 && git diff origin/main...fork/fix/faddeeva-ghq-1459
66a72eb2d fix(#1459): address rp-review feedback
49e772075 fix(#1459): replace biased Faddeeva oracle with spectrally-accurate GHQ
3ada0d624 test(#1378): remove prematurely-committed failing row-permutation test
37d9e399f fix(#1379): clamp per-block penalty trace to [0, rank] so univariate matern(x) fits
915d7494f fix(#1456): stabilize 2-D center-selection split + drop unmet rotation test
[full diff inspected; substantive quadrature change in src/inference/quadrature.rs]
```

```text
$ git grep -n "faddeeva_upper_halfplane" -- .
docs/reviews/pr-1462-faddeeva-ghq.md:5:Reviewed `fork/fix/faddeeva-ghq-1459` (`49e772075`) against `origin/main` for PR #1462 / issue #1459. Focus areas: `src/inference/quadrature.rs`, `compute_gauss_hermite_n`, `logit_posterior_mean_exact`, removal of `faddeeva_upper_halfplane`, dead-code/caller checks, and focused tests.
docs/reviews/pr-1462-faddeeva-ghq.md:25:- `grep -R faddeeva_upper_halfplane ...` returns no matches on the PR branch.
```

```text
$ cargo +1.93.0 check --lib --tests 2>&1 | grep -iE "unused|dead.code"
[no output]
```

```text
$ cargo +1.93.0 test --lib "test_logit_posterior_mean_exact" 2>&1
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running unittests src/lib.rs (target/debug/deps/gam-041ef34deac4a4ac)

running 3 tests
test inference::quadrature::tests::test_logit_posterior_mean_exact_matches_high_res_integral ... ok
test inference::quadrature::tests::test_logit_posterior_mean_exact_symmetry_identity ... ok
test inference::quadrature::tests::test_logit_posterior_mean_exact_no_mu_linear_bias_1459 ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 3841 filtered out; finished in 1.81s
```

```text
$ git grep -n "logit_posterior_mean_exact" -- src tests
src/inference/quadrature.rs:80://!    (`logit_posterior_mean_exact`) documenting the mathematics.
src/inference/quadrature.rs:1162:    // scaled-erfcx terms (see `logit_posterior_mean_exact` below for the full
src/inference/quadrature.rs:4279:pub fn logit_posterior_mean_exact(mu: f64, sigma: f64) -> f64 {
src/inference/quadrature.rs:4968:    fn test_logit_posterior_mean_exact_symmetry_identity() {
src/inference/quadrature.rs:4972:            let p = logit_posterior_mean_exact(mu, sigma);
src/inference/quadrature.rs:4973:            let q = logit_posterior_mean_exact(-mu, sigma);
src/inference/quadrature.rs:4979:    fn test_logit_posterior_mean_exact_matches_high_res_integral() {
src/inference/quadrature.rs:4983:            let exact = logit_posterior_mean_exact(mu, sigma);
src/inference/quadrature.rs:4998:            let exact = logit_posterior_mean_exact(eta, se);
src/inference/quadrature.rs:5010:    fn test_logit_posterior_mean_exact_no_mu_linear_bias_1459() {
src/inference/quadrature.rs:5013:                let exact = logit_posterior_mean_exact(mu, sigma);
```
