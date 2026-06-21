# PR #1462 third review — Faddeeva to GHQ documentation follow-up

## Context / Scope

Reviewed `fork/fix/faddeeva-ghq-1459` against `origin/main`, focused on `src/inference/quadrature.rs` and the two prior secondary findings for `logit_posterior_mean_exact`.

Latest branch head verified:

```text
bddbc9ae9 fix(#1459): address secondary rp-review feedback
66a72eb2d fix(#1459): address rp-review feedback
49e772075 fix(#1459): replace biased Faddeeva oracle with spectrally-accurate GHQ
3ada0d624 test(#1378): remove prematurely-committed failing row-permutation test
```

## Findings

### Must fix: inline pole-geometry comment is still mathematically backwards

`src/inference/quadrature.rs:4283-4288` says the `√2σ` scaling moves poles to `±iπ/(√2σ)`, so they “recede from the real axis as σ GROWS” and that “large-σ” is well-conditioned.

That conclusion does not follow from the stated pole locations. For the integrand

```text
sigmoid(mu + √2 σ t)
```

nearest poles satisfy

```text
t = (iπ(2n+1) - mu) / (√2 σ)
```

so the nearest imaginary distance is

```text
π / (√2 |σ|)
```

which decreases as `σ` grows. The poles move closer to the real axis as `σ` grows, not farther away. This is the same class of documentation/comment accuracy issue the follow-up was meant to fix.

Suggested fix: rewrite the comment to say poles approach the real axis as `σ` grows, and describe the 128-point GHQ accuracy as empirically/regression validated over the practical grid rather than justified by large-σ pole recession.

### Suggestion: remaining module-level “exact evaluator” wording is stale

`src/inference/quadrature.rs:75-80` still says the module contains an “oracle-style exact evaluator (`logit_posterior_mean_exact`)”. The active implementation is now fixed 128-point GHQ, so “exact evaluator” is misleading. Prefer “oracle-style high-order GHQ reference” or similar.

## Verification

Function rustdoc above `logit_posterior_mean_exact` now correctly describes a 128-point GHQ implementation and places Faddeeva content under “Historical / alternative Faddeeva-series representation.” I did not find the old “mathematically exact up to numerical truncation” routine-level contract there.

No orphan references to the removed helper remain in `src/` or `tests/`:

```text
$ grep -rn "faddeeva_upper_halfplane" src/ tests/
<no output; exit code 1>
```

Focused tests pass:

```text
$ cargo +1.93.0 test --lib "test_logit_posterior_mean_exact" 2>&1
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.26s
     Running unittests src/lib.rs (target/debug/deps/gam-041ef34deac4a4ac)

running 3 tests
test inference::quadrature::tests::test_logit_posterior_mean_exact_matches_high_res_integral ... ok
test inference::quadrature::tests::test_logit_posterior_mean_exact_symmetry_identity ... ok
test inference::quadrature::tests::test_logit_posterior_mean_exact_no_mu_linear_bias_1459 ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 3841 filtered out; finished in 1.77s
```

## Verdict

REQUEST_CHANGES. The implementation and tests appear fine, but documentation/comment accuracy is still not fixed: the inline pole-distance statement contradicts the formula immediately preceding it. The PR is not ready to merge until that comment is corrected.
