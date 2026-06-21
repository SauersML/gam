# PR #1461 Review: large-K `from_dense_weights` coverage

## Context / Scope

Reviewed PR branch `fork/test/large-k-from-dense-weights-1450` at `0273db793` against `origin/main` after fetching both remotes. The substantive target change is the new test `from_dense_weights_large_k_support_proposal_1450` in `src/terms/sae/manifold/tests.rs`.

## Commands and outputs

```text
$ cd /Users/t3rpz/projects/gam && git fetch origin && git fetch fork && git log --oneline fork/test/large-k-from-dense-weights-1450 -3 && git diff origin/main...fork/test/large-k-from-dense-weights-1450
From https://github.com/SauersML/gam
 * [new branch]          claude/upbeat-thompson-g4twh8 -> origin/claude/upbeat-thompson-g4twh8
   3ada0d624..2e2871e75  main        -> origin/main
 * [new branch]          verify-1454 -> origin/verify-1454
 * [new tag]             main-3ada0d6243992a934469948a0e1ea3fc6ed08c3f -> main-3ada0d6243992a934469948a0e1ea3fc6ed08c3f
 * [new tag]             v0.3.120    -> v0.3.120
From https://github.com/HomunculusLabs/gam
 * [new branch]          claude/upbeat-thompson-1oa2kb -> fork/claude/upbeat-thompson-1oa2kb
 * [new branch]          fix-1384-compare-models-family-guard -> fork/fix-1384-compare-models-family-guard
   98a3d70ef..b00a327a2  fix/feature-uniqueness-tautology-1413 -> fork/fix/feature-uniqueness-tautology-1413
 + ae3877752...6bf6852fc fix/sae-stale-dim-assertions-1442 -> fork/fix/sae-stale-dim-assertions-1442  (forced update)
   cc0bf3824..d5846515f  fix/sae-typed-outer-gradient-error-1436 -> fork/fix/sae-typed-outer-gradient-error-1436
 * [new branch]          hunt/numrobust-1781963611 -> fork/hunt/numrobust-1781963611
 * [new branch]          main       -> fork/main
0273db793 test(#1450): large-K from_dense_weights end-to-end coverage
3ada0d624 test(#1378): remove prematurely-committed failing row-permutation test
37d9e399f fix(#1379): clamp per-block penalty trace to [0, rank] so univariate matern(x) fits
[diff inspected; target patch adds 30 lines to src/terms/sae/manifold/tests.rs]
```

```text
$ cd /Users/t3rpz/projects/gam && git switch --detach fork/test/large-k-from-dense-weights-1450
HEAD is now at 0273db793 test(#1450): large-K from_dense_weights end-to-end coverage
```

```text
$ cd /Users/t3rpz/projects/gam && cargo +1.93.0 test --lib "from_dense_weights_large_k_support_proposal_1450" 2>&1
   Compiling gam v0.3.120 (/Users/t3rpz/projects/gam)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1m 27s
     Running unittests src/lib.rs (target/debug/deps/gam-041ef34deac4a4ac)

running 1 test
test terms::sae::manifold::tests::from_dense_weights_large_k_support_proposal_1450 ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3843 filtered out; finished in 0.03s
```

```text
$ cd /Users/t3rpz/projects/gam && wc -l src/terms/sae/manifold/tests.rs
   10000 src/terms/sae/manifold/tests.rs
```

## Findings

- `src/terms/sae/manifold/row_layout.rs:118-124` uses `idx.select_nth_unstable_by(cap - 1, ...)` when `cap < k`, then truncates to `cap`; this is the intended partial-select path, not a full sort.
- `src/terms/sae/manifold/tests.rs:3912-3918` calls `SaeRowLayout::from_dense_weights(...)` directly. It does not use the existing `from_active_atoms` hand-picked-index helper path.
- The test uses `k_atoms = 100_000`, `k_true = 4`, and passes `k_true` as the active cap. Since `cap = 4 < k = 100000`, the test exercises the partial-select branch.
- The planted atoms are `[0, 25000, 50000, 75000]`; `100000 / 4` divides exactly here, so there is no truncation/collision problem in this test.
- Planted values are approximately `0.20..0.26`; row-relative cutoff is `1e-3 * row_peak`, around `2e-4`; background is `1e-9`. The cutoff is appropriate for dropping the noise, and the top-4 selection should be robust.
- `tests.rs` is exactly 10,000 lines. It does not exceed the 10,000-line gate, but it is at the limit.

## Recommendations

- Optional: change the comment wording from “end-to-end” to “direct large-K `from_dense_weights` coverage” unless a separate production-path test through `IBPMap`/`assemble_arrow_schur` is added.
- Optional: replace the final work inequality with an exact compact-work assertion or compute the dense comparison in `u128`; current arithmetic is fine on 64-bit but is not portable to 32-bit `usize`.

## Verdict

APPROVE. The test covers the actual `from_dense_weights` large-K partial-select path and passes. The issues above are wording/robustness suggestions, not merge blockers.
