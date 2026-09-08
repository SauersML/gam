# Issue 2627: verification ledger

The acceptance condition remains zero failing tests across the full Rust
workspace, gam-pyffi, Python (including slow and torch populations), reference
quality tests, and doctests. A passing subset, missing population, timeout,
deleted regression, or build failure does not satisfy that condition.

## September 8: periodic penalty regression

`periodic_bspline_terms_build_with_cyclic_penalty_and_formula_alias` requested
`BSplineIdentifiability::None` but expected the constant-function penalty to
vanish under sum-to-zero centering. The recorded MSI failure was `2 != 1`.
The production builder intentionally retains that penalty when the constant
survives (#2783); discarding it would restore intercept confounding.

The test now checks both cases without changing production behavior:

- Uncentered: two distinct penalty sources, roughness nullity one, constant
  penalty rank one, zero roughness energy on the constant vector, and positive
  constant-function energy.
- Centered: one full-rank penalty in a coefficient chart one dimension smaller.
- Existing periodic wrapping, derivatives, formula aliases, and tensor-margin
  assertions remain active.

MSI verification: **5 passed, 0 failed, 0 ignored, 0 filtered out, 0.10 seconds**.
The exact edited file was compiled with `rustc --test`, linked against the
existing `w5-target/debug/deps/libgam-d2bdd2ba4cf1842d.rlib` and its matching
`libndarray-3e2fc606731d0706.rlib`. The periodic builder's SHA-256 was identical
in the local, y2, and w5 sources:
`f4c93765f226c90803f04a347d6d16d484618d7314cc253c7b1f0f155ed35300`.
This is a focused test receipt against existing compiled dependencies, not a
fresh whole-worktree build. Output is in
`bench/measurements/issue_triage_20260907/periodic-2627-direct.log`.

## Remaining verification and failures

The attempted Cargo/nextest root integration build did not reach test execution.
After removing obsolete inherited OpenBLAS linker flags, the root scanner
reported 26 violations: four undocumented unwraps in the atlas nerve test and
22 unused underscore parameters in newly added derivative APIs. These must be
resolved before the current worktree can receive a full build verdict.

The September 8 w5 logs also contain remaining CLI, survival, model
materialization, and SAE failures and timeouts. Their selections exclude other
tests, so they are not a whole-suite denominator. The active GitHub Rust CI
run discovered during this audit is 34253714873, at 7c3c5b43b0e34823d2fd4c9d135d7cac63694676.
Its terminal results still need inspection. No full-suite pass has been proven;
the issue must remain open.
