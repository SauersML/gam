# Batch 778–957 triage — agent gam-closed-778-957

## CRITICAL FINDING (anchor issue #932)
HEAD does NOT build. `build.rs` ban-scanner aborts every cargo/maturin invocation
with 8 violations across 4 rules — all debris from the in-flight #932 jet-cutover
that was landed half-finished and the issue closed "completed":

- 2× `#[allow(clippy::too_many_arguments)]`  flex_jet.rs:4795,4879
- 3× `#[ignore]` timing microbenches          gradient_paths.rs:2609,
                                               cell_moment_assembly.rs:4774,
                                               row_jet_program.rs:1480
- 1× `#[test]` without assertions             gradient_paths.rs:2608 (same bench)
- 2× `#[cfg(test)]` on src/ item              multinomial_reml.rs:935,1061

This blocks ALL verification for every agent on every box. Fixing first.

## Plan
1. Unblock the build (proper fixes, not lint-silencing): delete pure-timing
   microbenches (covered by non-ignored correctness siblings), bundle test-helper
   args into a struct, move cfg(test) methods into the test mod's impl block.
2. Address #932 residual: one live production hand-derivative path remains
   (row_primary_hessian.rs compute_row_analytic_flex_from_parts_into).
3. Continue triage: #855 (tolerance-mask), #850 (warm-start freeze), #901 (disabled FD gate).

## Other improperly-closed candidates from parallel triage
- #855 SAS observed-Jacobian: closed via test-tolerance loosening only, no prod change.
- #850 SAE inner_max_iter=0 freeze: bug reproduces on HEAD (call-path clobbers seed).
- #901 non-Gaussian REML pseudo-logdet FD gate commented out of build.
- #778 explicit "not implemented / defer" marked COMPLETED.
