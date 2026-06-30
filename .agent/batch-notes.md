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

## #850 — SAE warm-start freeze at inner_max_iter==0 (reopened 2026-06-30)
IMPROPERLY CLOSED. Regression test `seed_inner_state_installs_and_reuses_matching_beta`
FAILS on HEAD (hidden by the broken build unblocked in #1652).
Root cause: at max_iter==0, `run_joint_fit_arrow_schur` (fit_drivers.rs) skips the empty
Newton loop but STILL runs the #1026 post-loop decoder-LSQ polish (3544-3584) which refits
β to the LSQ argmin + entry-stage guards — moving β off the seed. The freeze comment at
construction.rs:6264 ("left the seed untouched") is false.
Fix: honor the inner_max_iter==0 freeze contract end-to-end.
Branch stacked on agent/reopen-932 (main does not build without #1652).

## #901 — projected-logdet REML gradient (re-home orphaned FD gate)
Fix commit 7a5bfd9b2 exists (intrinsic pseudo-logdet over range(H_pen)). But the
END-TO-END iso_kappa FD oracles on real Duchon/Matérn smooths — the headline #901
reproduction — were orphaned out of the build by #1601 (deps live in gam-models
drivers post-#1521 carve, not gam_terms::smooth where the include! was commented).
Re-homed into crates/gam-models/.../drivers/iso_kappa_reml_gradient_fd_tests.rs.
Result: 6/8 pass — ρ grads match FD to 1e-7..1e-5 (NO sign flip), ψ grads to
1e-3..3e-3 (NO 1e5 blow-up). The issue's an=-7.72e5/fd=+4.0 catastrophe is GONE.
Open: iso_kappa_duchon_n_smaller_fd (n=20) at worst_psi_rel=7.69e-3 vs tol 5e-3.
Under investigation: FD conditioning at small n vs residual ψ-grad inaccuracy.
