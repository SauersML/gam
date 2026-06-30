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

# Triage batch: CLOSED #238–#417 (agent gam-closed-238-417)

## Method
For each issue: read `gh issue view <N> --comments`, verify the fix in CURRENT code,
check for a regression test. Reopen only when there is concrete evidence of improper
closure (closure contradiction, missing wiring, reproducible bug, or missing promised
work). The repo is heavily defended (build-time owed-work bans), so most closures are
genuine — improper ones are subtle.

## Confirmed improperly-closed
- **#415** — GPU↔CPU row-kernel parity-lock. Owner's own closing comment says
  "Leaving open as a standalone testing task — it shouldn't be auto-closed by a dedup
  pass," yet the issue is CLOSED as COMPLETED. The promised CI parity-lock fixtures
  (CPU formula == CPU-oracle-of-device-kernel == device kernel, every formula variant)
  do not exist as a unified, build-wired harness. REOPENING + building the harness.

## Verified PROPER (spot-checked in code, agents' "no-commit" flags were false positives)
- #252 BMS degree-21 cache: base built at degree 9, d15/d21 extended lazily (cell_moment_assembly.rs).
- #253 BMS row neglog/grad discard: RowPrimaryEvalCache now stores neglog+grad+hess.
- #276 PIRLS X·Qs before admission: try_gpu_pirls_loop_admit gates before lazy materialize.
- #298 arrow_schur PCG diagnostics: PcgDiagnostics struct exists + returned.
- #301 Surv shorthand: resolve_saved_survival_time_columns helper wired to 4 call sites.
- #335/#337 family y-validation: validate_response_support wired into fit + CLI paths.
- #344/#351 cloglog/probit tail: expm1/erfc forms in inverse_link.rs, no tail collapse.

## Still to deep-check (SUSPICIOUS from triage)
- #356 SAE joint solve, #365 GAMLSS smooth-noise mean, #366 loc-scale survival, #410/#416 unify.
