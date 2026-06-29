# Batch triage: closed issues 418–597 (agent gam-closed-418-597)

## CRITICAL: HEAD build is broken (blocks all test verification)
`./build.sh nextest run` aborts with 8 ban-gate violations across 4 rules,
introduced by recent #932 perf commits (31e038a30, 152529b48):
- flex_jet.rs:4795,4879  — #[allow(clippy::too_many_arguments)] ×2
- gradient_paths.rs:2608/2609 — #[ignore] + #[test] without assertions (bench)
- cell_moment_assembly.rs:4774 — #[ignore] bench
- row_jet_program.rs:1480 — #[ignore] microbench
- multinomial_reml.rs:935,1061 — #[cfg(test)] on src/ items

This is the SAME recurring class as closed issues #455/#503/#530
("main does not build: ban-gate"). Reopening #503 as the umbrella.

## Triage plan for the rest of 418–597
Parallel-triaged. Strong improperly-closed candidates to verify by running tests:
#500/#509 (over-smoothing under linear constraints still reproduces per release notes),
#563 (RP I-spline λ), #565 (survival edf), #557 (multinomial non-finite Newton),
#561 (multinomial per-class λ fusion), #562 (CI under-coverage), #566 (loc-scale scale underfit),
#582 (Vp scale equivariance). Most "slop/dedup" issues (418–497) verified PROPERLY_FIXED.
