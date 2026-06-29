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
