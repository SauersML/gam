# Issue 2627: certificate limit bypass located

The shared-tangent diagnostic exposed an ordering defect in
`crates/gam-solve/src/rho_optimizer/run.rs`: the caller's projected-gradient
requirement caps the value-agreement audit, but subsequent curvature and
gradient-reproducibility rungs widen the stationarity bound again. The recorded
rotated fit consequently certified a 3.061144327128992e-7 gradient despite its
2 * sqrt(f64::EPSILON) requirement.

The local fix reapplies the caller requirement after both widening calculations,
before the final stationarity verdict, preserving the caller-requirement label.
A regression in `run_plan_caller_requirement_tests_2568.rs` exercises both
curvature and reproducibility: each measured rung certifies without a caller
limit and must refuse the same point when the limit is stricter.

This code is **unvalidated and not pushed**. `git diff --check` passed. Both files
were uploaded to the shared MSI source, but the rebuild did not start:

- `msi doctor`: login reachable; acn112 and acn116 DOWN.
- Direct acn112 command returned SSH exit 255.
- `sinfo` showed compute nodes invalid/drained/unknown; acn116 reported
  IDLE+DRAIN+INVALID_REG with reason Low TmpDisk.
- A four-CPU, 16 GiB `srun --immediate=10` allocation on msismall failed:
  Requested nodes are busy. Job 2000 is absent from squeue afterward.
- The requested build log does not exist. There is no live owned build or
  pending allocation to mistake for validation in progress.

Next: rebuild gam-solve and gam-models using the warm y2-target cache on MSI,
run the caller-requirement regression and existing requirement tests, then the
seven response_geometry tests and original gam-pyffi rotation regression.
The change must be tested before claiming it repairs optimized rotation: it may
first expose the search's premature stop as a refusal instead of a bad fit.

CI run 34253714873 (7c3c5b43b) also finished Rust shard 8, job 102173061557:
**943 run, 915 passed, 24 failed, 4 timed out, 8,658 skipped by shard selection,
3,779.472 seconds**. The final summary and all 28 unsuccessful identities are
in `bench/measurements/issue_triage_20260907/rust-shard8-summary-34253714873.txt`.
These results remain an older-head partial census, not a current-main total.

## Immediate wrap-up requested

acn112 became reachable again on the next check. A four-worker warm rebuild of
gam-solve and gam-models started, but the user then twice requested an immediate
finish. The owned build process group 14235 was inspected (timeout, cargo, and
its rustc children only) and terminated with SIGTERM. The pending fix therefore
still has no completed build or test receipt and remains unpublished. Source,
regressions, and diagnostics are preserved for resumption. Other users' jobs
and the existing CI runs were left alone. The build wrapper automatically
retried exit 143, so its owning shell 13962 and inspected retry group 16614
(timeout, cargo, rustc) were also terminated to prevent another restart.
