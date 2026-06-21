"""Regression guard for #1390 — brms MCMC CV per-fold timeout wiring.

Issue #1390: on the `us48_demand_31day` benchmark shard the brms reference
contender kept running full Bayesian MCMC cross-validation after the GAM lane
had finished, overrunning the 42-minute shard budget. The GNU `timeout`
SIGKILLed the whole shard (exit 124) AFTER gam was done, discarding every
result with no per-fold attribution.

The fix (commit d74ef8af) bounds each brms CV fold with a per-invocation
timeout so a slow/hung fold becomes a recorded, visible failure instead of
consuming the shard, and pairs it with a scenario-aware shard budget for the
heavy daily-demand panel.

This is a pure-source contract test (no Rust build, no subprocess). It pins the
three load-bearing pieces of that fix so a future refactor cannot silently drop
them and re-open the bulk-shard-kill:

  1. `run_cmd` accepts a `timeout_sec` override and actually enforces it
     (waits with that deadline, then terminates/kills the child).
  2. the brms CV driver reads `BENCH_BRMS_FOLD_TIMEOUT_SEC` and passes the
     resulting per-fold cap into `run_cmd(..., timeout_sec=...)`.
  3. the benchmark workflow gives `us48_demand_31day` an explicit (larger)
     shard budget and the brms-fold cap is referenced in the workflow rationale.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_run_cmd_accepts_and_enforces_per_invocation_timeout():
    src = _read("bench/_run_suite_datasets.py")
    # The override parameter must exist on run_cmd.
    assert "def run_cmd(" in src, "run_cmd helper missing"
    assert "timeout_sec" in src, "run_cmd lost its timeout_sec override (#1390)"
    # It must actually be enforced: a bounded wait that escalates to kill on
    # overrun (not merely accepted and ignored).
    assert "effective_timeout" in src
    assert "proc.wait(timeout=effective_timeout)" in src, (
        "run_cmd no longer waits on the per-invocation timeout (#1390)"
    )
    assert "except subprocess.TimeoutExpired" in src, (
        "run_cmd no longer catches the per-invocation timeout (#1390)"
    )
    assert "proc.kill()" in src, "run_cmd no longer kills an overrunning child (#1390)"
    # rc=124 is the conventional timeout exit code the shard log greps for.
    assert "rc=124" in src, "run_cmd lost the timeout exit-code attribution (#1390)"


def test_brms_cv_driver_caps_each_fold():
    src = _read("bench/_run_suite_external.py")
    assert "def run_external_r_brms_cv(" in src, "brms CV driver missing"
    # The env override the fix introduced, with a finite default.
    assert "BENCH_BRMS_FOLD_TIMEOUT_SEC" in src, (
        "brms per-fold timeout env override removed (#1390)"
    )
    assert "brms_fold_timeout" in src
    # The cap must be threaded into the actual run_cmd call for the fold, not
    # just computed and dropped.
    assert "timeout_sec=brms_fold_timeout" in src, (
        "brms fold timeout no longer reaches run_cmd (#1390)"
    )


def test_benchmark_workflow_gives_heavy_shard_an_explicit_budget():
    wf = _read(".github/workflows/benchmark.yml")
    # The scenario that triggered the bug must carry its own (larger) budget.
    assert "us48_demand_31day" in wf, "heavy demand shard scenario missing from workflow"
    assert "BENCH_SHARD_TIMEOUT" in wf, "per-shard budget knob removed from workflow (#1390)"
    # The workflow rationale references the per-fold cap so the two stay coupled.
    assert "BENCH_BRMS_FOLD_TIMEOUT_SEC" in wf, (
        "workflow no longer references the brms per-fold cap (#1390)"
    )
