"""Quick mode of the term-structure convergence fuzzer as a regression test.

Runs the real ``fuzz_terms_quick`` plan (a fixed set of fuzz cases that
together cover every term kind, at n in {50, 500}, all three families, a
fresh subprocess per fit) and requires every fit to be clean: finished, every
phase ran, the summary certifies the optimum, and every point and interval
prediction is finite. A failure names the rep so it reruns in isolation with
``python worker.py gamfit FAMILY N DESIGN SEED``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .fuzz_terms import KINDS, case_terms, failure_cause
from .plans import FUZZ_QUICK_CASES, PLANS

BENCH_DIR = Path(__file__).resolve().parent.parent


def test_quick_cases_cover_every_term_kind() -> None:
    covered = {t.kind for case in FUZZ_QUICK_CASES for t in case_terms(case)}
    assert covered == set(KINDS), sorted(set(KINDS) - covered)


def test_fuzz_terms_quick_plan_has_no_failures(tmp_path: Path) -> None:
    out = tmp_path / "fuzz_quick"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pygam_compare.run",
            "fuzz_terms_quick",
            "--out",
            str(out),
            "--quiet",
        ],
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    records = [
        json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()
    ]
    plan = PLANS["fuzz_terms_quick"]
    assert len(records) == len(plan.cells) * plan.reps * len(plan.libs)
    failures = [
        f"{r['family']} {r['n']} {r['design']} {r['seed']}: {failure_cause(r)}"
        for r in records
        if failure_cause(r) is not None
    ]
    assert not failures, "\n".join(failures)


def test_failure_cause_groups_by_recorded_exception_head() -> None:
    # The worker keeps only the traceback's last 2000 characters, so a long
    # engine message loses its exception line; triage must group by the head
    # the worker recorded, not by whatever line the tail happens to start on.
    message = "Outer optimization did not certify: |Pg|=1.5e-6 " + "x" * 3000
    tail = f"{message}\nvariant: EstimationError::RemlDidNotConverge\ncategory: convergence\n"
    rec = {
        "status": "ok",
        "errors": {"fit": tail[-2000:]},
        "error_types": {"fit": "RemlConvergenceError"},
        "error_heads": {"fit": message},
    }
    cause = failure_cause(rec)
    assert cause is not None
    assert cause.startswith(
        "fit:RemlConvergenceError: [EstimationError::RemlDidNotConverge] "
        "Outer optimization did not certify: |Pg|=#"
    ), cause
