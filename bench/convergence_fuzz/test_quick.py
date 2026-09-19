"""Regression gate for the convergence fuzzer.

``test_quick_plan_has_no_failures`` runs the real ``quick`` plan end to end -
the seeded fixture of every root cause the fuzzer found and fixed, plus the
first cases of the DGP space at small ``n`` - each rep in its own isolated
worker, and requires zero failures of any kind. The other tests pin the
classifier on hand-built records, so a triage that stopped recognising a
failure would fail here rather than quietly passing the gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dgp import FAMILIES, case_spec, draw
from .run import FIXTURES, PLANS
from .triage import failure_causes

BENCH_DIR = Path(__file__).resolve().parent.parent


def test_quick_plan_has_no_failures(tmp_path: Path) -> None:
    out = tmp_path / "quick"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "convergence_fuzz.run",
            "quick",
            "--out",
            str(out),
            "--quiet",
        ],
        cwd=BENCH_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    records = [
        json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(PLANS["quick"]())
    failed = {
        f"case{r['case']}/{r['family']}/n{r['n']}": r["causes"]
        for r in records
        if r["causes"]
    }
    assert not failed, json.dumps(failed, indent=2)
    for rec in records:
        assert rec["certified"] is True and rec["refit"]["certified"] is True


def test_quick_plan_carries_every_fixture() -> None:
    quick = PLANS["quick"]()
    for cause, rep in FIXTURES.items():
        assert rep in quick, cause


def test_full_plan_meets_the_lane_size() -> None:
    reps = PLANS["full"]()
    assert len(reps) >= 2000
    assert {r.family for r in reps} == set(FAMILIES)
    assert {case_spec(r.case).p for r in reps} == set(range(1, 9))


def test_draw_is_seeded() -> None:
    a = draw(3, "poisson", 100)
    b = draw(3, "poisson", 100)
    for name in a.train:
        assert (a.train[name] == b.train[name]).all()


def _certified(score: float) -> dict[str, Any]:
    return {
        "reml_score": score,
        "certified": True,
        "convergence": {"certified": True, "outer": {"kind": "stationary"}},
    }


def test_triage_labels() -> None:
    assert failure_causes({"status": "timeout"}) == ["hang"]
    assert failure_causes({"status": "crash"}) == ["crash"]
    raised = failure_causes(
        {"status": "error", "errors": {"fit": "ValueError: rank 3 of 12"}}
    )
    assert raised == ["raise:fit:ValueError: rank # of #"]
    ok = {"status": "ok", **_certified(10.0), "refit": _certified(10.0)}
    assert failure_causes(ok) == []
    worse = {"status": "ok", **_certified(10.5), "refit": _certified(10.0)}
    assert failure_causes(worse) == ["reml_mismatch:fit_worse"]
    uncert = {
        "status": "ok",
        **_certified(10.0),
        "certified": False,
        "refit": _certified(10.0),
    }
    assert failure_causes(uncert)[0].startswith("uncertified:fit:")
    nonfinite = {
        "status": "ok",
        **_certified(10.0),
        "refit": _certified(10.0),
        "pred_finite": False,
    }
    assert failure_causes(nonfinite) == ["nonfinite:predict"]
