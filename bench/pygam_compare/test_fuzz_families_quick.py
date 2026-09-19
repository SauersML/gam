"""Quick mode of the family/link convergence fuzzer as a regression test.

Runs the real ``fuzz_families_quick`` plan (every family/link label at the
base, edge, zeros and lowdisp regimes, n in {50, 500}, a fresh subprocess per
fit) and requires every fit to be clean: finished, every phase ran, the
summary certifies the optimum, every point and interval prediction is finite
and the estimated scale is finite and positive. A failure names the rep so it
reruns in isolation with ``python worker.py gamfit FAMILY N DESIGN SEED``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .fuzz_families import CASES, FAMILY_LABELS, REGIMES, draw, failure_cause
from .plans import PLANS

BENCH_DIR = Path(__file__).resolve().parent.parent


def test_quick_plan_covers_every_family_label() -> None:
    plan = PLANS["fuzz_families_quick"]
    assert {c.family for c in plan.cells} == set(FAMILY_LABELS)


def test_every_draw_is_finite_and_in_support() -> None:
    for case in CASES:
        for regime in REGIMES:
            data = draw(case.label, 50, regime, 0)
            y = data.train["y"]
            assert np.all(np.isfinite(y)), (case.label, regime)
            assert np.all(np.isfinite(data.mu_test)), (case.label, regime)
            kind = case.kind.split(":")[0]
            if kind in ("gamma", "invgauss"):
                assert np.all(y > 0), (case.label, regime)
            elif kind in ("poisson", "negbin", "tweedie"):
                assert np.all(y >= 0), (case.label, regime)
            elif kind == "beta":
                assert np.all((y > 0) & (y < 1)), (case.label, regime)
            elif kind == "binomial":
                assert np.all((y >= 0) & (y <= 1)), (case.label, regime)
                assert np.all(data.train["w"] >= 1), (case.label, regime)


def test_fuzz_families_quick_plan_has_no_failures(tmp_path: Path) -> None:
    out = tmp_path / "fuzz_families_quick"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pygam_compare.run",
            "fuzz_families_quick",
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
    plan = PLANS["fuzz_families_quick"]
    assert len(records) == len(plan.cells) * plan.reps * len(plan.libs)
    failures = [
        f"'{r['family']}' {r['n']} {r['design']} {r['seed']}: {failure_cause(r)}"
        for r in records
        if failure_cause(r) is not None
    ]
    assert not failures, "\n".join(failures)
