"""docs/benchmarks.md is generated from bench/pygam_comparison/ and must match it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmarks_page_matches_committed_measurements():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_benchmarks_doc.py"), "--check"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_benchmarks_page_lists_every_accuracy_loss():
    page = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    source = (ROOT / "bench" / "pygam_comparison" / "accuracy_cv.md").read_text(encoding="utf-8")
    losses = [line for line in source.splitlines() if line.startswith("| ") and line.rstrip().endswith("|")
              and "| **LOSS** |" in line]
    assert losses
    for line in losses:
        case, n, family, metric = (cell.strip() for cell in line.strip("| ").split("|")[:4])
        assert f"| {case} | {n} | {family} | {metric} |" in page, line
