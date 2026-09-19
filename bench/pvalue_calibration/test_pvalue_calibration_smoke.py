"""Regression + contract tests for the p-value calibration harness.

``test_ci_plan_is_calibrated`` runs the real ``ci`` plan end to end (a few
cells, 200 fixed seeds each, policed worker subprocesses) and fails when any
gamfit p-value surface rejects its true null more often than a valid p-value
can. The tolerance is not hand-picked: it is the upper quantile of the
rejection count's own sampling law, ``Binomial(R, a)``, at family-wise
false-alarm rate ``report.FALSE_ALARM`` (see ``report.reject_bound``). Every
rep that produced no p-value counts as a rejection, so a failing fit can only
make the check stricter.

The other tests pin the harness's own rules on hand-built records, so a report
that stopped flagging an anti-conservative row, or started dropping reps that
produced no p-value, fails here rather than quietly flattering gamfit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import report
from .plans import PLANS, Cell, Plan
from .run import pending_chunks, run_chunk
from .worker import alt_delta, expected_surfaces, make_data

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent
BASELINE = Path(__file__).resolve().parent / "baseline" / "quick"


def test_ci_plan_is_calibrated(tmp_path: Path) -> None:
    out = tmp_path / "ci"
    proc = subprocess.run(
        [sys.executable, "-m", "pvalue_calibration.run", "ci", "--out", str(out), "--quiet"],
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    records = [json.loads(ln) for ln in (out / "records.jsonl").read_text().splitlines()]
    plan = PLANS["ci"]
    assert len(records) == len(plan.cells) * plan.reps
    # A rep that produced no p-value (a fit that raised, a missing row) is not
    # excused here: anti_conservative counts it as a rejection. What must hold
    # is that every fit that did produce a p-value reported it where expected.
    for rec in records:
        if rec["status"] != "ok":
            continue
        for hyp in ("null", "alt"):
            for lib, surfaces in rec["expected_surfaces"].items():
                for s in surfaces:
                    name = f"{lib}.{s}"
                    if f"{hyp}.{name}" in rec["missing"]:
                        continue
                    assert name in rec["p"][hyp], (rec["key"], rec["seed"], hyp, name)
    table = report.rows(records)
    assert table and all(r.reps == plan.reps for r in table)
    flagged = [(r.cell, r.surface, a, r.rejections) for r, a in report.anti_conservative(table)]
    assert not flagged, flagged
    # Every row has a power under its matched alternative.
    assert all(r.power is not None for r in table)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["thread_env"]["RAYON_NUM_THREADS"] == "1"
    assert "safety net" in meta["safety_net"]
    assert "## Calibration" in (out / "report.md").read_text()


def test_chunk_killed_by_safety_net_records_the_seed_it_died_on(tmp_path: Path) -> None:
    recs = run_chunk(Cell("gaussian", 60, "smooth"), 3, 6, ("gamfit",), 0.0, 1e9, str(tmp_path))
    assert [(r["seed"], r["status"]) for r in recs] == [(3, "timeout")]


# Stands in for worker.py: finishes every seed at once except ``HANG``, which
# runs until the safety net kills it.
FAKE_WORKER = """
import json, sys, time
family, n, null, start, stop = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
for seed in range(start, stop):
    if seed == {hang}:
        time.sleep(3600)
    rec = dict(family=family, n=n, null=null, seed=seed, status="ok",
               expected_surfaces={{"gamfit": ["wald"]}},
               p={{"null": {{"gamfit.wald": 0.5}}, "alt": {{"gamfit.wald": 0.001}}}},
               missing={{}}, errors={{}})
    print("RESULT " + json.dumps(rec), flush=True)
"""


def test_seeds_after_a_killed_rep_are_run_not_blamed(tmp_path: Path, monkeypatch: Any) -> None:
    # One rep that outlives the safety net must be the only rep charged with
    # it: the seeds queued behind it in the same chunk never started, and
    # recording them as timeouts would count them as rejections.
    from . import run as run_mod

    fake = tmp_path / "fake_worker.py"
    fake.write_text(FAKE_WORKER.format(hang=4))
    monkeypatch.setattr(run_mod, "WORKER", fake)
    cell = Cell("gaussian", 60, "smooth")
    plan = Plan("t", "", (cell,), reps=8, chunk=8, timeout_s=5.0, libs=("gamfit",))
    recs = run_mod.run_plan(plan, tmp_path / "out", jobs=1, memcap_mb=1e9, progress=False)
    status = {r["seed"]: r["status"] for r in recs}
    assert status == {0: "ok", 1: "ok", 2: "ok", 3: "ok", 4: "timeout", 5: "ok", 6: "ok", 7: "ok"}
    # A resume finds nothing left to run.
    again = run_mod.run_plan(plan, tmp_path / "out", jobs=1, memcap_mb=1e9, progress=False)
    assert len(again) == plan.reps


def test_pending_chunks_resume_only_missing_seeds() -> None:
    cell = Cell("gaussian", 60, "smooth")
    plan = Plan("t", "", (cell,), reps=10, chunk=4, timeout_s=1.0)
    assert [(s, e) for _, s, e in pending_chunks(plan, set())] == [(0, 4), (4, 8), (8, 10)]
    done = {(cell.key, s) for s in (0, 1, 5, 9)}
    assert [(s, e) for _, s, e in pending_chunks(plan, done)] == [(2, 5), (6, 9)]
    everything = {(cell.key, s) for s in range(10)}
    assert list(pending_chunks(plan, everything)) == []


def test_datasets_are_seeded_and_null_has_no_effect() -> None:
    a = make_data("poisson", 200, "smooth", 7, "null")
    b = make_data("poisson", 200, "smooth", 7, "null")
    np.testing.assert_array_equal(a["y"], b["y"])
    c = make_data("poisson", 200, "smooth", 8, "null")
    assert not np.array_equal(a["y"], c["y"])
    # Same-seed null and alternative use independent streams.
    alt = make_data("poisson", 200, "smooth", 7, "alt")
    assert not np.array_equal(a["x2"], alt["x2"])
    # The matched alternative has the same noncentrality at every n.
    assert np.isclose(alt_delta("gaussian", 200) ** 2 * 200, alt_delta("gaussian", 5000) ** 2 * 5000)


def _records(cell: str, null_p: list[float | None], surface: str = "gamfit.wald") -> list[dict[str, Any]]:
    lib, s = surface.split(".")
    out = []
    for seed, p in enumerate(null_p):
        rec: dict[str, Any] = {
            "key": cell,
            "seed": seed,
            "status": "ok",
            "expected_surfaces": {lib: [s]},
            "p": {"null": {}, "alt": {surface: 0.001}},
            "missing": {},
            "errors": {},
        }
        if p is None:
            rec["missing"][f"null.{lib}.{s}"] = "p_value=None"
        else:
            rec["p"]["null"][surface] = p
        out.append(rec)
    return out


def test_uniform_p_values_are_valid() -> None:
    p = list((np.arange(500) + 0.5) / 500)
    table = report.rows(_records("gaussian/n=200/smooth", p))
    assert report.anti_conservative(table) == []
    assert "| valid |" in report.calibration_table(_records("gaussian/n=200/smooth", p))


def test_anti_conservative_p_values_are_flagged() -> None:
    # Size 0.10 at the 0.05 level over 500 reps is ~5 MCSE above nominal.
    p = [0.02] * 50 + [0.5] * 450
    recs = _records("gaussian/n=200/smooth", p)
    flagged = report.anti_conservative(report.rows(recs))
    assert {a for _, a in flagged} == {0.05}
    assert "**ANTI-CONSERVATIVE** at 0.05" in report.calibration_table(recs)


def test_conservative_p_values_are_never_flagged() -> None:
    # A point mass at 1 (a boundary-shrunk term) is conservative, hence valid.
    recs = _records("gaussian/n=200/smooth", [1.0] * 500)
    assert report.anti_conservative(report.rows(recs)) == []


def test_missing_p_value_is_unusable_not_dropped() -> None:
    recs = _records("gaussian/n=200/smooth", [0.5, None, 0.7])
    (row,) = report.rows(recs)
    assert (row.reps, row.usable) == (3, 2)
    text = report.render(recs, {})
    assert "1 unusable" in text
    assert "null.gamfit.wald: 1x" in text
    none = _records("gaussian/n=200/smooth", [None, None])
    assert "**NO P-VALUE**" in report.calibration_table(none)


def test_unusable_reps_count_as_rejections() -> None:
    # 500 reps at p = 0.5 never reject. Replacing enough of them with reps that
    # produced no p-value must flag the row: those reps may have been the
    # rejections, so dropping them could hide an anti-conservative test.
    reps, a = 500, 0.01
    extra = report.reject_bound(reps, a, len(report.LEVELS)) + 1
    clean = _records("gaussian/n=200/smooth", [0.5] * reps)
    assert report.anti_conservative(report.rows(clean)) == []
    holed = _records("gaussian/n=200/smooth", [0.5] * (reps - extra) + [None] * extra)
    flagged = report.anti_conservative(report.rows(holed))
    assert 0.01 in {lvl for _, lvl in flagged}
    assert "**ANTI-CONSERVATIVE** at 0.01" in report.calibration_table(holed)


def test_tolerance_is_the_binomial_quantile() -> None:
    # The bound tightens as reps grow, in MCSE units it stays put.
    for reps in (200, 500, 2000):
        bound = report.reject_bound(reps, 0.05, 1)
        z = (bound / reps - 0.05) / np.sqrt(0.05 * 0.95 / reps)
        assert 2.5 < z < 3.6, (reps, z)


def test_pygam_has_no_surface_without_a_counterpart() -> None:
    assert expected_surfaces("pygam", "negbin", "smooth") == ()
    assert expected_surfaces("pygam", "gaussian", "ti") == ()
    assert expected_surfaces("pygam", "gaussian", "re") == ()
    assert expected_surfaces("pygam_gs", "poisson", "smooth") == ("wald",)


def test_docs_table_is_generated_from_the_committed_baseline() -> None:
    records, meta = report.load([BASELINE])
    page = (REPO_ROOT / "docs" / "pvalues.md").read_text()
    assert report.splice_docs(page, report.docs_block(records, meta)) == page, (
        "docs/pvalues.md is stale; regenerate it with "
        "python -m bench.pvalue_calibration.report "
        "bench/pvalue_calibration/baseline/quick --docs docs/pvalues.md"
    )
