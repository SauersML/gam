"""Smoke + contract tests for the gamfit-vs-pyGAM benchmark harness.

The end-to-end test runs the real ``smoke`` plan (n=300, three families, all
three libraries, fresh subprocess per rep) and checks that every rep produced
a complete record and that the report carries the ratio columns. The other
tests pin the verdict rules on hand-built records, so a report that stopped
marking a loss would fail here rather than quietly flattering gamfit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .plans import HOST_WORKERS, PLANS, Cell
from .report import paired_verdict, ratio_verdict, render
from .run import THREAD_ENV, run_batch, run_rep, thread_env
from .worker import COUNT_FAMILIES, COUNT_SLOPE, EXPOSURE_RATE, make_data, supports

BENCH_DIR = Path(__file__).resolve().parent.parent

REQUIRED_OK_FIELDS = (
    "import_cpu_s",
    "fit_s",
    "fit_cpu_s",
    "pred_cpu_s",
    "interval_cpu_s",
    "peak_rss_mb",
    "peak_tree_rss_mb",
    "peak_threads",
    "rmse_mu",
    "deviance",
    "logscore",
    "coverage",
    "edf",
    "lib_version",
)


def test_smoke_plan_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pygam_compare.run",
            "smoke",
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
    plan = PLANS["smoke"]
    assert len(records) == len(plan.cells) * plan.reps * len(plan.libs)
    for rec in records:
        assert rec["status"] == "ok", rec
        missing = [f for f in REQUIRED_OK_FIELDS if rec.get(f) is None]
        assert not missing, (rec["lib"], rec["family"], missing)
        assert 0.0 <= rec["coverage"] <= 1.0
        # Thread env pinned: gamfit's Rayon pool and the BLAS pools are one
        # thread each, so the process stays within a handful of threads.
        assert rec["peak_threads"] <= 8, rec
    meta = json.loads((out / "meta.json").read_text())
    assert meta["thread_env"]["RAYON_NUM_THREADS"] == "1"
    assert "safety net" in meta["safety_net"]
    report = (out / "report.md").read_text()
    for header in ("## Status", "## fit CPU", "vs pygam_gs", "## Losses"):
        assert header in report
    for family in ("gaussian", "binomial", "poisson"):
        assert f"{family} n=300 p1" in report


def test_timeout_is_recorded_as_status(tmp_path: Path) -> None:
    rec = run_rep("gamfit", Cell("gaussian", 300, "p1"), 0, 0.0, 1e9, str(tmp_path))
    assert rec["status"] == "timeout"
    assert rec["lib"] == "gamfit"


def _rec(lib: str, seed: int, status: str = "ok", **metrics: Any) -> dict[str, Any]:
    return dict(
        lib=lib,
        family="gaussian",
        n=100,
        design="p1",
        seed=seed,
        status=status,
        **metrics,
    )


def test_ratio_above_one_is_a_loss() -> None:
    g = [_rec("gamfit", 0, fit_cpu_s=2.0)]
    c = [_rec("pygam", 0, fit_cpu_s=1.0)]
    v = ratio_verdict(g, c, "fit_cpu_s")
    assert v.loss and "2.00x **LOSS**" in v.text
    assert ratio_verdict(c, g, "fit_cpu_s").win


def test_paired_accuracy_verdicts() -> None:
    # Consistently worse by far more than 2 SE -> LOSS.
    g = [_rec("gamfit", s, rmse_mu=1.0 + 0.01 * s) for s in range(3)]
    c = [_rec("pygam", s, rmse_mu=0.5) for s in range(3)]
    assert paired_verdict(g, c, "rmse_mu").loss
    # Worse on average but inside the noise -> not a loss, but labelled.
    g = [_rec("gamfit", s, rmse_mu=v) for s, v in enumerate((1.0, 0.0, 1.1))]
    c = [_rec("pygam", s, rmse_mu=0.5) for s in range(3)]
    v = paired_verdict(g, c, "rmse_mu")
    assert not v.loss and v.text.startswith("worse n.s.")
    # Coverage is judged on distance from the nominal 0.95.
    g = [_rec("gamfit", s, coverage=0.80) for s in range(3)]
    c = [_rec("pygam", s, coverage=0.94 + 0.001 * s) for s in range(3)]
    assert paired_verdict(g, c, "coverage").loss
    # A metric gamfit fails to report where pyGAM does is a loss, not a gap.
    g = [_rec("gamfit", 0)]
    c = [_rec("pygam", 0, logscore=1.0)]
    assert paired_verdict(g, c, "logscore").text == "**LOSS(missing)**"


def test_count_plans_run_pygam_only_where_it_has_the_family() -> None:
    for name in ("count_small", "count_1e4", "count_1e5"):
        families = {cell.family for cell in PLANS[name].cells}
        assert families == set(COUNT_FAMILIES), name
    assert supports("gamfit", "negbin") and supports("gamfit", "tweedie")
    assert not supports("pygam", "negbin") and not supports("pygam_gs", "tweedie")
    assert supports("pygam_gs", "poisson_exposure")


def test_exposure_draw_carries_its_offset() -> None:
    X, y, mu, weights, offset = make_data(500, "p1", "poisson_exposure", 0)
    assert weights is None
    assert offset is not None and offset.shape == y.shape
    rate = mu / np.exp(offset)
    # The rate is the level times the smooth; the exposure is all in the offset.
    assert np.allclose(
        rate, EXPOSURE_RATE * np.exp(COUNT_SLOPE * np.sin(2 * np.pi * X[:, 0]))
    )
    for family in ("poisson_lo", "negbin", "tweedie"):
        assert make_data(50, "p1", family, 0)[3:] == (None, None)


def test_timeout_is_a_listed_loss_not_a_skip() -> None:
    records = [
        _rec("gamfit", 0, status="timeout"),
        _rec("pygam", 0, fit_cpu_s=1.0, rmse_mu=0.1),
        _rec("pygam_gs", 0, fit_cpu_s=3.0, rmse_mu=0.1),
    ]
    text = render(records)
    assert "0/1 ok, 1 timeout" in text
    losses = text.split("## Losses")[1]
    assert "status vs pygam " in losses and "status vs pygam_gs" in losses


def test_n_predict_cells_are_reported_apart() -> None:
    # Two cells that differ only in n_predict are separate report rows, and a
    # post-fit metric gets its own table only once some rep measured it.
    records = [
        dict(_rec(lib, 0, pred_cpu_s=t), n_predict=m)
        for m, t in ((100, 0.01), (1_000_000, 1.0))
        for lib in ("gamfit", "pygam_gs")
    ]
    text = render(records)
    assert "gaussian n=100 p1 n_predict=100" in text
    assert "gaussian n=100 p1 n_predict=1e+06" in text
    assert "## partial dependence CPU" not in text
    records[0]["pd_cpu_s"] = 0.5
    records[1]["pd_cpu_s"] = 0.25
    assert "## partial dependence CPU" in render(records)
    assert PLANS["postfit"].postfit
    assert all(cell.n_predict is not None for cell in PLANS["postfit"].cells)


def test_thread_settings_reach_every_pool_variable() -> None:
    assert thread_env(1) == THREAD_ENV
    assert thread_env(4) == {k: "4" for k in THREAD_ENV}
    # auto removes the variables; it does not set them to some guessed width.
    assert thread_env(None) == {}
    # The default cell keeps the name every committed baseline uses.
    assert Cell("gaussian", 300, "p1").key == "gaussian/n=300/p1"
    assert (
        Cell("gaussian", 300, "p1", threads=None, concurrency=4).key
        == "gaussian/n=300/p1 threads=auto x4"
    )


def test_scaling_plans_cover_the_sweep() -> None:
    cells = PLANS["threads"].cells
    assert {c.n for c in cells} == {10_000, 100_000, 1_000_000}
    assert {c.threads for c in cells} == {1, 2, 4, 8, None}
    assert {c.design for c in cells} == {"p5", "p20", "te"}
    assert {c.family for c in cells} == {"gaussian", "binomial"}
    fanout = PLANS["oversubscribe"].cells
    assert {c.concurrency for c in fanout} == {1, HOST_WORKERS}
    assert {c.threads for c in fanout} == {1, None}
    # Every fan-out shape has its one-process reference at the same threads.
    for c in fanout:
        assert Cell(c.family, c.n, c.design, c.threads, 1) in fanout


def test_batch_records_every_process(tmp_path: Path) -> None:
    cell = Cell("gaussian", 300, "p1", threads=None, concurrency=2)
    recs = run_batch("gamfit", cell, 0, 0.0, 1e9, str(tmp_path))
    assert [r["slot"] for r in recs] == [0, 1]
    assert all(r["status"] == "timeout" for r in recs)
    assert all(r["threads"] is None and r["concurrency"] == 2 for r in recs)
    assert recs[0]["batch_wall_s"] == recs[1]["batch_wall_s"] > 0.0


def _threaded(threads: Any, concurrency: int, seed: int, **metrics: Any) -> dict:
    return _rec("gamfit", seed, threads=threads, concurrency=concurrency, **metrics)


def test_report_renders_scaling_and_fanout() -> None:
    records = [
        _threaded(1, 1, 0, fit_s=8.0, proc_wall_s=10.0),
        _threaded(4, 1, 0, fit_s=4.0, proc_wall_s=6.0),
        _threaded(None, 1, 0, fit_s=2.0, proc_wall_s=4.0),
    ]
    # Four processes at once, each as slow as one alone: throughput 4x.
    records += [
        dict(_threaded(1, 4, 0, fit_s=8.0, proc_wall_s=10.0), batch_wall_s=10.0)
        for _ in range(4)
    ]
    # Four at the default width that take 8x as long: a 0.5x collapse.
    records += [
        dict(_threaded(None, 4, 0, fit_s=30.0, proc_wall_s=32.0), batch_wall_s=32.0)
        for _ in range(4)
    ]
    text = render(records)
    scaling = text.split("## Thread scaling")[1].split("## Process fan-out")[0]
    assert "threads=1 | threads=4 | threads=auto" in scaling
    assert "| gaussian n=100 p1 | 8 s (1.00x) | 4 s (2.00x) | 2 s (4.00x) |" in scaling
    fanout = text.split("## Process fan-out")[1].split("## Status")[0]
    assert "| gaussian n=100 p1 x4 |" in fanout and "| 4x |" in fanout
    assert "| gaussian n=100 p1 threads=auto x4 |" in fanout and "| 0.5x |" in fanout
    # The per-setting cells stay separate rows elsewhere in the report.
    assert "gaussian n=100 p1 threads=4 |" in text.split("## Status")[1]


def test_single_core_records_render_without_scaling_sections() -> None:
    text = render([_rec("gamfit", 0, fit_s=1.0), _rec("pygam", 0, fit_s=2.0)])
    assert "## Thread scaling" not in text and "## Process fan-out" not in text
    assert "| gaussian n=100 p1 |" in text
