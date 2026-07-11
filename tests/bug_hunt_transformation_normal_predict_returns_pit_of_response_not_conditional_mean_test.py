"""Bug hunt (#1612): ``gam predict`` on a conditional transformation-normal (CTM)
model must return the response-scale conditional mean ``E[Y|x]``, not the
probability-integral transform ``h(y|x)`` of the supplied outcome.

A CTM fits a smooth, strictly monotone map ``h(Y | x)`` such that the latent
``h(Y | x) ~ N(0, 1)``. The fitted model is a conditional distribution of ``Y``
given the covariates ``x`` alone; a *prediction* is therefore a statement about
``Y`` at a new ``x`` and must not require — or depend on — an observed outcome.
The correct response-scale point prediction is the conditional mean

    E[Y | x] = E_{Z ~ N(0,1)}[ h^{-1}(Z | x) ],

a function of ``x`` alone, obtained by numerically inverting the monotone
transform on a standard-normal quadrature and averaging.

The original predict path instead precomputed ``h(y | x)`` (the PIT of the
*supplied* response) and returned it as both ``linear_predictor`` and ``mean``.
That had two observable consequences this test pins down:

  (a) prediction wrongly REQUIRED the outcome column ``y`` — a covariate-only
      frame (the realistic prediction case) was rejected; and
  (b) at a fixed covariate ``x`` the predicted "mean" swept with ``y`` — feeding
      two different ``y`` values for the same ``x`` produced two different
      "means", which is impossible for a genuine ``E[Y|x]``.

This test drives the ``gam`` CLI end to end. It fits a CTM, then asserts:
  * predicting on a frame WITHOUT the response column succeeds and returns one
    finite prediction per row (covariate-only prediction works); and
  * the prediction is invariant to the value placed in the response column —
    two frames sharing the same ``x`` but carrying wildly different ``y`` yield
    byte-for-byte identical ``mean`` columns (the prediction is ``E[Y|x]``, not a
    transform of ``y``); and
  * the conditional mean lies inside the observed response range (a sane
    response-scale value, not a standardized z-score).

Before the fix the covariate-only predict aborts (the response column is
required) and the two-``y`` frames disagree; after the fix all three hold.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import subprocess
import tempfile
from typing import Any

pytest: Any = importlib.import_module("pytest")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAM_BIN = os.path.join(REPO_ROOT, "target", "release", "gam")


def _ensure_binary() -> str:
    if os.path.exists(GAM_BIN):
        return GAM_BIN
    build = subprocess.run(
        ["cargo", "build", "--release", "--bin", "gam"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CARGO_PROFILE_DEV_DEBUG": "0"},
    )
    if build.returncode != 0 or not os.path.exists(GAM_BIN):
        pytest.skip(f"could not build release gam binary:\n{build.stderr[-2000:]}")
    return GAM_BIN


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [GAM_BIN, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )


def _write_csv(path: str, header: list[str], rows: list[list[float]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.17g}" for v in row])


def _read_col(path: str, name: str) -> list[float]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None and name in reader.fieldnames, (
            f"prediction CSV {path} is missing column '{name}'; "
            f"has {reader.fieldnames}"
        )
        return [float(row[name]) for row in reader]


def test_transformation_normal_predict_returns_conditional_mean_not_pit() -> None:
    _ensure_binary()

    # ---- deterministic synthetic bounded, x-dependent data ------------------
    # y in (0,1), location moving with x so E[Y|x] genuinely varies; a plain
    # LCG keeps the data identical on every platform without an RNG dependency.
    state = 0x2545F4914F6CDD1D
    def nextf() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) / float(1 << 53))

    n = 300
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(n):
        x = nextf()
        # logistic location shift in x, plus bounded noise; clamp into (0.02,0.98).
        loc = 0.25 + 0.5 * x
        y = min(0.98, max(0.02, loc + 0.12 * (nextf() - 0.5)))
        xs.append(x)
        ys.append(y)

    y_min, y_max = min(ys), max(ys)

    with tempfile.TemporaryDirectory(prefix="gam_ctm_1612_") as tmp:
        train_csv = os.path.join(tmp, "train.csv")
        _write_csv(train_csv, ["x", "y"], [[xs[i], ys[i]] for i in range(n)])

        model = os.path.join(tmp, "ctm.gam")
        fit = _run(["fit", train_csv, "y ~ s(x, k=6)", "--transformation-normal",
                    "--out", model])
        assert fit.returncode == 0, (
            f"transformation-normal fit failed:\n{fit.stdout}\n{fit.stderr}"
        )

        # Held-out covariate grid (the rows we actually predict at).
        grid_x = [0.1, 0.3, 0.5, 0.7, 0.9]

        # (1) Covariate-only frame: NO response column at all. This is the
        # realistic prediction case and must succeed.
        cov_only = os.path.join(tmp, "cov_only.csv")
        _write_csv(cov_only, ["x"], [[gx] for gx in grid_x])
        pred_cov = os.path.join(tmp, "pred_cov.csv")
        r_cov = _run(["predict", model, cov_only, "--out", pred_cov])
        assert r_cov.returncode == 0, (
            "covariate-only prediction (no response column) must succeed for a "
            f"CTM, but failed:\n{r_cov.stdout}\n{r_cov.stderr}"
        )
        mean_cov = _read_col(pred_cov, "mean")
        assert len(mean_cov) == len(grid_x)
        assert all(math.isfinite(m) for m in mean_cov), f"non-finite means: {mean_cov}"

        # (2) Two frames sharing the same x but carrying very different y. A
        # genuine E[Y|x] is invariant to y; the buggy PIT path swept with y.
        frame_lo = os.path.join(tmp, "frame_lo.csv")
        frame_hi = os.path.join(tmp, "frame_hi.csv")
        _write_csv(frame_lo, ["x", "y"], [[gx, 0.05] for gx in grid_x])
        _write_csv(frame_hi, ["x", "y"], [[gx, 0.95] for gx in grid_x])
        pred_lo = os.path.join(tmp, "pred_lo.csv")
        pred_hi = os.path.join(tmp, "pred_hi.csv")
        r_lo = _run(["predict", model, frame_lo, "--out", pred_lo])
        r_hi = _run(["predict", model, frame_hi, "--out", pred_hi])
        assert r_lo.returncode == 0, f"predict (y=lo) failed:\n{r_lo.stderr}"
        assert r_hi.returncode == 0, f"predict (y=hi) failed:\n{r_hi.stderr}"
        mean_lo = _read_col(pred_lo, "mean")
        mean_hi = _read_col(pred_hi, "mean")

        for a, b in zip(mean_lo, mean_hi):
            assert abs(a - b) <= 1e-9, (
                "CTM prediction mean depends on the supplied response y: "
                f"y=0.05 -> {a}, y=0.95 -> {b} at the same x. A response-scale "
                "E[Y|x] must be a function of x alone."
            )

        # The covariate-only mean must match the y-bearing frames too (same x).
        for a, b in zip(mean_cov, mean_lo):
            assert abs(a - b) <= 1e-9, (
                f"covariate-only mean {a} != mean with y column {b} at same x"
            )

        # (3) The labelled-data transform is an explicit, separate command.
        # It must move monotonically with the observed response at fixed x;
        # accepting predict's y-independent `eta`/`mean` columns here was the
        # large-scale marginal-slope benchmark's semantic corruption (#979).
        score_lo_path = os.path.join(tmp, "score_lo.csv")
        score_hi_path = os.path.join(tmp, "score_hi.csv")
        score_lo_run = _run(
            ["transformation-score", model, frame_lo, "--out", score_lo_path]
        )
        score_hi_run = _run(
            ["transformation-score", model, frame_hi, "--out", score_hi_path]
        )
        assert score_lo_run.returncode == 0, score_lo_run.stderr
        assert score_hi_run.returncode == 0, score_hi_run.stderr
        score_lo = _read_col(score_lo_path, "score")
        score_hi = _read_col(score_hi_path, "score")
        assert all(math.isfinite(value) for value in [*score_lo, *score_hi])
        assert all(high > low for low, high in zip(score_lo, score_hi)), (
            score_lo,
            score_hi,
        )

        # (4) The conditional mean is a response-scale value inside the observed
        # range, not a standardized z-score (which would routinely exceed [0,1]).
        lo_bound = y_min - 0.1 * (y_max - y_min)
        hi_bound = y_max + 0.1 * (y_max - y_min)
        for m in mean_cov:
            assert lo_bound <= m <= hi_bound, (
                f"conditional mean {m} outside the plausible response range "
                f"[{lo_bound:.3f}, {hi_bound:.3f}]; looks like a z-score, not E[Y|x]"
            )

        # E[Y|x] should increase with x given the x-dependent location.
        assert mean_cov[-1] > mean_cov[0], (
            "E[Y|x] should track the x-dependent location (increasing in x), "
            f"got {mean_cov}"
        )
