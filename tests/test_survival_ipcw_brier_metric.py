"""Regression tests for #1563: the survival metric reported under ``brier`` must
be a genuine (integrated, IPCW) Brier score — the Graf et al. (1999) estimator
that scikit-survival / pec / ``survival::brier`` report — and *not* the hazard
quadratic score ``0.5∫h² − δ·h(T)`` that the field used to (mis)report.

The oracle below is a deliberately independent, textbook reimplementation of the
Graf integrated Brier score in plain Python, so a regression in the Rust path
cannot hide behind a shared implementation.
"""

import importlib
from typing import Any, Callable, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit._rust as _rust


def _km_censoring(time: list[float], event: list[float]) -> Callable[[float], float]:
    """Right-continuous Kaplan–Meier of the censoring survival ``G(t)=P(C>t)``.

    The censoring "event" is ``event == 0``; deaths censor the reversed process.
    """
    rows = sorted((t, e) for t, e in zip(time, event) if np.isfinite(t) and t > 0.0)
    steps: list[tuple[float, float]] = []
    at_risk = float(len(rows))
    surv = 1.0
    i = 0
    while i < len(rows):
        t = rows[i][0]
        j = i
        deaths = 0
        while j < len(rows) and rows[j][0] == t:
            deaths += 1 if rows[j][1] <= 0.5 else 0  # a censoring is an "event"
            j += 1
        if deaths > 0 and at_risk > 0.0:
            surv *= max((at_risk - deaths) / at_risk, 0.0)
            steps.append((t, surv))
        at_risk -= j - i
        i = j

    def g(t: float) -> float:
        s = 1.0
        for tt, ss in steps:
            if tt <= t:
                s = ss
            else:
                break
        return s

    return g


def _graf_bs(
    s_col: np.ndarray,
    time: list[float],
    event: list[float],
    tau: float,
    g: Callable[[float], float],
) -> float | None:
    """Graf per-time Brier ``BS(tau)`` — sample mean over all valid subjects."""
    n = 0
    acc = 0.0
    for i in range(len(time)):
        if not np.isfinite(time[i]) or time[i] <= 0.0:
            continue
        n += 1
        if time[i] <= tau and event[i] > 0.5:
            gi = g(time[i])
            if gi > 0.0:
                acc += (s_col[i] ** 2) / gi
        elif time[i] > tau:
            gt = g(tau)
            if gt > 0.0:
                acc += ((1.0 - s_col[i]) ** 2) / gt
        # else: censored at/before tau -> contributes 0
    return acc / n if n > 0 else None


def _graf_ibs(
    surv: np.ndarray,
    time: list[float],
    event: list[float],
    grid: list[float],
    horizon: float,
    g: Callable[[float], float],
) -> float | None:
    pts: list[tuple[float, float]] = []
    for k, t in enumerate(grid):
        if t > horizon:
            break
        bs = _graf_bs(surv[:, k], time, event, t, g)
        if bs is not None:
            pts.append((t, bs))
    if len(pts) < 2:
        return None
    span = pts[-1][0] - pts[0][0]
    integral = sum(
        0.5 * (pts[a][1] + pts[a + 1][1]) * (pts[a + 1][0] - pts[a][0])
        for a in range(len(pts) - 1)
    )
    return integral / span


def _hazard_quadratic(
    surv: np.ndarray,
    time: list[float],
    event: list[float],
    grid: list[float],
    eps: float = 1e-12,
) -> float:
    """The *old* per-subject hazard quadratic score, replicated to prove the new
    ``brier`` is a different quantity and ``hazard_quadratic_score`` matches this.
    """
    s = surv.copy()
    for r in range(s.shape[0]):
        s[r, 0] = 1.0
        prev = 1.0
        for c in range(s.shape[1]):
            s[r, c] = min(max(s[r, c], eps), prev)
            prev = s[r, c]
    cumhaz = -np.log(np.clip(s, eps, 1.0))
    dt = np.diff(grid)
    haz = np.maximum(np.diff(cumhaz, axis=1) / dt, 0.0)
    losses = []
    for r, t in enumerate(time):
        j = int(np.searchsorted(grid, t, side="left"))
        if j >= len(grid):
            j = len(grid) - 1
        idx = max(j - 1, 0)
        if abs(grid[j] - t) <= 1e-12:
            h_z = haz[r, idx]
            h2 = float(np.sum(haz[r, :j] ** 2 * dt[:j]))
        else:
            elapsed = t - grid[idx]
            h_z = haz[r, idx]
            h2 = float(np.sum(haz[r, :idx] ** 2 * dt[:idx])) + h_z * h_z * elapsed
        losses.append(0.5 * h2 - (h_z if event[r] > 0.5 else 0.0))
    return float(np.mean(losses))


def _valid_curves(col: list[float], grid_len: int) -> np.ndarray:
    """Per-subject curves that are already monotone, ≤1, first column 1 — so the
    FFI's clamp/monotonize step is a no-op and the oracle sees identical input."""
    surv = np.zeros((len(col), grid_len), dtype=float)
    for i, c in enumerate(col):
        surv[i, 0] = 1.0
        for k in range(1, grid_len):
            # gentle monotone decay toward the subject's plateau c
            surv[i, k] = max(c, 1.0 - (1.0 - c) * k / (grid_len - 1))
    return surv


def test_brier_field_is_integrated_ipcw_brier_not_hazard_quadratic() -> None:
    time = [2.0, 8.0, 10.0, 3.0, 6.0, 1.5]
    event = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    grid = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0]
    col = [0.3, 0.7, 0.6, 0.2, 0.5, 0.4]
    surv = _valid_curves(col, len(grid))

    out = _rust.survival_lifted_metrics_from_predictions(
        time, event, grid, surv, None, 1e-12
    )

    # The field exists and is honest about what it is.
    assert "brier" in out and out["brier"] is not None
    assert "hazard_quadratic_score" in out and out["hazard_quadratic_score"] is not None

    g = _km_censoring(time, event)
    horizon = max(t for t in time if np.isfinite(t))
    expected_ibs = _graf_ibs(surv, time, event, grid, horizon, g)
    assert expected_ibs is not None
    assert out["brier"] == pytest.approx(expected_ibs, rel=1e-9, abs=1e-12), (
        f"brier={out['brier']} expected integrated IPCW Brier={expected_ibs}"
    )

    # The old (mislabeled) quantity is preserved under its honest name...
    expected_hq = _hazard_quadratic(surv, time, event, grid)
    assert out["hazard_quadratic_score"] == pytest.approx(expected_hq, rel=1e-9, abs=1e-12)

    # ...and the two are genuinely different numbers — the root-cause guard.
    assert abs(out["brier"] - out["hazard_quadratic_score"]) > 1e-6, (
        "brier must no longer be the hazard quadratic score"
    )


def test_brier_no_censoring_reduces_to_integrated_plain_brier() -> None:
    # With no censoring G(t) ≡ 1, so the IPCW Brier is the ordinary integrated
    # Brier of S(t) against the alive-indicator I(T_i > t).
    time = [2.0, 8.0, 5.0, 3.0]
    event = [1.0, 1.0, 1.0, 1.0]  # every subject has an event -> no censoring
    grid = [0.0, 1.0, 2.0, 4.0]
    col = [0.3, 0.7, 0.6, 0.2]
    surv = _valid_curves(col, len(grid))

    out = _rust.survival_lifted_metrics_from_predictions(
        time, event, grid, surv, None, 1e-12
    )

    # Plain integrated Brier with G ≡ 1.
    pts = []
    for k, t in enumerate(grid):
        bs = np.mean(
            [
                (surv[i, k] - (1.0 if time[i] > t else 0.0)) ** 2
                for i in range(len(time))
            ]
        )
        pts.append((t, bs))
    span = pts[-1][0] - pts[0][0]
    expected = (
        sum(
            0.5 * (pts[a][1] + pts[a + 1][1]) * (pts[a + 1][0] - pts[a][0])
            for a in range(len(pts) - 1)
        )
        / span
    )
    assert out["brier"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_lifted_brier_is_relative_ipcw_brier_skill() -> None:
    time = [2.0, 8.0, 10.0, 3.0, 6.0, 1.5]
    event = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    grid = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0]
    col = [0.3, 0.7, 0.6, 0.2, 0.5, 0.4]
    surv = _valid_curves(col, len(grid))
    # A null curve: a single shared (population) survival, broadcast to all rows.
    null_col = [0.5] * len(col)
    null_surv = _valid_curves(null_col, len(grid))

    out = _rust.survival_lifted_metrics_from_predictions(
        time, event, grid, surv, null_surv, 1e-12
    )

    g = _km_censoring(time, event)
    horizon = max(t for t in time if np.isfinite(t))
    model_ibs = _graf_ibs(surv, time, event, grid, horizon, g)
    null_ibs = _graf_ibs(null_surv, time, event, grid, horizon, g)
    assert model_ibs is not None and null_ibs is not None
    expected = (null_ibs - model_ibs) / max(abs(null_ibs), 1e-12)
    assert out["lifted_brier"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert out["lifted_hazard_quadratic_score"] is not None


def test_score_grid_is_data_driven_and_spans_the_observed_range() -> None:
    # On a small (sub-unit) time scale the old fixed {0,1,2,5,10,median} grid ran
    # an order of magnitude past the data; the quantile grid must track the data.
    rng = np.random.default_rng(1563)
    times = (rng.uniform(0.01, 0.5, size=400)).tolist()
    grid = np.asarray(_rust.survival_score_grid_from_times(times), dtype=float)

    assert grid.ndim == 1 and grid.shape[0] >= 2
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) > 0.0), "grid must be strictly increasing"
    # Top of the grid is the observed max, not a magic constant far past it.
    assert grid[-1] == pytest.approx(max(times), rel=0, abs=1e-12)
    assert grid[-1] < 1.0, "grid must not run past the sub-unit observed range"
    # Interior knots track the data (median lands inside the grid span).
    assert grid[1] < float(np.median(times)) < grid[-1]
