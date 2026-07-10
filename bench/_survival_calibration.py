"""Benchmark-only survival calibration through the external reference tool.

These helpers deliberately live under ``bench/`` rather than in gamfit or its
PyO3 module.  They turn scalar risk scores into reference survival curves for
cross-library scoring; they are not part of the supported GAM engine surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _survival_arrays(
    train_times: Any,
    train_events: Any,
    grid: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(train_times, dtype=np.float64).reshape(-1)
    events = np.asarray(train_events, dtype=np.float64).reshape(-1)
    score_grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if times.shape != events.shape:
        raise ValueError(
            "survival calibration length mismatch: "
            f"times={times.size} events={events.size}"
        )
    if times.size == 0:
        raise ValueError("survival calibration requires training observations")
    if not np.all(np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("survival calibration times must be finite and positive")
    if not np.all(np.isfinite(events)) or np.any((events < 0.0) | (events > 1.0)):
        raise ValueError("survival calibration events must lie in [0, 1]")
    if not np.all(np.isfinite(score_grid)) or np.any(score_grid < 0.0):
        raise ValueError("survival calibration grid must be finite and non-negative")
    if score_grid.size > 1 and np.any(np.diff(score_grid) < 0.0):
        raise ValueError("survival calibration grid must be non-decreasing")
    return times, events, score_grid


def kaplan_meier_curve(
    train_times: Any,
    train_events: Any,
    grid: Any,
) -> np.ndarray:
    """Evaluate the lifelines Kaplan–Meier estimate on ``grid``."""

    from lifelines import KaplanMeierFitter

    times, events, score_grid = _survival_arrays(train_times, train_events, grid)
    fitted = KaplanMeierFitter().fit(times, event_observed=events > 0.5)
    return np.asarray(fitted.predict(score_grid), dtype=np.float64).reshape(-1)


def calibrated_survival_matrix(
    train_times: Any,
    train_events: Any,
    train_risk: Any,
    test_risk: Any,
    grid: Any,
) -> np.ndarray:
    """Fit a one-covariate lifelines Cox model and predict survival curves."""

    from lifelines import CoxPHFitter

    times, events, score_grid = _survival_arrays(train_times, train_events, grid)
    train_score = np.asarray(train_risk, dtype=np.float64).reshape(-1)
    test_score = np.asarray(test_risk, dtype=np.float64).reshape(-1)
    if train_score.shape != times.shape:
        raise ValueError(
            "survival calibration length mismatch: "
            f"times={times.size} risk={train_score.size}"
        )
    if not np.all(np.isfinite(train_score)) or not np.all(np.isfinite(test_score)):
        raise ValueError("survival calibration risk scores must be finite")

    # A constant score contains no covariate information. Its Cox prediction is
    # exactly the intercept-free population curve, so use the KM estimate
    # directly instead of asking a rank-deficient Cox fit to discover that fact.
    if train_score.size < 2 or np.all(train_score == train_score[0]):
        curve = kaplan_meier_curve(times, events, score_grid)
        return np.broadcast_to(curve, (test_score.size, curve.size)).copy()

    frame = pd.DataFrame(
        {"time": times, "event": events > 0.5, "risk": train_score}
    )
    fitted = CoxPHFitter().fit(
        frame,
        duration_col="time",
        event_col="event",
        formula="risk",
    )
    predicted = fitted.predict_survival_function(
        pd.DataFrame({"risk": test_score}),
        times=score_grid,
    )
    return np.asarray(predicted, dtype=np.float64).T


__all__ = ["calibrated_survival_matrix", "kaplan_meier_curve"]
