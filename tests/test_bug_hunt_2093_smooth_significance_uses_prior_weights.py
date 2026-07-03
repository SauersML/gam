"""Regression for #2093: smooth_significance must reuse saved prior weights."""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
pytest.importorskip("gamfit._rust")


def _frame(weights: np.ndarray) -> dict[str, list[float]]:
    x = np.linspace(0.0, 1.0, weights.size)
    y = np.sin(7.0 * x) + 0.15 * np.cos(17.0 * x)
    return {"x": list(x), "y": list(y), "w": list(weights)}


def test_smooth_significance_uses_saved_weight_column() -> None:
    n = 96
    base_weights = np.ones(n)
    base_weights[n // 2 :] = 9.0
    model = gamfit.fit(_frame(base_weights), "y ~ s(x)", family="gaussian", weights="w")

    base = model.smooth_significance(_frame(base_weights))[0]
    swapped = model.smooth_significance(_frame(base_weights[::-1]))[0]

    assert abs(float(base["statistic_lr"]) - float(swapped["statistic_lr"])) > 1e-5


def test_smooth_significance_requires_saved_weight_column() -> None:
    weights = np.linspace(0.5, 2.0, 64)
    frame = _frame(weights)
    model = gamfit.fit(frame, "y ~ s(x)", family="gaussian", weights="w")

    with pytest.raises(Exception, match="w|weight"):
        model.smooth_significance({"x": frame["x"], "y": frame["y"]})
