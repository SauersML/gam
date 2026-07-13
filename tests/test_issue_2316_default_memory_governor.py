"""Installed-wheel regressions for the default memory governor (#2316/#2317).

These are the three small, public fits from #2316. They deliberately exercise
distinct dense-admission consumers (coefficient covariance, binomial Firth
initialization, and two-block location-scale Jacobians) so a zero default
budget cannot brick one path unnoticed while another happens to pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit

pytest.importorskip("gamfit._rust")


@pytest.fixture(scope="module")
def issue_2316_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + (0.1 + 0.5 * x) * rng.standard_normal(n)
    x2 = rng.uniform(0.0, 1.0, n)
    probability = 1.0 / (
        1.0 + np.exp(-(1.5 * np.sin(2.0 * np.pi * x) + 2.0 * (x2 - 0.5)))
    )
    binary_y = (rng.uniform(size=n) < probability).astype(int)
    gaussian = pd.DataFrame({"x": x, "y": y})
    binomial = pd.DataFrame({"x1": x, "x2": x2, "y": binary_y})
    return gaussian, binomial


def _assert_finite_training_prediction(model: object, frame: pd.DataFrame) -> None:
    prediction = np.asarray(model.predict(frame), dtype=float)  # type: ignore[attr-defined]
    assert prediction.shape[0] == len(frame)
    assert np.all(np.isfinite(prediction))


def test_default_governor_admits_small_gaussian_fit(
    issue_2316_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gaussian, _ = issue_2316_frames
    model = gamfit.fit(gaussian, "y ~ smooth(x)", family="gaussian")
    _assert_finite_training_prediction(model, gaussian)


def test_default_governor_admits_small_binomial_fit(
    issue_2316_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, binomial = issue_2316_frames
    model = gamfit.fit(
        binomial,
        "y ~ smooth(x1) + smooth(x2)",
        family="binomial",
    )
    _assert_finite_training_prediction(model, binomial)


def test_default_governor_admits_small_gaussian_location_scale_fit(
    issue_2316_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gaussian, _ = issue_2316_frames
    model = gamfit.fit(
        gaussian,
        "y ~ smooth(x)",
        family="gaussian",
        config={"noise_formula": "smooth(x)"},
    )
    _assert_finite_training_prediction(model, gaussian)
