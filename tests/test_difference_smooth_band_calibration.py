"""Calibration of simultaneous difference-smooth bands and the no-difference test.

The full seeded study lives in ``bench/pvalue_calibration/pv-bands``; these
are its fast regression slices.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

FORMULA = "y ~ g + s(x, by=g, k=10)"
GRID = 50


def _two_groups(family: str, n: int, seed: int, difference: bool) -> dict[str, Any]:
    """Unequal groups (A is the 30% minority) sharing a curve unless ``difference``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = np.where(rng.uniform(size=n) < 0.3, "A", "B")
    eta = 0.2 + 0.8 * np.sin(2.0 * np.pi * x)
    if difference:
        eta = eta + np.where(g == "B", 0.6 * np.cos(np.pi * x), 0.0)
    if family == "gaussian":
        y = eta + 0.5 * rng.standard_normal(n)
    else:
        y = rng.binomial(1, 0.5 * (1.0 + np.tanh(0.5 * eta))).astype(float)
    return {"x": x, "g": g, "y": y}


def _rows(model: Any, *, level: float = 0.95, simultaneous: bool = True) -> list[dict[str, Any]]:
    return model.difference_smooth(
        view="x",
        group="g",
        pairs=[("B", "A")],
        n=GRID,
        level=level,
        simultaneous=simultaneous,
        return_type="list",
    )


def _column(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=float)


def test_band_uses_the_published_covariance_when_the_correction_is_unavailable() -> None:
    # This binomial fit certifies a smoothing parameter at its rail, so its
    # smoothing correction is typed unavailable and it publishes the
    # conditional covariance. The band used to refuse the fit outright.
    model = gamfit.fit(_two_groups("binomial", 400, 304_001, False), FORMULA, family="binomial")
    grid = np.linspace(0.0, 1.0, GRID)
    design_a = model.design_matrix({"x": grid, "g": np.array(["A"] * GRID)})
    design_b = model.design_matrix({"x": grid, "g": np.array(["B"] * GRID)})
    assert design_b.covariance_smoothing_corrected is None

    rows = _rows(model)
    assert {row["covariance_kind"] for row in rows} == {"conditional"}
    assert not any(row["covariance_corrected"] for row in rows)
    contrast = np.asarray(design_b.matrix) - np.asarray(design_a.matrix)
    conditional = np.asarray(design_b.covariance_conditional)
    expected_se = np.sqrt(np.einsum("ij,jk,ik->i", contrast, conditional, contrast))
    np.testing.assert_allclose(_column(rows, "se"), expected_se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_column(rows, "diff"), contrast @ np.asarray(design_b.coefficients), atol=1e-10)


def test_no_difference_p_value_is_the_simultaneous_band_test() -> None:
    model = gamfit.fit(_two_groups("gaussian", 300, 11, True), FORMULA)
    rows = _rows(model)
    p_values = {row["p_value"] for row in rows}
    assert len(p_values) == 1
    (p_value,) = p_values
    assert 0.0 < p_value <= 1.0
    for level in (0.90, 0.95, 0.99):
        band = _rows(model, level=level)
        excludes_zero = bool(np.any((_column(band, "lower") > 0.0) | (_column(band, "upper") < 0.0)))
        assert excludes_zero == (p_value <= 1.0 - level), (level, p_value)
    assert all(row["p_value"] is None for row in _rows(model, simultaneous=False))


def test_gaussian_simultaneous_band_covers_the_whole_curve_and_holds_its_size() -> None:
    replicates = 60
    level = 0.95
    covered = []
    pointwise_covered = []
    rejected = []
    for replicate in range(replicates):
        alternative = gamfit.fit(_two_groups("gaussian", 200, 40_000 + replicate, True), FORMULA)
        band = _rows(alternative, level=level)
        truth = 0.6 * np.cos(np.pi * _column(band, "x"))
        covered.append(np.all((_column(band, "lower") <= truth) & (truth <= _column(band, "upper"))))
        pointwise = _rows(alternative, level=level, simultaneous=False)
        pointwise_covered.append(
            np.all((_column(pointwise, "lower") <= truth) & (truth <= _column(pointwise, "upper")))
        )
        null = gamfit.fit(_two_groups("gaussian", 200, 50_000 + replicate, False), FORMULA)
        rejected.append(_rows(null, level=level)[0]["p_value"] <= 1.0 - level)
    mcse = math.sqrt(level * (1.0 - level) / replicates)
    coverage = float(np.mean(covered))
    size = float(np.mean(rejected))
    assert abs(coverage - level) <= 2.0 * mcse, (coverage, mcse)
    assert abs(size - (1.0 - level)) <= 2.0 * mcse, (size, mcse)
    # The pointwise band is what G2 measured: it misses somewhere on the curve
    # far more often than 5% of the time.
    assert float(np.mean(pointwise_covered)) < level - 2.0 * mcse
