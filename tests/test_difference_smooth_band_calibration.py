"""Calibration of simultaneous difference-smooth bands.

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
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
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


@pytest.mark.parametrize(
    ("family", "seed", "publishes_corrected"),
    [
        ("binomial", 304_001, True),
        # This fit types its smoothing correction unavailable, so it publishes
        # the conditional covariance; the band used to refuse it outright.
        ("poisson", 204_000, False),
    ],
)
def test_band_is_priced_from_the_published_covariance(
    family: str, seed: int, publishes_corrected: bool
) -> None:
    # The band's standard errors and row correlation come from the covariance
    # the fit publishes: smoothing-corrected when the fit carries it,
    # conditional when the correction is typed unavailable.
    model = gamfit.fit(_two_groups(family, 400, seed, True), FORMULA, family=family)
    rows = _rows(model)
    grid = _column(rows, "x")
    design_a = model.design_matrix({"x": grid, "g": np.array(["A"] * GRID)})
    design_b = model.design_matrix({"x": grid, "g": np.array(["B"] * GRID)})
    corrected = design_b.covariance_smoothing_corrected
    assert (corrected is not None) == publishes_corrected
    published = np.asarray(design_b.covariance_conditional if corrected is None else corrected)

    expected_kind = "conditional" if corrected is None else "smoothing-corrected"
    assert {row["covariance_kind"] for row in rows} == {expected_kind}
    assert {row["covariance_corrected"] for row in rows} == {corrected is not None}
    contrast = np.asarray(design_b.matrix) - np.asarray(design_a.matrix)
    expected_se = np.sqrt(np.einsum("ij,jk,ik->i", contrast, published, contrast))
    np.testing.assert_allclose(_column(rows, "se"), expected_se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_column(rows, "diff"), contrast @ np.asarray(design_b.coefficients), atol=1e-10)


def test_gaussian_simultaneous_band_covers_the_whole_curve() -> None:
    replicates = 60
    level = 0.95
    covered = []
    pointwise_covered = []
    for replicate in range(replicates):
        alternative = gamfit.fit(_two_groups("gaussian", 200, 40_000 + replicate, True), FORMULA)
        band = _rows(alternative, level=level)
        truth = 0.6 * np.cos(np.pi * _column(band, "x"))
        covered.append(np.all((_column(band, "lower") <= truth) & (truth <= _column(band, "upper"))))
        pointwise = _rows(alternative, level=level, simultaneous=False)
        pointwise_covered.append(
            np.all((_column(pointwise, "lower") <= truth) & (truth <= _column(pointwise, "upper")))
        )
    mcse = math.sqrt(level * (1.0 - level) / replicates)
    coverage = float(np.mean(covered))
    assert abs(coverage - level) <= 2.0 * mcse, (coverage, mcse)
    # The pointwise band is what G2 measured: it misses somewhere on the curve
    # far more often than 5% of the time.
    assert float(np.mean(pointwise_covered)) < level - 2.0 * mcse
