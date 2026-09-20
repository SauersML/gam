"""Seeded calibration of the per-term smooth likelihood-ratio p-value (pyGAM
audit, lane pv-lr-selection).

The Monte Carlo study lives in ``bench/pvalue_calibration/pv-lr-selection/``;
this file is its seeded regression, on the same 600 datasets per cell.

Truth ``eta = b0 + a1 sin(2 pi x1) + a3 cos(2 pi x3)`` with ``x ~ U(0,1)^3``,
model ``y ~ s(x1) + s(x2) + s(x3)``. ``s(x2)`` is null (size) and ``s(x3)``
is weak (power).

Size. At ``a`` = 0.10, 0.05 and 0.01 the rejection rate must lie within two
binomial Monte-Carlo standard errors of ``a``, on both sides: an
anti-conservative reference fails the upper bound and a conservative one the
lower.

Shape. The null p-value must be ``U(0, 1)`` over its whole range: the
two-sided Kolmogorov-Smirnov test of every null p-value against ``U(0, 1)``
must not reject at 0.05. A mass of p-values piled near 1 fails it exactly as
an excess near 0 does.

Power. The weak-term power at 0.05 must not fall below that of the
reference this lane replaced, measured on the same datasets, by more than the
same two Monte-Carlo standard errors the size is held to.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

gamfit = pytest.importorskip("gamfit")

REPS = 600
FORMULA = "y ~ s(x1) + s(x2) + s(x3)"
CELLS = {
    # name: (family, n, b0, a1, a3, sigma, power at 0.05 of the reference on
    # main before this lane, on these datasets)
    "gaussian_n60": ("gaussian", 60, 0.0, 1.0, 0.30, 0.5, 440 / 600),
    "gaussian_n200": ("gaussian", 200, 0.0, 1.0, 0.30, 1.0, 407 / 600),
    "binomial_n400": ("binomial", 400, 0.0, 1.5, 0.60, None, 542 / 596),
    "poisson_n200": ("poisson", 200, 0.5, 0.8, 0.25, None, 491 / 600),
}


def _replicate(cell: str, rep: int):
    family, n, b0, a1, a3, sigma, _ = CELLS[cell]
    # The bench harness's seed stream: replicate r is default_rng(50000 + r).
    rng = np.random.default_rng(50000 + rep)
    x = rng.uniform(0.0, 1.0, (n, 3))
    eta = b0 + a1 * np.sin(2 * np.pi * x[:, 0]) + a3 * np.cos(2 * np.pi * x[:, 2])
    if family == "gaussian":
        y = eta + rng.normal(0.0, sigma, n)
    elif family == "binomial":
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return family, {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y}


def _p_values(cell: str):
    null, weak = [], []
    for rep in range(REPS):
        family, data = _replicate(cell, rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = gamfit.fit(data, FORMULA, family=family).smooth_significance(data)
        by_name = {row["name"]: row for row in rows}
        null.append(by_name["s(x2)"]["p_value_corrected"])
        weak.append(by_name["s(x3)"]["p_value_corrected"])
    return np.asarray(null, dtype=float), np.asarray(weak, dtype=float)


@pytest.mark.slow
@pytest.mark.parametrize("cell", sorted(CELLS))
def test_smooth_lr_is_sized_and_keeps_its_power(cell):
    null, weak = _p_values(cell)
    assert np.all(np.isfinite(null)), "a null replicate published no p-value"
    for level in (0.10, 0.05, 0.01):
        rate = float(np.mean(null <= level))
        mcse = np.sqrt(level * (1.0 - level) / null.size)
        assert abs(rate - level) <= 2.0 * mcse, (
            f"{cell}: size {rate:.4f} at {level} is {(rate - level) / mcse:+.2f} MCSE"
        )
    assert stats.kstest(null, "uniform").pvalue > 0.05, cell
    power = float(np.mean(weak <= 0.05))
    before = CELLS[cell][-1]
    mcse = np.sqrt(before * (1.0 - before) / weak.size)
    assert power >= before - 2.0 * mcse, (
        f"{cell}: weak-term power {power:.4f} fell from {before:.4f} "
        f"({(power - before) / mcse:+.2f} MCSE)"
    )
