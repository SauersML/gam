"""pyGAM audit lane ``basis-defaults-te-tp``: default basis dimensions that are
adequate for the data, and no silently-flat thin-plate smooth on outlying x.

Each case is one the audit measured gamfit losing at the formula defaults:

* ``te(x0, x1)`` on a two-bump surface (n = 4000). The default tensor was a
  fixed 7 x 7 whatever the row count, and its own basis-adequacy test rejected
  it (p ~ 1e-17), so the truth-MSE sat at ~0.008. A formula-default ``te`` of
  ``d`` covariates now takes the same total basis dimension as every other
  ``d``-covariate default smooth on those rows, split across its margins and
  capped per margin by the covariate's distinct values; the REML penalty, not
  the basis size, sets the smoothness.
* ``te(season, hour)`` on the bike-sharing torus data. ``hour`` has 24 distinct
  values but the default margin used 6 of them (held-out MSE ~0.17): each
  margin was capped at its 1-D internal-knot count (#3181). Margins are now
  capped by their distinct values, and the row-count budget gives both 13.
* ``s(x, bs='tps')`` with one x = 1e6 among 300 rows on [0, 1). The isotropic
  standardization put the whole bulk inside ~2e-5 of standardized space, where
  the r^3 kernel differences are ~1e-15 of the outlier's: every bulk bending
  direction fell under the numerical-rank floor and the fit came back as a flat
  line with edf 1 and no warning. That basis is now refused with a typed error
  that names the outlying span and the remedies.
* ``s(x)`` on sin(12 pi x) (n = 500) already clears its bar at HEAD since the
  adaptive 1-D B-spline default landed; it is pinned here so it stays cleared.
"""

from __future__ import annotations

import pathlib
import zlib

import numpy as np
import pytest

import gamfit
from gamfit.errors import InvalidConfigurationError

BIKE_CSV = pathlib.Path(__file__).resolve().parents[1] / "bench" / "datasets" / "bike_sharing_torus.csv"


def _holdout_split(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = n - n // 5
    return order[:cut], order[cut:]


def _subset(cols: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[rows] for name, values in cols.items()}


def test_default_1d_smooth_resolves_a_six_cycle_sine() -> None:
    n = 500
    rng = np.random.default_rng(zlib.crc32(f"sin6:{n}".encode()))
    x = rng.uniform(0.0, 1.0, n)
    mu = np.sin(2 * np.pi * 6 * x)
    y = mu + rng.normal(0.0, 0.3 * np.std(mu) + 0.1, n)
    train, test = _holdout_split(n)
    model = gamfit.fit({"x": x[train], "y": y[train]}, "y ~ s(x)")
    pred = np.asarray(model.predict({"x": x[test]}), float).ravel()
    assert float(np.mean((pred - mu[test]) ** 2)) < 0.02


def test_default_te_resolves_a_two_bump_surface() -> None:
    n = 4000
    rng = np.random.default_rng(n + 1)
    X = rng.uniform(0.0, 1.0, (n, 2))
    mu = 3 * np.exp(-((X[:, 0] - 0.3) ** 2 + (X[:, 1] - 0.6) ** 2) / 0.05) + 2 * np.exp(
        -((X[:, 0] - 0.75) ** 2 + (X[:, 1] - 0.25) ** 2) / 0.02
    )
    y = mu + rng.normal(0.0, 0.3, n)
    cols = {"x0": X[:, 0], "x1": X[:, 1]}
    train, test = _holdout_split(n)
    model = gamfit.fit({**_subset(cols, train), "y": y[train]}, "y ~ te(x0, x1)")
    pred = np.asarray(model.predict(_subset(cols, test)), float).ravel()
    assert float(np.mean((pred - mu[test]) ** 2)) < 0.004


def test_default_te_uses_the_hours_the_bike_data_actually_has() -> None:
    frame = np.genfromtxt(BIKE_CSV, delimiter=",", names=True)
    cols = {name: np.asarray(frame[name], float) for name in ("season", "hour")}
    y = np.asarray(frame["log_count"], float)
    train, test = _holdout_split(len(y))
    model = gamfit.fit({**_subset(cols, train), "y": y[train]}, "y ~ te(season, hour)")
    pred = np.asarray(model.predict(_subset(cols, test)), float).ravel()
    assert float(np.mean((y[test] - pred) ** 2)) < 0.08


def test_thin_plate_smooth_refuses_a_bulk_flattened_by_an_outlier() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)
    x[0] = 1.0e6
    with pytest.raises(InvalidConfigurationError) as caught:
        gamfit.fit({"x": x, "y": y}, "y ~ s(x, bs=tps)")
    message = str(caught.value)
    assert "cannot resolve the bulk of its data" in message
    assert "1.000000e6" in message
    assert "middle half spans" in message
    assert "bs='cr'" in message


def test_thin_plate_smooth_on_the_same_bulk_without_the_outlier_still_bends() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x, bs=tps)")
    grid = np.linspace(0.05, 0.95, 19)
    pred = np.asarray(model.predict({"x": grid}), float).ravel()
    assert float(np.mean((pred - np.sin(2 * np.pi * grid)) ** 2)) < 0.01


@pytest.mark.parametrize("formula", ["y ~ s(b) + te(b, c)", "y ~ s(b) + te(b, c, k=[14, 14])"])
def test_tensor_sharing_a_margin_with_a_smooth_predicts_from_its_frozen_basis(formula: str) -> None:
    # A te whose margin also carries its own s() is whitened against the design
    # Gram, so its frozen chart is not orthonormal. Its null-function ridge was
    # classified by a spectral rank test in that chart, which over-counted the
    # joint null and made the rebuilt prediction design disagree with the fit.
    n = 1000
    rng = np.random.default_rng(zlib.crc32(formula.encode()))
    b = rng.uniform(0.0, 1.0, n)
    c = rng.uniform(0.0, 1.0, n)
    mu = np.sin(3 * b) + b * c
    y = mu + rng.normal(0.0, 0.1, n)
    train, test = _holdout_split(n)
    cols = {"b": b, "c": c}
    model = gamfit.fit({**_subset(cols, train), "y": y[train]}, formula)
    pred = np.asarray(model.predict(_subset(cols, test)), float).ravel()
    fitted = np.asarray(model.predict(_subset(cols, train)), float).ravel()
    assert np.all(np.isfinite(pred))
    assert float(np.mean((fitted - mu[train]) ** 2)) < 0.005
    assert float(np.mean((pred - mu[test]) ** 2)) < 0.005
