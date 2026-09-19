"""Binomial ``y ~ s(x)`` on separated and rare-event data must fit, not fail.

A flat prior on the unpenalized block of a binomial smooth gives an improper
posterior as soon as that block admits a (quasi-)separating direction, and the
smooth's null-space shrinkage ridge does not cure it: REML drives the ridge's
lambda to zero along the separating direction. The Jeffreys (Firth) prior makes
the posterior proper, and the engine engages it when the flat-prior fit refuses.

These fixtures used to fail: the perfectly separated step ground for more than
20 minutes without returning, and the quasi-separated one raised
``RemlConvergenceError`` with the outer search stalled at ``|g| = 0.23``. Both
came from the #784 block quadrature correction integrating the flat-prior
posterior on a Firth fit, which is improper along the separating direction.
The Firth rescue engages only on certified separation, and the quasi-separated
fixture has no strict separator: its flat-prior fit returned an optimum with
the null-space ridge's lambda railed at zero. So the pre-fit check also
certifies quasi-complete separation along a null-space direction.

Each fit must carry a convergence certificate, and its posterior-mean
predictions must lie in (0, 1) with finite intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _perfect() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 200)
    return x, (x > 0.5).astype(float)


def _quasi() -> tuple[np.ndarray, np.ndarray]:
    # x on the tenths grid; the rows tied at the boundary x = 0.5 are split
    # between the classes, so the separating hyperplane passes through them.
    rng = np.random.default_rng(11)
    x = np.round(rng.uniform(0.0, 1.0, 200), 1)
    y = (x > 0.5).astype(float)
    y[np.flatnonzero(x == 0.5)[::2]] = 1.0
    return x, y


def _rare() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 1000)
    y = np.zeros(1000)
    y[rng.choice(1000, 3, replace=False)] = 1.0
    return x, y


CASES = {"perfect_step": _perfect, "quasi_separated": _quasi, "rare_events": _rare}


def _assert_certified_proper_fit(x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial", **kwargs)
    convergence = model.summary().convergence
    assert convergence["certified"] is True, convergence
    assert convergence["inner_status"] == "Converged", convergence

    grid = np.linspace(0.0, 1.0, 41)
    pred = model.predict({"x": grid}, interval=0.95)
    mean = np.asarray(pred["posterior_mean"], dtype=float)
    lower = np.asarray(pred["posterior_mean_lower"], dtype=float)
    upper = np.asarray(pred["posterior_mean_upper"], dtype=float)
    for name, values in (("mean", mean), ("lower", lower), ("upper", upper)):
        assert np.all(np.isfinite(values)), (name, values)
    assert np.all((mean > 0.0) & (mean < 1.0)), mean
    assert np.all((lower >= 0.0) & (upper <= 1.0)), (lower, upper)
    assert np.all((lower <= mean) & (mean <= upper)), (lower, mean, upper)


@pytest.mark.parametrize("case", sorted(CASES))
def test_binomial_smooth_fits_under_separation(case: str) -> None:
    x, y = CASES[case]()
    _assert_certified_proper_fit(x, y)


@pytest.mark.parametrize("case", ["perfect_step", "quasi_separated"])
def test_binomial_smooth_fits_under_separation_with_explicit_firth(case: str) -> None:
    x, y = CASES[case]()
    _assert_certified_proper_fit(x, y, firth=True)


def test_perfect_step_fit_follows_the_step() -> None:
    x, y = _perfect()
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial")
    pred = model.predict({"x": np.array([0.1, 0.9])}, interval=0.95)
    mean = np.asarray(pred["posterior_mean"], dtype=float)
    assert mean[0] < 0.5 < mean[1], mean
