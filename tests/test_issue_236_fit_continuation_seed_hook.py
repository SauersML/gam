"""Public fit-to-completion guards for https://github.com/SauersML/gam/issues/236.

Standard REML publishes and consumes original-basis coefficient state. Cached
coefficients are now bound to the exact outer coordinate that owns them and are
installed after the per-seed reset. These tests keep the public Gaussian paths
honest without asserting behavior of a deleted speculative startup phase.
"""

from __future__ import annotations

from importlib import import_module
from typing import cast

import numpy as np

import gamfit

pytest = cast("object", import_module("pytest"))


def _quickstart_table() -> dict[str, list[float]]:
    # Wider than the README sample so the spline term has enough degrees
    # of freedom to fit. Standard Gaussian-with-smooth path is enough to
    # exercise the regression — issue #236 fails at outer startup before
    # the inner solver runs, so dataset size does not matter past the
    # design rank.
    n = 32
    x = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(20260525)
    y = 0.4 + 1.3 * np.sin(2.0 * np.pi * x) + rng.normal(scale=0.05, size=n)
    return {"x": x.tolist(), "y": y.tolist()}


def test_issue_236_fit_smooth_returns_model() -> None:
    train = _quickstart_table()
    model = gamfit.fit(train, "y ~ s(x)", family="gaussian")
    assert model is not None, "fit('y ~ s(x)') must return a fitted model"


def test_issue_236_fit_linear_returns_model() -> None:
    train = _quickstart_table()
    model = gamfit.fit(train, "y ~ x", family="gaussian")
    assert model is not None, "fit('y ~ x') must return a fitted model"


def test_issue_236_fit_array_returns_model() -> None:
    rng = np.random.default_rng(0)
    X = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0] + rng.normal(scale=0.01, size=16)
    model = gamfit.fit_array(X, y, "y ~ x0", family="gaussian")
    assert model is not None, "fit_array('y ~ x0') must return a fitted model"


def test_issue_236_readme_quickstart_runs() -> None:
    """The exact shape advertised in README_PYPI.md must work."""
    train = [
        {"y": 1.2, "x": 0.0},
        {"y": 1.9, "x": 1.0},
        {"y": 3.1, "x": 2.0},
        {"y": 4.5, "x": 3.0},
        {"y": 5.7, "x": 4.0},
        {"y": 6.9, "x": 5.0},
        {"y": 8.1, "x": 6.0},
        {"y": 9.5, "x": 7.0},
    ]
    model = gamfit.fit(train, "y ~ s(x)")
    assert model is not None, "README quickstart fit must return a model"
