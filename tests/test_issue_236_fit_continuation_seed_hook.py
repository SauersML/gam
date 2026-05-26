"""RED tests for https://github.com/SauersML/gam/issues/236

Standard REML fits via `gamfit.fit(...)` / `gamfit.fit_array(...)` and the
README quickstart all fail with::

    GamError: REML smoothing optimization failed to converge: no candidate
    seeds passed outer startup validation (standard REML):
      ... solver_started=0
      seed 0 (validation): continuation pre-warm refused before seed eval:
      Invalid input: cached inner beta has length N, but this objective
      does not expose an inner-state seeding hook

Root cause: standard REML's value-and-grad closure publishes
`inner_beta_hint = Some(non-empty)`, continuation forwards that hint into
`ClosureObjective::seed_inner_state`, which rejects any non-empty β
because the standard REML closure is built without
`with_seed_inner_state(...)`. Pre-warm wraps the error into a
SeedRejection and every seed is dropped before the inner solver starts.

These tests must be GREEN once the contract is restored: every Gaussian
formula fit on a minimal dataset returns a fitted model.
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


def test_issue_236_pre_warm_error_does_not_leak_to_user() -> None:
    """If a fit raises, the seed-hook error message must not be the cause.

    This is the most specific symptom check: even if the fit fails for
    some unrelated reason, the user must never see "this objective does
    not expose an inner-state seeding hook" surfaced from continuation
    pre-warm.
    """
    train = _quickstart_table()
    try:
        gamfit.fit(train, "y ~ s(x)", family="gaussian")
    except Exception as exc:  # noqa: BLE001 — we inspect the message
        message = str(exc)
        assert "inner-state seeding hook" not in message, (
            "issue #236: continuation pre-warm leaked the seed-hook "
            f"rejection through to the user. message={message!r}"
        )
        assert "continuation pre-warm refused before seed eval" not in message, (
            "issue #236: continuation pre-warm refused before seed eval "
            f"is the original bug. message={message!r}"
        )
