"""Regression for issue #381.

`response_geometry` (spherical / simplex / clr / alr) must accept smooth RHS
terms (`s()`, `te()`, ...). A single thin-plate `s(x)` smooth expands to two
penalty blocks (wiggle + null-space ridge), which used to trip a single-penalty
closed-form guard ("requires exactly one smoothing penalty; got 2"). The
geometry tangent fit now routes through the general multi-penalty REML solver,
so every smooth RHS is fittable and predictions land back on the manifold.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit  # noqa: E402  (import after skip guard)


def _sphere_design(n: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return {
        "u1": v[:, 0].tolist(),
        "u2": v[:, 1].tolist(),
        "u3": v[:, 2].tolist(),
        "x": rng.normal(0.0, 1.0, n).tolist(),
        "z": rng.normal(0.0, 1.0, n).tolist(),
    }


def _simplex_design(n: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    raw = rng.gamma(2.0, 1.0, (n, 3)) + 0.1
    comp = raw / raw.sum(axis=1, keepdims=True)
    return {
        "p1": comp[:, 0].tolist(),
        "p2": comp[:, 1].tolist(),
        "p3": comp[:, 2].tolist(),
        "x": rng.normal(0.0, 1.0, n).tolist(),
        "z": rng.normal(0.0, 1.0, n).tolist(),
    }


@pytest.mark.parametrize("formula", ["~ s(x)", "~ s(x) + z", "~ x + s(z)"])
def test_spherical_response_accepts_smooth_terms(formula: str) -> None:
    data = _sphere_design(240, seed=5)
    model = gamfit.fit(
        data,
        formula,
        response_geometry="spherical",
        response_columns=["u1", "u2", "u3"],
    )
    pred = model.predict(data, return_type="dict")
    out = np.column_stack([pred[name] for name in ("u1", "u2", "u3")])
    # Predictions must lie on the unit sphere (the exp map renormalizes the
    # tangent prediction back onto the manifold).
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(out.shape[0]), atol=1e-9)
    # The fit must have used multi-penalty REML: an s() smooth contributes more
    # than one penalty block per coordinate.
    # The fit must have used multi-penalty REML: an s() smooth contributes more
    # than one penalty block. The smoothing parameters are now SHARED across the
    # tangent coordinates (issue #967), so `lambdas` is a single per-smooth
    # vector of length M rather than a (D, M) per-coordinate grid — one lambda
    # per formula smooth, common to every coordinate.
    summary = model.summary()
    assert summary["shared_smoothing"] is True
    lambdas = np.asarray(summary["shared_fit"]["lambdas"], dtype=float)
    assert lambdas.ndim == 1  # shared per-smooth, not per-coordinate
    assert lambdas.shape[0] >= 2  # >= two penalty blocks for an s() smooth
    assert np.all(np.isfinite(lambdas))


@pytest.mark.parametrize("formula", ["~ s(x)", "~ s(x) + z"])
def test_simplex_response_accepts_smooth_terms(formula: str) -> None:
    data = _simplex_design(240, seed=9)
    model = gamfit.fit(
        data,
        formula,
        response_geometry="simplex",
        response_columns=["p1", "p2", "p3"],
    )
    pred = model.predict(data, return_type="dict")
    out = np.column_stack([pred[name] for name in ("p1", "p2", "p3")])
    # Compositional predictions must be positive and sum to one.
    assert np.all(out > 0.0)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(out.shape[0]), atol=1e-9)
    # Shared per-smooth smoothing parameters (issue #967): a length-M vector,
    # one lambda per formula smooth common to every compositional coordinate.
    lambdas = np.asarray(model.summary()["shared_fit"]["lambdas"], dtype=float)
    assert lambdas.ndim == 1
    assert lambdas.shape[0] >= 2
    assert np.all(np.isfinite(lambdas))


def test_smooth_tangent_fit_tracks_signal_better_than_intercept() -> None:
    # A clear smooth signal in the tangent direction should be recovered: the
    # smooth fit must reduce residual variance well below the marginal variance
    # of the response coordinates, which a degenerate (penalty-saturated) or
    # rejected fit could not do.
    rng = np.random.default_rng(17)
    n = 300
    x = np.sort(rng.uniform(-2.0, 2.0, n))
    angle = 0.9 * np.sin(1.5 * x) + 0.05 * rng.normal(0.0, 1.0, n)
    u1 = np.cos(angle)
    u2 = np.sin(angle)
    u3 = np.full(n, 0.1)
    norm = np.sqrt(u1 * u1 + u2 * u2 + u3 * u3)
    data = {
        "u1": (u1 / norm).tolist(),
        "u2": (u2 / norm).tolist(),
        "u3": (u3 / norm).tolist(),
        "x": x.tolist(),
    }
    model = gamfit.fit(
        data,
        "~ s(x)",
        response_geometry="spherical",
        response_columns=["u1", "u2", "u3"],
    )
    pred = model.predict(data, return_type="dict")
    out = np.column_stack([pred[name] for name in ("u1", "u2", "u3")])
    observed = np.column_stack([data[name] for name in ("u1", "u2", "u3")])
    resid_ss = float(np.sum((observed - out) ** 2))
    total_ss = float(np.sum((observed - observed.mean(axis=0)) ** 2))
    assert resid_ss < 0.5 * total_ss
