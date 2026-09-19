"""A railed smoothing parameter does not cost the other terms their
smoothing-parameter correction.

On this ten-term Gaussian additive model REML saturates several null and
linear terms at the infinite-smoothing rail. The outer certificate judges the
ρ-Hessian's definiteness OFF those railed coordinates; the correction used to
judge the full matrix instead, met a roundoff-negative curvature on one
saturated axis, refused the whole inverse, and the fit shipped with the
conditional covariance only: every one of its default intervals silently lost
the ``J V_ρ Jᵀ`` term.

A railed axis carries zero ρ-variance exactly (``∂β̂/∂ρ_k = 0`` on a rail
face), so the correction is formed on the interior directions and published.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_TERMS = 10
_N = 300
_NOISE_SD = 0.6
# Replicate 32 of the fixed-seed coverage simulation in the pull request that
# added this test: the first replicate whose fit shipped without a correction.
_SEED = 1032


def _truth(term: int, x: Any) -> Any:
    kind = term % 4
    if kind == 0:
        return np.sin(2.5 * x + 0.2 * term)
    if kind == 1:
        return 0.6 * x
    if kind == 2:
        return 0.8 * np.exp(-4.0 * (x - 0.1 * (term % 3)) ** 2) - 0.35
    return 0.0 * x


@pytest.fixture(scope="module")
def fitted() -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(_SEED)
    x = rng.uniform(-1.0, 1.0, size=(_N, _TERMS))
    mean = sum(_truth(term, x[:, term]) for term in range(_TERMS))
    data: dict[str, Any] = {f"x{term}": x[:, term] for term in range(_TERMS)}
    data["y"] = mean + rng.normal(scale=_NOISE_SD, size=_N)
    formula = "y ~ " + " + ".join(f"s(x{term}, k=10)" for term in range(_TERMS))
    return gamfit.fit(data, formula, family="gaussian"), data


def _inference(model: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.gam")
        model.save(path)
        with open(path) as handle:
            return json.load(handle)["payload"]["fit_result"]["inference"]


def test_the_fit_is_rail_certified(fitted: tuple[Any, dict[str, Any]]) -> None:
    """Premise: without a railed coordinate the rest would test nothing."""
    model, _ = fitted
    convergence = model.summary().convergence
    assert convergence["certified"] is True
    assert len(convergence["outer"]["lambdas_railed"]) >= 1, convergence["outer"]


def test_the_correction_is_published_not_absent(fitted: tuple[Any, dict[str, Any]]) -> None:
    model, _ = fitted
    inference = _inference(model)
    assert inference.get("smoothing_correction_absence") is None, inference.get(
        "smoothing_correction_absence"
    )
    assert inference.get("smoothing_correction_method") is not None
    assert model.summary().covariance_kind != "conditional"


def test_the_default_interval_is_wider_than_the_conditional_one(
    fitted: tuple[Any, dict[str, Any]],
) -> None:
    model, data = fitted
    grid = {name: values for name, values in data.items() if name != "y"}
    corrected = np.asarray(
        model.predict(grid, interval=0.95, return_type="dict")[
            "posterior_mean_standard_error"
        ],
        dtype=float,
    )
    conditional = np.asarray(
        model.predict(
            grid, interval=0.95, covariance_mode="conditional", return_type="dict"
        )["posterior_mean_standard_error"],
        dtype=float,
    )
    assert np.all(np.isfinite(corrected)) and np.all(np.isfinite(conditional))
    # `J V_ρ Jᵀ` is positive semidefinite, so no standard error shrinks, and a
    # present correction moves them.
    assert np.all(corrected >= conditional * (1.0 - 1e-12))
    assert np.mean(corrected / conditional) > 1.0 + 1e-3
