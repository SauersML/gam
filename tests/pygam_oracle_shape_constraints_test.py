"""pyGAM oracle: shape constraints (pygam/tests/test_penalties.py).

pyGAM fits ``constraints='monotonic_inc' | 'monotonic_dec' | 'convex' |
'concave'`` and checks the sign of the first or second difference of the
sorted prediction. gamfit's shape constraints are exact cone constraints on
the function, so the guarantee is stronger and is checked on every object the
user can read:

* the plug-in prediction ``mean_plugin``,
* the reported posterior mean,
* the term's ``partial_dependence`` curve, and
* every posterior draw from ``Model.sample``.

``tests/test_posterior_monotone_shape_constraint.py`` covers draws for the
increasing, decreasing and convex shapes only; the concave draws and the
shaped partial dependence of all four shapes were unlocked.

The data are a full period of a sine, so every one of the four constraints is
active somewhere on the support: an unconstrained fit violates each of them,
and a fit that ignored the ``shape=`` option would fail every test here.
pyGAM is never imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gamfit

# shape -> (difference order, required sign of that difference)
SHAPES: dict[str, tuple[int, int]] = {
    "monotone_increasing": (1, +1),
    "monotone_decreasing": (1, -1),
    "convex": (2, +1),
    "concave": (2, -1),
}


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(30)
    n = 200
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = 2.0 + np.sin(2.0 * np.pi * x) + 0.5 * x + rng.normal(0.0, 0.2, n)
    return {"x": x, "y": y}


def _grid(n: int = 120) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _worst(values: np.ndarray, shape: str) -> np.ndarray:
    """Most-violating signed difference along the last axis (>= 0 means obeyed)."""
    order, sign = SHAPES[shape]
    return np.min(sign * np.diff(np.asarray(values, float), n=order, axis=-1), axis=-1)


def _tol(values: np.ndarray) -> float:
    # Rounding of the evaluated curve, relative to its size.
    return 1e-10 * max(1.0, float(np.max(np.abs(values))))


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    return _data()


@pytest.fixture(scope="module", params=list(SHAPES))
def shaped(
    request: pytest.FixtureRequest, data: dict[str, np.ndarray]
) -> tuple[str, Any]:
    shape = str(request.param)
    return shape, gamfit.fit(data, f"y ~ s(x, shape={shape})")


def _term(model: Any) -> str:
    (block,) = [b for b in model.term_blocks if b.kind != "intercept"]
    return str(block.name)


def test_every_shape_is_active_on_this_data(data: dict[str, np.ndarray]) -> None:
    """Guard for the tests below: the free fit breaks all four shapes, so a
    constraint that were silently dropped would be caught."""
    free = gamfit.fit(data, "y ~ s(x)")
    curve = np.asarray(
        free.predict({"x": _grid()}, return_type="dict")["mean_plugin"], float
    )
    for shape in SHAPES:
        assert _worst(curve, shape) < -1e3 * _tol(curve), shape


def test_plugin_prediction_obeys_shape(shaped: tuple[str, Any]) -> None:
    shape, m = shaped
    curve = np.asarray(
        m.predict({"x": _grid()}, return_type="dict")["mean_plugin"], float
    )
    assert _worst(curve, shape) >= -_tol(curve), shape


def test_posterior_mean_obeys_shape(shaped: tuple[str, Any]) -> None:
    shape, m = shaped
    curve = np.asarray(m.predict({"x": _grid()}), float)
    assert _worst(curve, shape) >= -_tol(curve), shape


def test_partial_dependence_obeys_shape(shaped: tuple[str, Any]) -> None:
    shape, m = shaped
    curve = np.asarray(m.partial_dependence(_term(m), grid=_grid())["predicted"], float)
    assert _worst(curve, shape) >= -_tol(curve), shape
    # The pdep is the plug-in curve minus the intercept, so it is the same
    # shaped function, not a re-centred or re-fitted one.
    plugin = np.asarray(
        m.predict({"x": _grid()}, return_type="dict")["linear_predictor_plugin"]
    )
    np.testing.assert_allclose(
        np.diff(curve), np.diff(plugin), rtol=0.0, atol=_tol(plugin)
    )


def test_posterior_draws_obey_shape(
    shaped: tuple[str, Any], data: dict[str, np.ndarray]
) -> None:
    shape, m = shaped
    post = m.sample(data, samples=80, seed=0)
    draws = np.asarray(post.predict_draws({"x": _grid(60)}).eta, float)
    assert draws.ndim == 2 and draws.shape[1] == 60
    worst = _worst(draws, shape)
    bad = int(np.sum(worst < -_tol(draws)))
    assert bad == 0, f"{bad}/{draws.shape[0]} posterior draws violate {shape}"
    # Truncation to the cone must not collapse the posterior onto one curve.
    assert float(np.max(np.std(draws, axis=0))) > 1e-3


@pytest.mark.parametrize("shape", ["monotone_increasing", "monotone_decreasing"])
def test_monotone_credible_band_endpoints_are_monotone(
    shape: str, data: dict[str, np.ndarray]
) -> None:
    """Every admissible curve is monotone, so the pointwise band of the mean
    shifts in the constrained direction and both endpoints inherit the order."""
    m = gamfit.fit(data, f"y ~ s(x, shape={shape})")
    band = m.predict({"x": _grid()}, interval=0.95)
    for key in ("posterior_mean_lower", "posterior_mean_upper"):
        v = np.asarray(band[key], float)
        assert _worst(v, shape) >= -_tol(v), (shape, key)


@pytest.mark.parametrize("seed", [31, 32])
def test_monotone_term_beside_a_free_smooth_in_a_poisson_model(seed: int) -> None:
    """pyGAM's constrained Poisson fit: the shaped term keeps its shape when a
    second, unconstrained smooth shares the predictor. The z effect is a
    decreasing logistic step, so the constraint is well specified while an
    unconstrained smooth rings around the step."""
    rng = np.random.default_rng(seed)
    n = 800
    t = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    eta = 1.0 + 0.8 * np.sin(4.0 * t) - 1.5 / (1.0 + np.exp(-(z - 0.5) / 0.05))
    d = {"t": t, "z": z, "y": rng.poisson(np.exp(eta)).astype(float)}
    m = gamfit.fit(d, "y ~ s(t) + s(z, shape=monotone_decreasing)", family="poisson")
    (term,) = [b.name for b in m.term_blocks if "z" in b.name]
    curve = np.asarray(m.partial_dependence(term, grid=_grid())["predicted"], float)
    assert _worst(curve, "monotone_decreasing") >= -_tol(curve)
    # Ignoring the shape would show: the free fit overshoots the step.
    free = gamfit.fit(d, "y ~ s(t) + s(z)", family="poisson")
    (free_term,) = [b.name for b in free.term_blocks if "z" in b.name]
    free_curve = np.asarray(
        free.partial_dependence(free_term, grid=_grid())["predicted"], float
    )
    assert _worst(free_curve, "monotone_decreasing") < -1e3 * _tol(free_curve)
