"""Bug hunt (#2140): ``response_geometry="stiefel(k=1)"`` refuses widely spread
data that the identical ``"sphere"`` geometry fits.

``St(n, 1)`` is *exactly* the sphere ``S^{n-1}`` — the Stiefel ``k=1`` exp/log/
metric dispatch straight to the sphere
(``crates/gam-geometry/src/manifolds/stiefel.rs::as_sphere``). So
``response_geometry="stiefel(k=1)"`` and ``response_geometry="sphere"`` describe
the SAME manifold on the SAME data. Yet on widely spread responses ``"sphere"``
fit fine while ``"stiefel(k=1)"`` used to abort the whole fit with::

    response geometry Fréchet mean did not reach stationarity within max_iter

Root cause: the two geometries are backed by two different Fréchet/Karcher-mean
drivers. ``"sphere"`` uses ``sphere_frechet_mean`` (multistart, keeps the best
iterate, never errors on a budget shortfall). ``"stiefel(k=1)"`` (and grassmann/
spd/poincare/constant_curvature) uses the generic ``response_frechet_mean``,
whose ``max_iter``-exhausted branch *discarded* its tracked best iterate and
returned ``Err`` — unlike its own two other non-stationary exits (stall,
rejected step) and unlike the sphere driver. On a positively curved manifold
(``St(3,1) = S^2``) with responses spread over the whole sphere the single-seed
descent converges only linearly and can exhaust the 256-iteration budget while
still making monotone progress, so the fit aborted even though a perfectly good
approximate Fréchet mean was in hand.

Fixed in ``crates/gam-geometry/src/response_geometry.rs``: all non-stationary
exits now keep the best on-manifold iterate, and the generic driver multistarts
over admissible seeds (keeping the lowest-dispersion mean) for positively curved
geometries — mirroring ``sphere_frechet_mean``. This test builds the issue's
deterministic 80-point Fibonacci S^2 cover and asserts that BOTH ``"sphere"``
(control) and ``"stiefel(k=1)"`` fit and predict on-manifold (unit-norm) rows.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic Fibonacci-lattice cover of S^2, spread over the WHOLE
    sphere — the widely spread cloud that makes the Fréchet objective nearly
    flat and exhausts the single-seed Karcher budget."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def _predict_matrix(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "values"):
        return np.asarray(prediction.values, dtype=float)
    return np.asarray(prediction, dtype=float)


def _fit_and_check_unit(geometry: str) -> None:
    v = _fibonacci_sphere(80)
    x = np.linspace(0.0, 1.0, 80)
    df = pd.DataFrame({"x": x, "a": v[:, 0], "b": v[:, 1], "c": v[:, 2]})

    model = gamfit.fit(
        df,
        "r ~ s(x)",
        response_geometry=geometry,
        response_columns=["a", "b", "c"],
    )

    preds = _predict_matrix(model.predict(df.head(20)))
    assert preds.shape[1] == 3, f"{geometry}: expected 3 columns, got {preds.shape}"
    norms = np.linalg.norm(preds, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    assert worst < 1e-8, (
        f"{geometry}: predictions must lie on the sphere (unit-norm); "
        f"max|‖ŷ‖ - 1| = {worst:.3e}"
    )


def test_sphere_control_fits_widely_spread_cloud() -> None:
    """Control: the dedicated sphere driver fits the widely spread cloud."""
    _fit_and_check_unit("sphere")


def test_stiefel_k1_fits_the_same_data_as_sphere() -> None:
    """#2140: stiefel(k=1) IS the sphere; it must fit the identical data the
    sphere geometry fits, rather than aborting in the Fréchet-mean driver."""
    _fit_and_check_unit("stiefel(k=1)")
