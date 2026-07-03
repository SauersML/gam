"""Regression: a fitted response-geometry GAM must survive the public
save/load (and dumps/loads) round-trip.

Issue #2114: `gamfit.fit(..., response_geometry=...)` returns a
`ResponseGeometryModel` (sphere / simplex / SPD / Grassmann / Stiefel /
Poincaré / constant-curvature responses) that fits and predicts in memory, but
the public persistence API could not round-trip it:

  * `ResponseGeometryModel` defined no `save`/`dumps`, so `gamfit.save(m, path)`
    raised `TypeError` and `m.dumps()` raised `AttributeError`.
  * `gamfit.loads` only ever rebuilt a plain `Model` (or a `MultinomialModel`);
    the response-geometry payload has a different on-disk schema, so no branch
    could reconstruct a `ResponseGeometryModel`.

This mirrors the already-fixed #2078 (`MultinomialModel` save/load). This test
fits a spherical-response GAM, persists it through the public API, reloads it,
and asserts the reload is a `ResponseGeometryModel` whose manifold-valued
predictions match the original to atol 1e-12.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit
from gamfit._response_geometry import ResponseGeometryModel, geometry_exp_map


def _sphere_frame(n: int = 240, dim: int = 3, seed: int = 3) -> pd.DataFrame:
    """Plant unit-sphere responses as exact geodesic exp-images of a smooth
    predictor-driven tangent field at a fixed base point on the sphere."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    base = np.zeros(dim)
    base[0] = 1.0
    amp = 0.25
    tangent = np.zeros((n, dim))
    # Tangent must be orthogonal to `base` (base has a 1 in coord 0), so drive the
    # remaining coordinates only.
    tangent[:, 1] = amp * np.sin(1.7 * x)
    if dim > 2:
        tangent[:, 2] = amp * (x**2 - 0.33)
    y = np.asarray(geometry_exp_map(tangent, geometry="sphere", base=base), dtype=float)
    y = y + 0.01 * rng.standard_normal(y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    cols = {f"y{j}": y[:, j] for j in range(dim)}
    cols["x"] = x
    return pd.DataFrame(cols)


def _fit(frame: pd.DataFrame, dim: int = 3):
    return gamfit.fit(
        frame,
        "y ~ s(x)",
        response_geometry="sphere",
        response_columns=[f"y{j}" for j in range(dim)],
    )


def _predict_matrix(model, frame, dim: int = 3) -> np.ndarray:
    pred = model.predict(frame, return_type="dict")
    return np.column_stack([np.asarray(pred[f"y{j}"], dtype=float) for j in range(dim)])


def test_response_geometry_model_save_load_round_trip() -> None:
    dim = 3
    frame = _sphere_frame(dim=dim)
    m = _fit(frame, dim)
    assert type(m) is ResponseGeometryModel

    expected = _predict_matrix(m, frame, dim)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "response_geometry.gam"
        gamfit.save(m, path)          # raised TypeError before the fix
        m2 = gamfit.load(path)

    assert type(m2) is ResponseGeometryModel
    assert m2.response_geometry == m.response_geometry
    assert m2.response_columns == m.response_columns
    assert m2.coordinates == m.coordinates

    got = _predict_matrix(m2, frame, dim)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_response_geometry_model_dumps_loads_round_trip() -> None:
    dim = 3
    frame = _sphere_frame(dim=dim)
    m = _fit(frame, dim)

    payload = m.dumps()               # raised AttributeError before the fix
    m2 = gamfit.loads(payload)

    assert type(m2) is ResponseGeometryModel
    assert m2.response_columns == m.response_columns
    np.testing.assert_allclose(
        _predict_matrix(m2, frame, dim),
        _predict_matrix(m, frame, dim),
        atol=1e-12,
    )
