"""Bug hunt: the spherical response-geometry fit is not rotation equivariant.

``gamfit.fit(..., response_geometry="spherical")`` advertises a coordinate-free
directional regression: each response row is mapped to the tangent space at the
*intrinsic* Fréchet/Karcher mean of the responses, a Gaussian REML fit is run on
the tangent coordinates, and predictions are mapped back with the geodesic
exponential.  Every ingredient of that construction (the Karcher mean, the
geodesic ``log``/``exp`` maps, the Frobenius tangent metric) is equivariant
under a rotation ``R`` of the ambient frame, so the *only* correct behaviour is

    predict(fit(R · responses)) == R · predict(fit(responses)).

In other words, relabelling the arbitrary ``(x, y, z)`` axes of the sphere must
not change the physical predictions.

It does.  ``gamfit/_response_geometry.py`` fits **each ambient tangent
coordinate as its own scalar Gaussian GAM with its own per-coordinate smoothing
parameter** (see ``SharedGaussianRemlTangentFit`` /
``ResponseGeometryModel`` docstrings).  Per-axis smoothing in an *arbitrary*
ambient frame is not rotation invariant: a rotation that mixes a high-curvature
tangent direction with a low-curvature one is smoothed completely differently
after the mix, so the fitted surface — and the predictions — depend on the axis
labelling.

This test fixes a deterministic anisotropic directional dataset (a wiggly
tangent direction crossed with a smooth one) near the north pole, fits it, then
fits the *same* data rotated by 45° about the ``z`` axis (which leaves the
Karcher mean's axis untouched, so the discrepancy can come only from the fit).
The model is perfectly deterministic — refitting identical data reproduces the
predictions to the bit — yet the rotated fit disagrees with the rotated
predictions by ~1e-2 (a ~1° geodesic angle), orders of magnitude above the
zero determinism floor.

When the tangent fit is made frame-equivariant (e.g. a single shared smoothing
parameter / isotropic penalty across the tangent coordinates, or an intrinsic
tangent-basis fit), the equivariance residual collapses to the float floor and
this test passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _anisotropic_directions(seed: int = 20250609, n: int = 900):
    """Smooth directional data on S²: a wiggly tangent axis crossed with a
    smooth one, exp-mapped from the north pole, plus small tangential noise."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    a = 0.5 * np.sin(8.0 * np.pi * x)  # high-frequency tangent direction (e_x)
    b = 0.6 * (x - 0.5)               # smooth tangent direction (e_y)
    theta = np.sqrt(a * a + b * b)
    scale = np.where(theta > 1e-12, np.sin(theta) / np.where(theta > 1e-12, theta, 1.0), 1.0)
    directions = np.stack([scale * a, scale * b, np.cos(theta)], axis=1)
    directions = directions + rng.normal(0.0, 0.03, directions.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return x, directions


def _fit_predict(x, directions, x_eval):
    df = pd.DataFrame(
        {"x": x, "u": directions[:, 0], "v": directions[:, 1], "w": directions[:, 2]}
    )
    model = gamfit.fit(
        df,
        "u ~ s(x, k=24)",
        response_geometry="spherical",
        response_columns=["u", "v", "w"],
    )
    pred = model.predict(pd.DataFrame({"x": x_eval}))
    return np.asarray(pred[["u", "v", "w"]], dtype=float)


def test_spherical_response_fit_is_rotation_equivariant() -> None:
    x, directions = _anisotropic_directions()
    x_eval = np.linspace(0.02, 0.98, 40)

    # 45° rotation about the z-axis: leaves the (north-pole) base-point axis
    # alone, so any equivariance failure must originate in the tangent fit.
    c, s = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    preds = _fit_predict(x, directions, x_eval)

    # Sanity: the fit is deterministic, so the equivariance residual below is
    # not sampling noise. Refitting identical data reproduces predictions.
    preds_again = _fit_predict(x, directions, x_eval)
    determinism_floor = float(np.abs(preds - preds_again).max())
    assert determinism_floor < 1e-9, (
        f"response-geometry fit is not deterministic (floor={determinism_floor:.2e}); "
        "the equivariance check below would be meaningless"
    )

    # Fit the rotated data and compare against the rotated predictions.
    preds_rotated = _fit_predict(x, directions @ rot.T, x_eval)
    expected = preds @ rot.T  # rotate every prediction row by the same rotation

    max_coord_err = float(np.abs(preds_rotated - expected).max())
    cos = np.clip((preds_rotated * expected).sum(axis=1), -1.0, 1.0)
    max_angle_deg = float(np.degrees(np.arccos(cos)).max())

    # A frame-equivariant fit reproduces R·predict() to the float floor
    # (~1e-9). The per-ambient-axis smoothing currently breaks this by ~1e-2
    # (≈1° of geodesic angle). The 1e-3 gate sits an order of magnitude below
    # the observed violation and well above a correct implementation's floor.
    assert max_coord_err < 1e-3, (
        "spherical response-geometry predictions are not rotation equivariant: "
        f"max |predict(fit(R·y)) - R·predict(fit(y))| = {max_coord_err:.3e} "
        f"(max geodesic angle {max_angle_deg:.3f}°), while refitting identical "
        f"data agrees to {determinism_floor:.1e}. The per-tangent-coordinate "
        "smoothing in fit_response_geometry depends on the arbitrary ambient frame."
    )
