"""#2899 P4: a position fit's basis is resolved once, in Rust.

``gaussian_reml_fit_positions`` and its backward and batched forms used to
resolve their basis in Python: the kind aliases, the default orders, a
binding-only default size of 10, the periodic knot grid, the Duchon wrap period
and the single-λ penalty. ``gam_terms::basis::position_basis`` now resolves it,
an omitted size takes the formula front door's default, and the payload reports
the basis the fit actually ran on.
"""

import numpy as np

import gamfit


def _positions(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    t = 0.5 + 0.45 * np.sin(np.arange(n) * 1.7)
    y = np.stack([np.sin(2.0 * np.pi * t), np.cos(3.0 * t)], axis=1)
    return t, y


def test_the_reported_basis_replays_the_fit() -> None:
    t, y = _positions()
    out = gamfit.gaussian_reml_fit_positions(t, y)
    assert out["basis_kind"] == "bspline"
    assert out["periodic"] is False and out["period"] is None
    design = gamfit.bspline_basis(t, out["knots_or_centers"], degree=out["basis_order"])
    fitted = np.asarray(out["fitted"], dtype=float)
    np.testing.assert_allclose(
        design @ np.asarray(out["coefficients"], dtype=float),
        fitted,
        rtol=0.0,
        atol=1e-9 * max(1.0, float(np.abs(fitted).max())),
    )


def test_a_periodic_duchon_fit_reports_the_wrap_it_used() -> None:
    """With no ``period``, the basis wraps at the center span plus one mean
    spacing. The payload used to report ``period=None`` although the fit had
    derived a wrap, so a replay could not reconstruct the basis."""
    t, y = _positions()
    out = gamfit.gaussian_reml_fit_positions(t, y, "duchon", 8, periodic=True)
    centers = np.asarray(out["knots_or_centers"], dtype=float)
    span = float(centers.max() - centers.min())
    assert out["periodic"] is True
    assert out["period"] == span + span / (centers.size - 1)


def test_a_periodic_thin_plate_fit_is_the_periodic_duchon_fit() -> None:
    """The 1-D thin-plate spline is Duchon ``m = 2``. Its periodic penalty used
    to be built at no period at all (the center span) while its basis used the
    derived wrap, so the two disagreed."""
    t, y = _positions()
    duchon = gamfit.gaussian_reml_fit_positions(t, y, "duchon", 8, periodic=True)
    thin_plate = gamfit.gaussian_reml_fit_positions(t, y, "thinplate", 8, periodic=True)
    assert thin_plate["basis_kind"] == "thinplate"
    assert thin_plate["basis_order"] == duchon["basis_order"] == 2
    assert thin_plate["period"] == duchon["period"]
    np.testing.assert_array_equal(thin_plate["knots_or_centers"], duchon["knots_or_centers"])
    np.testing.assert_array_equal(thin_plate["penalty"], duchon["penalty"])
    np.testing.assert_array_equal(thin_plate["fitted"], duchon["fitted"])


def test_the_batched_fit_reports_the_same_basis() -> None:
    t, y = _positions()
    offsets = np.array([0, 20, 40], dtype=np.uintp)
    single = gamfit.gaussian_reml_fit_positions(t, y, "duchon", 7)
    batched = gamfit.gaussian_reml_fit_positions_batched(t, y, offsets, "duchon", 7)
    np.testing.assert_array_equal(batched["knots_or_centers"], single["knots_or_centers"])
    np.testing.assert_array_equal(batched["penalty"], single["penalty"])
    assert batched["basis_order"] == single["basis_order"]
