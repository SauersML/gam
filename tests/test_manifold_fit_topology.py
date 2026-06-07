"""End-to-end topological-continuity guarantees for the geometric (manifold) smooths.

These are *fit-then-predict* regression checks, complementary to the basis-level
periodicity tests (``test_basis_descriptor_cylinder.py``, ``test_issue_225_periodic_spline_curve_basis.py``)
and the 1-D periodic-Duchon seam test (``test_periodic_duchon_positions_580_581.py``).
They pin the behaviour a *user* sees from a fitted surface:

  * a 2-D torus smooth ``te(u, v, periodic=[0,1], period=[2*pi, 2*pi])`` must wrap on
    BOTH axes (no seam at u=0/2pi or v=0/2pi);
  * a ``sphere(lat, lon)`` smooth must wrap in longitude (no seam at lon=0/2pi) and
    must collapse the pole to a single point (all longitudes at lat=pi/2 agree).

A regression in the periodic-margin wiring or the spherical chart would surface here as
a nonzero seam gap or a fanned-out pole, even if the basis-level contracts still hold.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit

TWO_PI = 2.0 * np.pi
# The periodic basis / spherical chart make the seam and the pole exact identities;
# only IEEE-754 round-off should remain. A real regression (broken wrap, fanned pole)
# is O(1e-1 .. 1e0), so this tolerance separates the two cleanly.
SEAM_TOL = 1e-6


def test_torus_fit_wraps_on_both_axes() -> None:
    rng = np.random.default_rng(50)
    n = 1200
    u = rng.uniform(0.0, TWO_PI, n)
    v = rng.uniform(0.0, TWO_PI, n)
    y = np.sin(u) + np.cos(v) + 0.5 * np.sin(u + v) + rng.normal(0.0, 0.2, n)

    model = gamfit.fit(
        pd.DataFrame({"u": u, "v": v, "y": y}),
        "y ~ te(u, v, periodic=[0,1], period=[2*pi, 2*pi])",
    )

    probe = np.linspace(0.1, TWO_PI - 0.1, 12)
    # u-axis seam: f(0, v) must equal f(2*pi, v)
    f_u0 = np.asarray(model.predict(pd.DataFrame({"u": np.zeros(12), "v": probe})))
    f_u2pi = np.asarray(model.predict(pd.DataFrame({"u": np.full(12, TWO_PI), "v": probe})))
    u_gap = float(np.max(np.abs(f_u0 - f_u2pi)))
    # v-axis seam: f(u, 0) must equal f(u, 2*pi)
    f_v0 = np.asarray(model.predict(pd.DataFrame({"u": probe, "v": np.zeros(12)})))
    f_v2pi = np.asarray(model.predict(pd.DataFrame({"u": probe, "v": np.full(12, TWO_PI)})))
    v_gap = float(np.max(np.abs(f_v0 - f_v2pi)))

    assert u_gap <= SEAM_TOL, f"torus u-axis seam not continuous: max|f(0,v)-f(2pi,v)|={u_gap:.3e}"
    assert v_gap <= SEAM_TOL, f"torus v-axis seam not continuous: max|f(u,0)-f(u,2pi)|={v_gap:.3e}"


def test_sphere_fit_wraps_in_longitude_and_collapses_pole() -> None:
    rng = np.random.default_rng(51)
    n = 1500
    lat = np.arcsin(rng.uniform(-1.0, 1.0, n))  # area-uniform latitudes in [-pi/2, pi/2]
    lon = rng.uniform(0.0, TWO_PI, n)
    y = np.sin(lat) + np.cos(lon) * np.cos(lat) + rng.normal(0.0, 0.2, n)

    model = gamfit.fit(
        pd.DataFrame({"lat": lat, "lon": lon, "y": y}),
        "y ~ sphere(lat, lon, radians=true)",
    )

    lats = np.linspace(-1.2, 1.2, 12)
    f_lon0 = np.asarray(model.predict(pd.DataFrame({"lat": lats, "lon": np.zeros(12)})))
    f_lon2pi = np.asarray(model.predict(pd.DataFrame({"lat": lats, "lon": np.full(12, TWO_PI)})))
    lon_gap = float(np.max(np.abs(f_lon0 - f_lon2pi)))
    assert lon_gap <= SEAM_TOL, (
        f"sphere longitude seam not continuous: max|f(lat,0)-f(lat,2pi)|={lon_gap:.3e}"
    )

    # The north pole is a single point: every longitude at lat=pi/2 must agree.
    lons = np.linspace(0.0, TWO_PI, 16, endpoint=False)
    f_pole = np.asarray(
        model.predict(pd.DataFrame({"lat": np.full(lons.size, np.pi / 2.0), "lon": lons}))
    )
    pole_spread = float(np.max(f_pole) - np.min(f_pole))
    assert pole_spread <= SEAM_TOL, (
        f"sphere north pole fans out across longitude: spread={pole_spread:.3e}"
    )
