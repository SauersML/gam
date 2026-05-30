"""Cross-language parity / regression lock for the sphere chart basis + jet.

Issue #404. The chart-local sphere basis and its first-derivative (lat/lon)
jet used to be implemented twice — once in the core Rust SAE path
(``SphereChartEvaluator``) and once in the PyFFI ``sphere_chart_basis_with_jet``
helper — and the two diverged in how they gated *saturated-latitude*
derivatives. Core Rust clamps latitude to ``[-pi/2, pi/2]`` and multiplies every
latitude partial by a ``chain_lat`` factor that is ``0`` outside the open
interval (the clamp truncates the dependence on the raw input there, so the
correct derivative w.r.t. the raw coordinate is zero). The PyFFI copy computed
the analogous derivatives but *did not* apply that gating, so the same model
produced different latitude derivatives depending on whether the sphere basis
was evaluated through the core path or the Python/Torch path that calls PyFFI.

The two paths have since been unified — both route through the single
``sphere_chart_basis_jet`` function — and a Rust-side parity guard pins the core
surface. This test pins the *Python* surface, i.e. the boundary where the drift
actually lived, so an ungated PyFFI/Torch copy can never silently reappear:

  * the basis is the clamped unit-sphere embedding ``[1, x, y, z, xy, yz, xz]``;
  * the chart penalty diagonal is the shared ``[1e-8, 1, 1, 1, 4, 4, 4]``;
  * the longitude jet matches the analytic reference (never gated);
  * the latitude jet matches central finite differences strictly inside the
    interior *and* strictly beyond the clamp (where it must be zero);
  * at the saturated latitudes the latitude jet is *exactly* zero — an ungated
    copy would return ``-sin(lat)cos(lon) != 0`` at a pole and fail here;
  * the Torch autograd backward (``_BasisWithJetFn``) contracts the gated jet,
    so the gradient w.r.t. a pole latitude is exactly zero.
"""
from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
gamfit = pytest.importorskip("gamfit")

HALF_PI = math.pi / 2.0


def _rust_sphere(coords: np.ndarray):
    """Evaluate the sphere chart through the PyFFI ``basis_with_jet`` dispatch.

    This is the exact entry point the Torch manifold-SAE path uses
    (``kind="sphere"``), so it exercises the PyFFI helper the issue named.
    """
    rust = gamfit._rust  # type: ignore[attr-defined]
    phi, jet, penalty = rust.basis_with_jet(
        "sphere", np.ascontiguousarray(coords, dtype=np.float64), {}
    )
    return np.asarray(phi), np.asarray(jet), np.asarray(penalty)


def _reference_basis(lat: float, lon: float) -> np.ndarray:
    """Independent NumPy reference for the 7-column clamped embedding."""
    latc = min(max(lat, -HALF_PI), HALF_PI)
    x = math.cos(latc) * math.cos(lon)
    y = math.cos(latc) * math.sin(lon)
    z = math.sin(latc)
    return np.array([1.0, x, y, z, x * y, y * z, x * z])


def _reference_jet(lat: float, lon: float) -> np.ndarray:
    """Independent NumPy reference for the (7, 2) jet [d/dlat, d/dlon].

    Latitude partials carry the ``chain_lat`` gate: zero unless ``lat`` is
    strictly inside ``(-pi/2, pi/2)``.
    """
    latc = min(max(lat, -HALF_PI), HALF_PI)
    interior = -HALF_PI < lat < HALF_PI
    g = 1.0 if interior else 0.0
    clat, slat = math.cos(latc), math.sin(latc)
    clon, slon = math.cos(lon), math.sin(lon)
    x = clat * clon
    y = clat * slon
    z = slat
    dx_dlat = -slat * clon * g
    dy_dlat = -slat * slon * g
    dz_dlat = clat * g
    dx_dlon = -clat * slon
    dy_dlon = clat * clon
    dz_dlon = 0.0
    jet = np.zeros((7, 2))
    # column 0 (constant) has zero derivatives
    jet[1] = [dx_dlat, dx_dlon]
    jet[2] = [dy_dlat, dy_dlon]
    jet[3] = [dz_dlat, dz_dlon]
    jet[4] = [dx_dlat * y + x * dy_dlat, dx_dlon * y + x * dy_dlon]
    jet[5] = [dy_dlat * z + y * dz_dlat, dy_dlon * z + y * dz_dlon]
    jet[6] = [dx_dlat * z + x * dz_dlat, dx_dlon * z + x * dz_dlon]
    return jet


# Interior points stay >1e-3 away from the clamp; "beyond" points stay >1e-3
# past it. Both margins exceed the finite-difference step so a central stencil
# never straddles the kink at +/- pi/2.
_INTERIOR = [(-1.2, -2.4), (-0.25, 0.0), (0.35, 0.9), (0.8, 2.1), (1.55, -1.1)]
_BOUNDARY = [(HALF_PI, 0.4), (-HALF_PI, -1.1)]
_BEYOND = [(2.3, 0.7), (-3.0, 1.9)]


def test_sphere_basis_matches_reference():
    coords = np.array(_INTERIOR + _BOUNDARY + _BEYOND, dtype=np.float64)
    phi, jet, penalty = _rust_sphere(coords)
    assert phi.shape == (coords.shape[0], 7)
    assert jet.shape == (coords.shape[0], 7, 2)
    np.testing.assert_array_almost_equal(
        np.diag(penalty), [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0], decimal=15
    )
    for row, (lat, lon) in enumerate(coords):
        np.testing.assert_allclose(
            phi[row], _reference_basis(lat, lon), rtol=0.0, atol=1e-12,
            err_msg=f"basis mismatch at lat={lat}, lon={lon}",
        )
        np.testing.assert_allclose(
            jet[row], _reference_jet(lat, lon), rtol=0.0, atol=1e-12,
            err_msg=f"jet mismatch at lat={lat}, lon={lon}",
        )


def test_latitude_jet_is_gated_to_zero_at_and_beyond_the_poles():
    """The drift lock: an ungated PyFFI copy returns -sin(lat)cos(lon) != 0 for
    the latitude partial at a pole; the gated truth is exactly zero."""
    coords = np.array(_BOUNDARY + _BEYOND, dtype=np.float64)
    _phi, jet, _pen = _rust_sphere(coords)
    # Every latitude partial (last-axis index 0) must be bit-zero.
    lat_partials = jet[:, :, 0]
    assert np.all(lat_partials == 0.0), (
        "saturated-latitude jet leaked a non-zero lat derivative — the "
        "chain_lat gating is not being applied on the PyFFI path:\n"
        f"{lat_partials}"
    )
    # Longitude partials are never gated and stay non-trivial.
    lon_partials = jet[:, :, 1]
    assert np.any(lon_partials != 0.0)


def test_jet_matches_finite_differences_off_the_kink():
    """Central differences are a valid oracle strictly inside the interior and
    strictly beyond the clamp (not *at* the kink, where the clamp is
    non-differentiable). Beyond the clamp the basis is constant in latitude, so
    the FD latitude partial is ~0, confirming the gated value is the correct
    derivative and not merely a convention."""
    h = 1e-6
    coords = np.array(_INTERIOR + _BEYOND, dtype=np.float64)
    _phi, jet, _pen = _rust_sphere(coords)
    for axis in (0, 1):  # 0 = lat, 1 = lon
        plus = coords.copy()
        minus = coords.copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        phi_plus, _, _ = _rust_sphere(plus)
        phi_minus, _, _ = _rust_sphere(minus)
        fd = (phi_plus - phi_minus) / (2.0 * h)
        np.testing.assert_allclose(
            jet[:, :, axis], fd, rtol=0.0, atol=1e-6,
            err_msg=f"jet axis {axis} disagrees with finite differences",
        )


def test_torch_autograd_backward_uses_the_gated_jet():
    """The production consumer: ``_BasisWithJetFn.backward`` contracts the saved
    jet, so the gradient w.r.t. a pole latitude must be exactly zero while the
    interior gradient matches finite differences of the same scalar."""
    torch = pytest.importorskip("torch")
    mod = import_module("gamfit.torch.manifold_sae")
    basis_fn = mod._BasisWithJetFn

    coords = [
        (1.2, 0.5),        # interior
        (HALF_PI, 0.7),    # north pole (saturated)
        (-HALF_PI, -1.1),  # south pole (saturated)
        (-0.4, 2.0),       # interior
    ]
    t = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    phi = basis_fn.apply(t, "sphere", "{}")
    phi.sum().backward()
    grad = t.grad
    assert grad is not None and torch.isfinite(grad).all()

    # Pole rows (1, 2): the latitude gradient is gated to exactly zero.
    assert grad[1, 0].item() == 0.0
    assert grad[2, 0].item() == 0.0
    # The longitude gradient at a pole is still defined and generally non-zero.
    assert torch.isfinite(grad[1, 1]).item()

    # Interior rows: backward gradient matches central FD of phi.sum().
    def _phi_sum(coords_tensor: "torch.Tensor") -> float:
        with torch.no_grad():
            out = basis_fn.apply(coords_tensor, "sphere", "{}")
        return float(out.sum().item())

    h = 1e-6
    base = torch.tensor(coords, dtype=torch.float64)
    for row in (0, 3):
        for axis in (0, 1):
            plus = base.clone()
            minus = base.clone()
            plus[row, axis] += h
            minus[row, axis] -= h
            fd = (_phi_sum(plus) - _phi_sum(minus)) / (2.0 * h)
            assert abs(grad[row, axis].item() - fd) <= 1e-6, (
                f"row {row} axis {axis}: autograd {grad[row, axis].item()} "
                f"vs finite difference {fd}"
            )
