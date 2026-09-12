"""Family-wide contract tests for BasisDescriptor jacobian/hessian.

For every concrete descriptor in
``gamfit.{BSpline, Matern, Duchon, Sphere, PeriodicSplineCurve,
TensorBSpline, Pca}`` and the topology helpers ``Cylinder`` / ``Torus``:

  * ``spec.jacobian(*coords)`` matches a central finite-difference estimate of
    the basis derivative wrt the input coordinates;
  * ``spec.hessian(*coords)`` returns finite values of shape ``(B, M, d, d)``.

Any exception fails, ``NotImplementedError`` included: every descriptor
advertises both derivatives, so a refusal is an owed derivative, not a pass.
The per-issue tests at the bottom pin the original failures: #230
(BSpline/Matern hessian leaked "element 0 of tensors does not require grad"),
#233 (Cylinder/Torus jacobian hit the refused dense periodic
``bspline_basis_derivative``) and #234 (Duchon jacobian reported "primary
penalty was not built").
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit
from gamfit import topology as _topology


def _fd_jacobian(
    spec: Any, coords: list[torch.Tensor], h: float = 1e-4
) -> torch.Tensor:
    """Central FD estimate of ∂Φ/∂x with shape (B, M, d)."""
    base = spec.evaluate(*[c.detach().clone() for c in coords])
    B, M = int(base.shape[0]), int(base.shape[1])
    d = len(coords)
    out = torch.zeros((B, M, d), dtype=base.dtype)
    for j in range(d):
        plus = [c.detach().clone() for c in coords]
        minus = [c.detach().clone() for c in coords]
        plus[j] = plus[j] + h
        minus[j] = minus[j] - h
        f_plus = spec.evaluate(*plus)
        f_minus = spec.evaluate(*minus)
        out[:, :, j] = (f_plus - f_minus) / (2 * h)
    return out


def _check_jacobian(spec: Any, coords: list[torch.Tensor], tol: float = 5e-3) -> None:
    jac = spec.jacobian(*coords)
    fd = _fd_jacobian(spec, coords)
    jac_np = jac.detach().to(dtype=torch.float64).cpu().numpy()
    fd_np = fd.detach().to(dtype=torch.float64).cpu().numpy()
    assert jac_np.shape == fd_np.shape, (
        f"{type(spec).__name__}.jacobian shape mismatch: {jac_np.shape} vs FD {fd_np.shape}"
    )
    diff = float(np.max(np.abs(jac_np - fd_np)))
    scale = float(np.max(np.abs(fd_np))) + 1e-12
    assert diff / scale < tol, (
        f"{type(spec).__name__}.jacobian disagrees with FD: "
        f"max|Δ|={diff:.3e}, scale={scale:.3e}, rel={diff/scale:.3e}"
    )


def _check_hessian(spec: Any, coords: list[torch.Tensor]) -> None:
    hess = spec.hessian(*coords)
    B = int(coords[0].shape[0])
    d = len(coords)
    expected_shape = (B, int(spec.basis_size), d, d)
    got = tuple(int(s) for s in hess.shape)
    assert got == expected_shape, (
        f"{type(spec).__name__}.hessian shape {got} != expected {expected_shape}"
    )
    assert torch.isfinite(hess).all().item(), (
        f"{type(spec).__name__}.hessian contained non-finite values"
    )


# ---------------------------------------------------------------- builders


def _bspline_1d() -> tuple[Any, list[torch.Tensor]]:
    knots = np.linspace(0.0, 1.0, 8 + 2 * 3)
    spec = gamfit.BSpline(knots=knots, degree=3, periodic=False)
    x = torch.linspace(0.1, 0.9, 11, dtype=torch.float64)
    return spec, [x]


def _periodic_bspline_1d() -> tuple[Any, list[torch.Tensor]]:
    spec = gamfit.BSpline(knots=10, degree=3, periodic=True)
    x = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    spec.evaluate(x)  # let auto-knots resolve, if needed
    return spec, [x]


def _matern_2d() -> tuple[Any, list[torch.Tensor]]:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((6, 2))
    spec = gamfit.Matern(centers=centers, nu=1.5, length_scale=0.5)
    pts = rng.standard_normal((7, 2))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    return spec, [x, y]


def _duchon_1d() -> tuple[Any, list[torch.Tensor]]:
    centers = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    return spec, [x]


def _sphere() -> tuple[Any, list[torch.Tensor]]:
    spec = gamfit.Sphere(n_centers=6, penalty_order=2, kernel="sobolev", radians=True)
    # Stay away from poles and the seam.
    lat = torch.linspace(0.2, 1.0, 8, dtype=torch.float64)
    lon = torch.linspace(0.3, 2.5, 8, dtype=torch.float64)
    return spec, [lat, lon]


def _periodic_curve() -> tuple[Any, list[torch.Tensor]]:
    spec = gamfit.PeriodicSplineCurve(n_knots=10, degree=3, output_dim=1)
    t = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    return spec, [t]


def _tensor_bspline_open() -> tuple[Any, list[torch.Tensor]]:
    spec = gamfit.TensorBSpline(
        marginals=[
            gamfit.BSpline(knots=8, degree=3, periodic=False),
            gamfit.BSpline(knots=6, degree=3, periodic=False),
        ]
    )
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    y = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    return spec, [x, y]


def _cylinder() -> tuple[Any, list[torch.Tensor]]:
    spec = _topology.Cylinder(n_knots=(7, 4))
    theta = torch.linspace(0.0, 1.0 - 1e-3, 11, dtype=torch.float64)
    ell = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    return spec, [theta, ell]


def _torus() -> tuple[Any, list[torch.Tensor]]:
    spec = _topology.Torus(n_knots=(6, 6))
    a = torch.linspace(0.0, 1.0 - 1e-3, 11, dtype=torch.float64)
    b = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    return spec, [a, b]


def _pca() -> tuple[Any, list[torch.Tensor]]:
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((5, 3))
    spec = gamfit.Pca(basis=basis)
    coords = [
        torch.tensor(rng.standard_normal(6), dtype=torch.float64) for _ in range(5)
    ]
    return spec, coords


BUILDERS: dict[str, Callable[[], tuple[Any, list[torch.Tensor]]]] = {
    "bspline_1d": _bspline_1d,
    "periodic_bspline_1d": _periodic_bspline_1d,
    "matern_2d": _matern_2d,
    "duchon_1d": _duchon_1d,
    "sphere": _sphere,
    "periodic_curve": _periodic_curve,
    "tensor_bspline_open": _tensor_bspline_open,
    "cylinder": _cylinder,
    "torus": _torus,
    "pca": _pca,
}


@pytest.mark.parametrize("name", sorted(BUILDERS.keys()))
def test_jacobian_matches_fd(name: str) -> None:
    """For every descriptor, jacobian() matches a central finite difference."""
    spec, coords = BUILDERS[name]()
    _check_jacobian(spec, coords)


@pytest.mark.parametrize("name", sorted(BUILDERS.keys()))
def test_hessian_is_finite_with_basis_shape(name: str) -> None:
    """For every descriptor, hessian() returns finite values of shape (B, M, d, d)."""
    spec, coords = BUILDERS[name]()
    _check_hessian(spec, coords)


# ------------------------------------------------------ per-issue regressions


def test_issue_230_bspline_hessian_is_finite() -> None:
    """Issue #230 — BSpline.hessian leaked the autograd-graph error."""
    spec, coords = _bspline_1d()
    _check_hessian(spec, coords)


def test_issue_230_matern_hessian_is_finite() -> None:
    """Issue #230 — Matern.hessian leaked the autograd-graph error."""
    spec, coords = _matern_2d()
    _check_hessian(spec, coords)


def test_issue_233_cylinder_jacobian_matches_fd() -> None:
    """Issue #233 — Cylinder.jacobian leaked the periodic-bspline Rust rejection."""
    spec, coords = _cylinder()
    _check_jacobian(spec, coords)


def test_issue_233_torus_jacobian_matches_fd() -> None:
    """Issue #233 — Torus.jacobian leaked the periodic-bspline Rust rejection."""
    spec, coords = _torus()
    _check_jacobian(spec, coords)


def test_issue_234_duchon_jacobian_matches_fd() -> None:
    """Issue #234 — Duchon.jacobian reported 'primary penalty was not built'."""
    spec, coords = _duchon_1d()
    _check_jacobian(spec, coords)
