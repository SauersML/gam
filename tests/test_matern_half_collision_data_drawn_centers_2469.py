"""#2469: the torch Matérn ν=1/2 boundary with centers drawn from the data.

``4add3103e`` stopped flooring ``r`` at ``1e-12`` in the ν=1/2 radial ratio, so
``matern_input_location_hessian`` refuses on a center instead of returning
``−s·1e12`` as the diagonal. With centers drawn from the data, evaluation rows
sit exactly on centers. These tests pin what ``gamfit.smooth.Matern`` does there.

* d = 1, ν=1/2. ``evaluate`` and ``jacobian`` stay finite. At the coincident
  column the jet is exactly 0, which is the symmetric Clarke subgradient of
  ``exp(−|t − c|/ℓ)`` (torch's convention for ``|x|`` at 0), and it agrees with
  a central difference. ``hessian`` on a center refuses and names the cusp.
  With the rows moved off the centers the Hessian is finite and matches a
  central difference of the jacobian.
* d = 2, ν=1/2. The forward basis already refuses: the collocation assembly
  meets ``r = 0`` on its own diagonal, for any rows. So the Hessian refusal
  cannot be reached in 2-D.
* d = 2, ν=3/2 on the same data-drawn centers. The Hessian exists at a
  collision and matches a central difference of the jacobian.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def _fd_jacobian(spec: Any, coords: list[torch.Tensor], h: float) -> torch.Tensor:
    base = spec.evaluate(*coords)
    rows, width = int(base.shape[0]), int(base.shape[1])
    out = torch.zeros((rows, width, len(coords)), dtype=torch.float64)
    for axis in range(len(coords)):
        plus = [c.clone() for c in coords]
        minus = [c.clone() for c in coords]
        plus[axis] = plus[axis] + h
        minus[axis] = minus[axis] - h
        out[:, :, axis] = (spec.evaluate(*plus) - spec.evaluate(*minus)) / (2.0 * h)
    return out


def _fd_hessian(spec: Any, coords: list[torch.Tensor], h: float) -> torch.Tensor:
    base = spec.jacobian(*coords)
    rows, width, dim = (int(s) for s in base.shape)
    out = torch.zeros((rows, width, dim, dim), dtype=torch.float64)
    for axis in range(dim):
        plus = [c.clone() for c in coords]
        minus = [c.clone() for c in coords]
        plus[axis] = plus[axis] + h
        minus[axis] = minus[axis] - h
        out[:, :, :, axis] = (spec.jacobian(*plus) - spec.jacobian(*minus)) / (2.0 * h)
    return out


def _data_drawn_1d(offset: float = 0.0) -> tuple[Any, list[torch.Tensor]]:
    rng = np.random.default_rng(2469)
    x = np.sort(rng.uniform(-1.0, 1.0, size=12))
    # Rows 0, 3, 6 and 9 are the centers.
    spec = gamfit.smooth.Matern(centers=x[::3].reshape(-1, 1).copy(), nu=0.5, length_scale=0.5)
    return spec, [torch.tensor(x + offset, dtype=torch.float64)]


def _data_drawn_2d(nu: float, offset: float = 0.0) -> tuple[Any, list[torch.Tensor]]:
    rng = np.random.default_rng(2469)
    pts = rng.uniform(-1.0, 1.0, size=(12, 2))
    spec = gamfit.smooth.Matern(centers=pts[::3].copy(), nu=nu, length_scale=0.5)
    shifted = pts + offset
    return spec, [
        torch.tensor(shifted[:, 0], dtype=torch.float64),
        torch.tensor(shifted[:, 1], dtype=torch.float64),
    ]


def test_matern_half_1d_forward_and_jacobian_on_data_drawn_centers_2469() -> None:
    spec, coords = _data_drawn_1d()
    values = spec.evaluate(*coords)
    assert values.shape[0] == 12
    assert torch.isfinite(values).all()
    jac = spec.jacobian(*coords)
    assert torch.isfinite(jac).all()
    for row in (0, 3, 6, 9):
        coincident = int(torch.argmax(values[row]))
        assert float(jac[row, coincident, 0]) == 0.0, (
            f"row {row} on a center: the jet must be the symmetric subgradient 0, "
            f"got {float(jac[row, coincident, 0])}"
        )
    fd = _fd_jacobian(spec, coords, h=1e-6)
    gap = float((jac - fd).abs().max())
    assert gap < 1e-6, f"jacobian vs central difference on data-drawn centers: max gap {gap:.3e}"


def test_matern_half_1d_hessian_refuses_on_a_data_drawn_center_2469() -> None:
    spec, coords = _data_drawn_1d()
    with pytest.raises(gamfit.errors.GamError, match="cusp"):
        spec.hessian(*coords)


def test_matern_half_1d_hessian_is_finite_off_the_data_drawn_centers_2469() -> None:
    spec, coords = _data_drawn_1d(offset=0.01)
    hess = spec.hessian(*coords)
    assert torch.isfinite(hess).all()
    fd = _fd_hessian(spec, coords, h=1e-6)
    scale = max(1.0, float(hess.abs().max()))
    gap = float((hess - fd).abs().max())
    assert gap < 1e-5 * scale, f"hessian vs central difference off the centers: max gap {gap:.3e}"


def test_matern_half_2d_forward_refuses_collocation_on_its_own_centers_2469() -> None:
    spec, coords = _data_drawn_2d(0.5, offset=0.05)
    with pytest.raises(gamfit.errors.GamError, match="singular Laplacian at center collisions"):
        spec.evaluate(*coords)


def test_matern_three_halves_2d_hessian_on_data_drawn_centers_2469() -> None:
    spec, coords = _data_drawn_2d(1.5)
    hess = spec.hessian(*coords)
    assert torch.isfinite(hess).all()
    fd = _fd_hessian(spec, coords, h=1e-6)
    scale = max(1.0, float(hess.abs().max()))
    gap = float((hess - fd).abs().max())
    assert gap < 1e-5 * scale, f"nu=3/2 hessian vs central difference at collisions: max gap {gap:.3e}"
