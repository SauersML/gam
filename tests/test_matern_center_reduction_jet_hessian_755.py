"""#755: one Matérn center reduction feeds the forward basis, the jet and the Hessian.

The forward Matérn basis drops centers beyond the kernel's data-supported rank
(#755). The input-location jet used the same reduction, but the Hessian
evaluated every requested center. So ``_JetFn.backward`` contracted a
``(B, K', d)`` cotangent against a ``(B, K, d, d)`` Hessian, and ``hessian()``
failed on the einsum shape whenever a center was dropped. A duplicated center
forces the reduction. With it, the basis, jacobian and Hessian must share the
reduced width and match central differences.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def _duplicated_center_2d() -> tuple[Any, list[torch.Tensor], int]:
    rng = np.random.default_rng(755)
    centers = rng.uniform(-1.0, 1.0, size=(5, 2))
    # Center 4 repeats center 1: the realized kernel block has two equal columns.
    centers[4] = centers[1]
    spec = gamfit.smooth.Matern(centers=centers, nu=1.5, length_scale=0.6)
    pts = rng.uniform(-1.0, 1.0, size=(9, 2))
    coords = [
        torch.tensor(pts[:, 0], dtype=torch.float64),
        torch.tensor(pts[:, 1], dtype=torch.float64),
    ]
    return spec, coords, centers.shape[0]


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


def test_matern_duplicated_center_is_reduced_in_the_forward_basis_755() -> None:
    spec, coords, requested = _duplicated_center_2d()
    values = spec.evaluate(*coords)
    assert values.shape == (9, requested - 1)
    assert torch.isfinite(values).all()


def test_matern_jacobian_shares_the_reduced_width_and_matches_fd_755() -> None:
    spec, coords, requested = _duplicated_center_2d()
    jac = spec.jacobian(*coords)
    assert jac.shape == (9, requested - 1, 2)
    fd = _fd_jacobian(spec, coords, h=1e-6)
    gap = float((jac - fd).abs().max())
    assert gap < 1e-6, f"jacobian vs central difference: max gap {gap:.3e}"


def test_matern_hessian_shares_the_reduced_width_and_matches_fd_755() -> None:
    spec, coords, requested = _duplicated_center_2d()
    hess = spec.hessian(*coords)
    assert hess.shape == (9, requested - 1, 2, 2)
    assert torch.isfinite(hess).all()
    fd = _fd_hessian(spec, coords, h=1e-6)
    scale = max(1.0, float(hess.abs().max()))
    gap = float((hess - fd).abs().max())
    assert gap < 1e-5 * scale, f"hessian vs central difference: max gap {gap:.3e}"
