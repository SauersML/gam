"""Callable-basis contract for :class:`gamfit.BSpline`.

Covers: shape conventions, autograd-consistency between ``.evaluate`` and
``.jacobian`` / ``.hessian``, partition-of-unity, and bit-equality with the
standalone ``gamfit.torch.bspline_basis``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def _make_spec() -> "gamfit.BSpline":
    knots = np.linspace(0.0, 1.0, 8 + 2 * 3)  # degree=3 clamped layout
    return gamfit.BSpline(knots=knots, degree=3, periodic=False)


def test_evaluate_shape_and_dtype() -> None:
    spec = _make_spec()
    x = torch.linspace(0.05, 0.95, 17, dtype=torch.float64)
    phi = spec.evaluate(x)
    assert phi.shape == (17, spec.basis_size)
    assert phi.dtype == torch.float64


def test_jacobian_shape_and_autograd_match() -> None:
    spec = _make_spec()
    x = torch.linspace(0.05, 0.95, 11, dtype=torch.float64, requires_grad=True)
    jac = spec.jacobian(x)
    M = spec.basis_size
    assert jac.shape == (11, M, 1)

    # Autograd reference: jacobian via torch.autograd.functional.
    def f(xx: torch.Tensor) -> torch.Tensor:
        return spec.evaluate(xx)

    ref = torch.autograd.functional.jacobian(f, x.detach().requires_grad_(True))
    # ref has shape (B, M, B); per-row independence -> diag along (0, 2).
    diag = torch.einsum("bmc->bm", ref * torch.eye(11).unsqueeze(1))
    assert torch.allclose(jac[..., 0], diag, atol=1e-5, rtol=1e-5)


def test_hessian_shape() -> None:
    spec = _make_spec()
    x = torch.linspace(0.05, 0.95, 9, dtype=torch.float64)
    H = spec.hessian(x)
    assert H.shape == (9, spec.basis_size, 1, 1)


def test_partition_of_unity_interior() -> None:
    spec = _make_spec()
    x = torch.linspace(0.1, 0.9, 31, dtype=torch.float64)
    phi = spec.evaluate(x)
    row_sums = phi.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-9)


def test_bit_equality_with_standalone() -> None:
    """``spec.evaluate(x) == gamfit.torch.bspline_basis(x, ...)`` exactly."""
    import gamfit.torch as gtorch

    spec = _make_spec()
    x = torch.linspace(0.05, 0.95, 13, dtype=torch.float64)
    phi_desc = spec.evaluate(x)
    phi_fn = gtorch.bspline_basis(x, spec.knots, degree=spec.degree, periodic=spec.periodic)
    assert torch.equal(phi_desc, phi_fn)
