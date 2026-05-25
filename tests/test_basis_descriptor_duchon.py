"""Callable-basis contract for :class:`gamfit.Duchon`.

Covers: shape, autograd-consistency, and bit-equality with the standalone
``gamfit.torch.duchon_basis``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def test_evaluate_shape_1d() -> None:
    centers = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.05, 0.95, 17, dtype=torch.float64)
    phi = spec.evaluate(x)
    assert phi.shape == (17, 8)


def test_jacobian_matches_autograd_1d() -> None:
    centers = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.05, 0.95, 7, dtype=torch.float64, requires_grad=True)
    jac = spec.jacobian(x)
    assert jac.shape == (7, 6, 1)

    def f(xx: torch.Tensor) -> torch.Tensor:
        return spec.evaluate(xx)

    ref = torch.autograd.functional.jacobian(f, x.detach().requires_grad_(True))
    diag = torch.einsum("bmc->bm", ref * torch.eye(7).unsqueeze(1))
    assert torch.allclose(jac[..., 0], diag, atol=1e-4, rtol=1e-4)


def test_bit_equality_with_standalone() -> None:
    import gamfit.torch as gtorch

    centers = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.05, 0.95, 13, dtype=torch.float64)
    phi_desc = spec.evaluate(x)
    phi_fn = gtorch.duchon_basis(x, centers, m=2)
    assert torch.equal(phi_desc, phi_fn)


def test_evaluate_2d_shape() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((9, 2))
    spec = gamfit.Duchon(centers=centers, m=2)
    pts = rng.standard_normal((5, 2))
    x = torch.as_tensor(pts[:, 0], dtype=torch.float64)
    y = torch.as_tensor(pts[:, 1], dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (5, 9)
