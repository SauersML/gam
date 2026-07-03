"""Regression: differentiable ``duchon_basis`` input-location VJP (gam#2097).

``gamfit.torch._basis.duchon_basis`` used to build its forward design with a
*different* Rust builder (``build_duchon_basis``) than the one that produced the
analytic jets its backward contracts against
(``build_duchon_basis_design_and_jets`` via ``duchon_basis_with_jets``). On the
dense path ``build_duchon_basis`` applies a batch-global data-metric radial
reparameterization ``V`` (gam#1355) that the jet builder does not, so the jets
were the derivative of a matrix the forward never returned: the analytic input
gradient came out wrong (nearly constant across evaluation points) and, when the
two builders returned different-width designs, the backward einsum raised a
shape-mismatch ``RuntimeError``.

These tests pin the contract every other differentiable basis primitive in
``tests/test_torch_basis_vjp.py`` already satisfies: the autograd input gradient
of ``duchon_basis`` equals central finite differences of *its own forward*.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gamfit.torch._basis import duchon_basis  # noqa: E402


def _central_fd_grad(points, centers, m, weights, eps=1e-6):
    """Central-difference gradient of ``(duchon_basis(points)*weights).sum()``."""
    fd = torch.zeros_like(points)
    d = points.shape[1]
    for a in range(points.shape[0]):
        for b in range(d):
            pp = points.clone()
            pp[a, b] += eps
            pm = points.clone()
            pm[a, b] -= eps
            plus = (duchon_basis(pp, centers=centers, m=m) * weights).sum()
            minus = (duchon_basis(pm, centers=centers, m=m) * weights).sum()
            fd[a, b] = (plus - minus) / (2.0 * eps)
    return fd


@pytest.mark.parametrize("m", [2, 3])
def test_duchon_basis_vjp_matches_fd_of_own_forward(m: int) -> None:
    points = torch.tensor(
        [[0.15, 0.25], [0.35, 0.65], [0.55, 0.45], [0.75, 0.85], [0.25, 0.55]],
        dtype=torch.float64,
    )
    centers = torch.tensor(
        [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.5, 0.5], [0.3, 0.7]],
        dtype=torch.float64,
    )

    design = duchon_basis(points, centers=centers, m=m)
    weights = torch.linspace(-1.0, 1.0, design.shape[1], dtype=torch.float64)

    p = points.clone().requires_grad_(True)
    (duchon_basis(p, centers=centers, m=m) * weights).sum().backward()
    analytic = p.grad

    fd = _central_fd_grad(points, centers, m, weights)

    # Before the fix this was ~0.7; the analytic gradient must now track the FD
    # of its own forward to finite-difference precision.
    assert (analytic - fd).abs().max().item() < 1e-6


def test_duchon_basis_vjp_width_reducing_config_runs_and_is_exact() -> None:
    # K=10 centers in 2D is exactly the configuration where the old two-builder
    # path returned different widths (forward 9 cols, jet 10) and the backward
    # einsum raised ``RuntimeError``. It must now run and match FD.
    g = torch.Generator().manual_seed(0)
    centers = torch.rand(10, 2, generator=g, dtype=torch.float64)
    points = torch.rand(4, 2, generator=g, dtype=torch.float64)

    design = duchon_basis(points, centers=centers, m=2)
    assert design.shape == (4, 10)
    weights = torch.linspace(-1.0, 1.0, design.shape[1], dtype=torch.float64)

    p = points.clone().requires_grad_(True)
    (duchon_basis(p, centers=centers, m=2) * weights).sum().backward()
    analytic = p.grad

    fd = _central_fd_grad(points, centers, 2, weights)
    assert (analytic - fd).abs().max().item() < 1e-6
