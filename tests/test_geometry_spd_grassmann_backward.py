"""End-to-end analytic backward checks for curved manifold descriptors."""

from __future__ import annotations

import pytest

from gamfit import RustExtensionUnavailableError, manifolds


torch = pytest.importorskip("torch")


def setup_module(module):  # noqa: D401 - pytest hook
    from gamfit._binding import rust_module

    try:
        rust_module()
    except RustExtensionUnavailableError as exc:  # pragma: no cover - env specific
        pytest.skip(f"gamfit._rust unavailable: {exc}")


def test_grassmann_k2_exp_backward_matches_torch_gradcheck():
    manifold = manifolds.Grassmann(k=2, n=4)
    point = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    tangent = torch.tensor(
        [0.2, -0.1, 0.05, 0.3, 0.7, -0.2, 0.1, 1.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda p, v: manifold.exp(p, v),
        (point, tangent),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_spd_exp_backward_matches_gradcheck_on_symmetric_coordinates():
    manifold = manifolds.Spd(n=2)

    def symmetric(coords):
        return torch.stack((coords[0], coords[1], coords[1], coords[2]))

    point_coords = torch.tensor(
        [1.3, 0.15, 0.9], dtype=torch.float64, requires_grad=True
    )
    tangent_coords = torch.tensor(
        [0.1, 0.04, -0.2], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda p, v: manifold.exp(symmetric(p), symmetric(v)),
        (point_coords, tangent_coords),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )
