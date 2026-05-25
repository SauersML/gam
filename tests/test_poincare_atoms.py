"""Tests for :class:`gamfit.PoincareAtoms`.

Cover the contract claims made in the docstring:

* ``distance(a, a) == 0``.
* Möbius addition with the origin is the identity on either side.
* Poincaré-ball and Lorentz decoders agree on small inputs.
* ``forward(z).backward()`` runs and produces gradients for atoms.
* Points placed near the ball boundary do not produce NaN/inf.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
gamfit = pytest.importorskip("gamfit")
pytest.importorskip("gamfit._rust")


def _atoms(**kwargs):
    return gamfit.PoincareAtoms(**kwargs)


def test_distance_self_is_zero() -> None:
    atoms = _atoms(F=3, ball_dim=4)
    a = torch.tensor([[0.1, -0.2, 0.05, 0.0]], dtype=torch.float64)
    d = atoms.distance(a, a)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-8)


def test_mobius_add_with_origin_is_identity() -> None:
    atoms = _atoms(F=2, ball_dim=3)
    v = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float64)
    zero = torch.zeros_like(v)
    assert torch.allclose(atoms.mobius_add(zero, v), v, atol=1e-12)
    assert torch.allclose(atoms.mobius_add(v, zero), v, atol=1e-12)


def test_distance_matches_hand_computation() -> None:
    atoms = _atoms(F=1, ball_dim=2)
    a = torch.tensor([[0.3, 0.1]], dtype=torch.float64)
    b = torch.tensor([[-0.2, 0.4]], dtype=torch.float64)
    diff_sq = float(((a - b) ** 2).sum())
    a_sq = float((a * a).sum())
    b_sq = float((b * b).sum())
    expected = math.acosh(1.0 + 2.0 * diff_sq / ((1.0 - a_sq) * (1.0 - b_sq)))
    d = float(atoms.distance(a, b).item())
    assert d == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_poincare_and_lorentz_paths_agree() -> None:
    torch.manual_seed(0)
    F, d = 5, 4
    atoms_p = _atoms(F=F, ball_dim=d, lorentz=False)
    atoms_l = _atoms(F=F, ball_dim=d, lorentz=True)
    # Force matching atom positions, well inside the ball.
    with torch.no_grad():
        shared = (torch.randn(F, d, dtype=torch.float64) * 0.1)
        atoms_p.atoms.data = shared.clone().to(atoms_p.atoms.dtype)
        atoms_l.atoms.data = shared.clone().to(atoms_l.atoms.dtype)

    z = torch.randn(7, F, dtype=atoms_p.atoms.dtype) * 0.2
    x_p = atoms_p(z)
    x_l = atoms_l(z)
    assert torch.allclose(x_p, x_l, atol=1e-5, rtol=1e-5)


def test_forward_backward_gradients_flow() -> None:
    atoms = _atoms(F=4, ball_dim=3)
    z = torch.randn(6, 4, requires_grad=True)
    x_hat = atoms(z)
    loss = (x_hat * x_hat).sum()
    loss.backward()
    assert atoms.atoms.grad is not None
    assert torch.isfinite(atoms.atoms.grad).all()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_near_boundary_inputs_are_finite() -> None:
    # Default curvature -1 → ball is the open unit ball.
    atoms = _atoms(F=1, ball_dim=2)
    raw = torch.tensor([[0.999, 0.0]], dtype=torch.float64)
    projected = atoms.project_into_ball(raw)
    # The projector keeps norm strictly < 1.
    assert torch.linalg.vector_norm(projected, dim=-1).item() < 1.0
    # Distance to itself near the boundary is still zero (and finite).
    d_self = atoms.distance(projected, projected)
    assert torch.isfinite(d_self).all()
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=1e-6)

    # Forward through a forced-near-boundary atom must not NaN.
    with torch.no_grad():
        atoms.atoms.data = projected.clone().to(atoms.atoms.dtype)
    z = torch.tensor([[1.0]], dtype=atoms.atoms.dtype)
    x_hat = atoms(z)
    assert torch.isfinite(x_hat).all()


def test_lorentz_near_boundary_is_finite() -> None:
    atoms = _atoms(F=2, ball_dim=3, lorentz=True)
    with torch.no_grad():
        a = torch.tensor(
            [[0.995, 0.0, 0.0], [0.0, 0.99, 0.0]], dtype=atoms.atoms.dtype
        )
        atoms.atoms.data = atoms.project_into_ball(a)
    z = torch.randn(3, 2, dtype=atoms.atoms.dtype)
    x_hat = atoms(z)
    assert torch.isfinite(x_hat).all()


def test_rejects_nonnegative_curvature() -> None:
    with pytest.raises(ValueError):
        _atoms(F=2, ball_dim=3, curvature=0.0)
    with pytest.raises(ValueError):
        _atoms(F=2, ball_dim=3, curvature=0.5)
