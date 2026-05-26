"""RED tests for issue #239: ``PoincareAtoms.distance(a, a)`` must be 0.

The Rust kernel `poincare_distance` clamps `acosh` arg to `1.0 + ORIGIN_EPS`
(`src/geometry/poincare.rs:172`), so self-distance returns ~4.47e-8 instead
of the mathematically required 0.0. These tests pin the principled contract
and will fail until the clamp is fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def _atoms(F: int = 3, ball_dim: int = 4):
    return gamfit.PoincareAtoms(F=F, ball_dim=ball_dim)


def test_self_distance_is_exact_zero_single_row() -> None:
    atoms = _atoms()
    a = torch.tensor([[0.1, -0.2, 0.05, 0.0]], dtype=torch.float64)
    d = atoms.distance(a, a)
    np.testing.assert_array_equal(
        d.detach().cpu().numpy(),
        np.zeros_like(d.detach().cpu().numpy()),
    )


def test_self_distance_is_exact_zero_batch() -> None:
    atoms = _atoms()
    a = torch.tensor(
        [[0.1, -0.2, 0.05, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [-0.3, 0.2, 0.1, -0.05]],
        dtype=torch.float64,
    )
    d = atoms.distance(a, a)
    np.testing.assert_array_equal(
        d.detach().cpu().numpy(),
        np.zeros(d.shape, dtype=np.float64),
    )


def test_self_distance_at_origin_is_exact_zero() -> None:
    atoms = _atoms()
    a = torch.zeros(1, 4, dtype=torch.float64)
    d = atoms.distance(a, a)
    np.testing.assert_array_equal(
        d.detach().cpu().numpy(),
        np.zeros(d.shape, dtype=np.float64),
    )


def test_distance_is_symmetric() -> None:
    atoms = _atoms()
    a = torch.tensor([[0.3, 0.1, 0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[-0.2, 0.4, 0.05, -0.1]], dtype=torch.float64)
    d_ab = atoms.distance(a, b).detach().cpu().numpy()
    d_ba = atoms.distance(b, a).detach().cpu().numpy()
    np.testing.assert_allclose(d_ab, d_ba, atol=1e-14, rtol=0.0)


def test_near_identical_points_yield_subepsilon_distance() -> None:
    """For a, b separated by 1e-10, d(a, b) ≈ 2e-10 — clamp noise floor
    (~4.5e-8) is 5 orders of magnitude larger. This catches the clamp
    polluting tiny but nonzero separations, not just exact equality.
    """
    atoms = _atoms()
    a = torch.tensor([[0.1, -0.2, 0.05, 0.0]], dtype=torch.float64)
    b = a + 1.0e-10
    d = atoms.distance(a, b).detach().cpu().numpy()
    assert float(d.max()) < 1.0e-7, (
        f"clamp noise floor leaked into tiny-separation distance: {d}"
    )
