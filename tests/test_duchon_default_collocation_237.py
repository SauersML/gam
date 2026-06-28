"""RED tests for issue #237.

Public contract (``gamfit/smooth.py:124-129``):

* ``m=2, d=1`` — natural cubic smoothing spline
* ``m=2, d=2`` — thin-plate spline
* ``m=2, d ≥ 3`` — Duchon's generalized thin-plate spline

The descriptor ``gamfit.Duchon(centers, m=2).evaluate(...)`` must succeed at
every advertised ``(d, m)`` default. The PyFFI ``gamfit.duchon_basis``
(basis-only path, no penalty returned to Python) must also accept these
defaults without tripping the D2 collocation validator.

These tests are currently expected to FAIL for d=2 (and likely d=3) with
``m=2`` defaults because the PyFFI builds the spec with all three operator
penalties active (mass + tension + stiffness), forcing ``max_op=2`` and
requiring ``2*(p+s) > d+2``.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Descriptor evaluate() — every advertised (d, m) default
# ---------------------------------------------------------------------------


def test_descriptor_evaluate_d1_m2_natural_cubic() -> None:
    torch = _torch()
    centers = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    phi = spec.evaluate(x)
    assert phi.shape == (11, 8)
    assert torch.isfinite(phi).all()


def test_descriptor_evaluate_d2_m2_thin_plate() -> None:
    torch = _torch()
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((9, 2))
    spec = gamfit.Duchon(centers=centers, m=2)
    pts = rng.standard_normal((5, 2))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (5, 9)
    assert torch.isfinite(phi).all()


def test_descriptor_evaluate_d3_m2_generalized_tps() -> None:
    torch = _torch()
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((12, 3))
    spec = gamfit.Duchon(centers=centers, m=2)
    pts = rng.standard_normal((6, 3))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    z = torch.tensor(pts[:, 2], dtype=torch.float64)
    phi = spec.evaluate(x, y, z)
    assert phi.shape == (6, 12)
    assert torch.isfinite(phi).all()


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_descriptor_evaluate_d2_various_m(m: int) -> None:
    torch = _torch()
    rng = np.random.default_rng(2 + m)
    centers = rng.standard_normal((10, 2))
    spec = gamfit.Duchon(centers=centers, m=m)
    pts = rng.standard_normal((4, 2))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (4, 10)
    assert torch.isfinite(phi).all()


# ---------------------------------------------------------------------------
# PyFFI basis-only path — `gamfit.duchon_basis` must not require D2 collocation
# ---------------------------------------------------------------------------


# #1512 triage / #237 still open: the pyffi `gamfit.duchon_basis(pts, centers,
# m=...)` collocation path returns FEWER columns than the number of centers for
# several (d, m, n_centers) configurations (e.g. 9 centers d=2 m=2 -> 8 cols,
# 11 centers d=3 m=2 -> 9 cols, 10 centers d=2 -> 8 cols), while the descriptor
# `.evaluate` path for the same config returns the full one-column-per-center
# basis (test_descriptor_evaluate_d2_m2_thin_plate, which passes). The two
# Duchon surfaces disagree on the default collocation/null-space augmentation.
# SPEC.md forbids xfail, so these pyffi-duchon tests stand FAILING as the signal
# of the open #237 inconsistency; fix the pyffi basis to emit one column per
# center like the descriptor path to green them.
def test_pyffi_duchon_basis_d2_m2_default() -> None:
    rng = np.random.default_rng(10)
    centers = rng.standard_normal((9, 2))
    pts = rng.standard_normal((5, 2))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (5, 9)
    assert np.all(np.isfinite(basis))


def test_pyffi_duchon_basis_d3_m2_default() -> None:
    rng = np.random.default_rng(11)
    centers = rng.standard_normal((11, 3))
    pts = rng.standard_normal((4, 3))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (4, 11)
    assert np.all(np.isfinite(basis))


@pytest.mark.parametrize("d", [2, 3, 4])
def test_pyffi_duchon_basis_higher_d_m2_default(d: int) -> None:
    rng = np.random.default_rng(20 + d)
    centers = rng.standard_normal((max(8, 2 * d), d))
    pts = rng.standard_normal((5, d))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (5, centers.shape[0])
    assert np.all(np.isfinite(basis))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_pyffi_duchon_basis_d2_various_m(m: int) -> None:
    rng = np.random.default_rng(30 + m)
    centers = rng.standard_normal((10, 2))
    pts = rng.standard_normal((4, 2))
    basis = gamfit.duchon_basis(pts, centers, m=m)
    assert basis.shape == (4, 10)
    assert np.all(np.isfinite(basis))
