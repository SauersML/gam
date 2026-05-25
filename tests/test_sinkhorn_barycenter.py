"""Python integration tests for gamfit.kernels.sinkhorn_barycenter."""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

from gamfit import kernels


def test_circular_cost_shape_and_symmetry() -> None:
    m = 7
    cc = kernels.circular_cost(m)
    assert cc.shape == (m, m)
    assert np.all(cc >= 0.0)
    assert np.allclose(cc, cc.T)
    assert np.all(np.diag(cc) == 0.0)
    # Distance between index 0 and m // 2 on a length-m cycle is floor(m/2).
    assert cc[0, m // 2] == (m // 2) ** 2


def test_euclidean_cost_shape_and_symmetry() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, 3))
    ec = kernels.euclidean_cost(points)
    assert ec.shape == (5, 5)
    assert np.allclose(ec, ec.T)
    assert np.all(np.diag(ec) == 0.0)
    # Validate against numpy reference.
    diff = points[:, None, :] - points[None, :, :]
    ref = (diff ** 2).sum(-1)
    assert np.allclose(ec, ref)


def test_geodesic_sphere_cost_shape_and_symmetry() -> None:
    dirs = np.eye(3, dtype=np.float64)
    gc = kernels.geodesic_sphere_cost(dirs)
    assert gc.shape == (3, 3)
    assert np.allclose(gc, gc.T)
    assert np.allclose(np.diag(gc), 0.0)
    # Orthogonal axes are at geodesic distance pi / 2.
    assert np.allclose(gc[0, 1], (np.pi / 2) ** 2)


def test_k_eq_1_recovers_atom() -> None:
    atom = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.04, 0.01])
    atoms = atom[None, :]
    bary = kernels.sinkhorn_barycenter(atoms, eps=0.05, n_iter=60)
    assert bary.shape == (8,)
    assert abs(bary.sum() - 1.0) < 1e-8
    assert np.max(np.abs(bary - atom)) < 5e-3


def test_k_eq_2_mean_is_between() -> None:
    m = 32
    pts = np.linspace(0.0, 1.0, m)[:, None]
    a = np.exp(-((pts.ravel() - 0.2) ** 2) / 0.005)
    b = np.exp(-((pts.ravel() - 0.8) ** 2) / 0.005)
    a /= a.sum()
    b /= b.sum()
    atoms = np.stack([a, b], axis=0)
    cost = kernels.euclidean_cost(pts)
    bary = kernels.sinkhorn_barycenter(
        atoms, weights=np.array([0.5, 0.5]), cost=cost, eps=0.005, n_iter=200
    )
    assert bary.shape == (m,)
    assert abs(bary.sum() - 1.0) < 1e-8
    mean_a = float((pts.ravel() * a).sum())
    mean_b = float((pts.ravel() * b).sum())
    mean_bary = float((pts.ravel() * bary).sum())
    assert mean_a < mean_bary < mean_b
    assert abs(mean_bary - 0.5 * (mean_a + mean_b)) < 0.05


def test_no_nan_at_small_eps() -> None:
    m = 16
    atoms = np.stack(
        [np.exp(-((np.arange(m) - 3.0) ** 2) / 4.0),
         np.exp(-((np.arange(m) - 11.0) ** 2) / 4.0)],
        axis=0,
    )
    bary = kernels.sinkhorn_barycenter(atoms, eps=1e-3, n_iter=50)
    assert np.all(np.isfinite(bary))
    assert abs(bary.sum() - 1.0) < 1e-8


def test_rejects_tiny_eps() -> None:
    atoms = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        kernels.sinkhorn_barycenter(atoms, eps=1e-15, n_iter=10)
