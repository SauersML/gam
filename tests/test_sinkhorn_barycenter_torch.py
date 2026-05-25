"""Torch differentiability test for the Sinkhorn barycenter."""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
torch = pytest.importorskip("torch")

from gamfit import kernels
from gamfit import kernels_torch


def test_torch_gradcheck_vs_finite_differences() -> None:
    rng = np.random.default_rng(7)
    m = 16
    k = 2
    n_iter = 200
    eps = 0.1
    atoms_np = rng.uniform(0.01, 1.0, size=(k, m))
    atoms_np = atoms_np / atoms_np.sum(axis=1, keepdims=True)
    weights_np = np.array([0.5, 0.5])
    cost_np = kernels.circular_cost(m)

    atoms = torch.tensor(atoms_np, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(weights_np, dtype=torch.float64, requires_grad=False)
    cost = torch.tensor(cost_np, dtype=torch.float64)

    bary = kernels_torch.sinkhorn_barycenter(
        atoms, weights, cost, eps=eps, n_iter=n_iter
    )
    target = torch.tensor(
        rng.normal(size=(m,)), dtype=torch.float64
    )
    loss = (bary * target).sum()
    loss.backward()
    analytic = atoms.grad.detach().cpu().numpy().copy()

    # Finite-difference at a handful of entries.
    h = 1e-5
    target_np = target.detach().cpu().numpy()
    for (ki, j) in [(0, 3), (1, 7), (0, 10)]:
        plus = atoms_np.copy()
        minus = atoms_np.copy()
        plus[ki, j] += h
        minus[ki, j] -= h
        bp = kernels.sinkhorn_barycenter(plus, weights_np, cost_np, eps=eps, n_iter=n_iter)
        bm = kernels.sinkhorn_barycenter(minus, weights_np, cost_np, eps=eps, n_iter=n_iter)
        fd = float(((bp - bm) * target_np).sum() / (2 * h))
        # adjoint iteration agrees with FD to a few percent.
        denom = max(abs(analytic[ki, j]), abs(fd), 1e-6)
        rel = abs(analytic[ki, j] - fd) / denom
        assert rel < 0.05, f"mismatch at ({ki},{j}): analytic={analytic[ki,j]} fd={fd} rel={rel}"
