"""Regression for issue #1282: top-1 torch ManifoldSAE routes two circles.

The failure mode reconstructed well but routed at chance: both atoms learned a
mixed union of two disjoint circles in R^4. This test scores the actual atom
winner against the planted active manifold label, up to atom-label permutation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _two_disjoint_noisy_circles(
    n: int, *, seed: int, noise: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = np.arange(n, dtype=int) % 2
    rng.shuffle(truth)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x = np.zeros((n, 4), dtype=np.float64)
    x[truth == 0, 0:2] = circle[truth == 0]
    x[truth == 1, 2:4] = circle[truth == 1]
    x += noise * rng.standard_normal(x.shape)
    return x, truth


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    return 1.0 - float(np.mean((fitted - x) ** 2) / np.var(x))


def _best_label_flip_accuracy(assignments: np.ndarray, truth: np.ndarray) -> float:
    pred = np.argmax(assignments, axis=1)
    return max(float(np.mean(pred == truth)), float(np.mean((1 - pred) == truth)))


def test_softmax_top1_routes_disjoint_noisy_circles() -> None:
    torch.manual_seed(1282)
    train, train_truth = _two_disjoint_noisy_circles(192, seed=1282)
    test, test_truth = _two_disjoint_noisy_circles(96, seed=1283)

    cfg = gt.ManifoldSAEConfig(
        input_dim=4,
        n_atoms=2,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=7,
        sparsity={
            "kind": "softmax_topk",
            "target_k": 1,
            "tau_start": 1.0,
            "tau_min": 0.05,
            "tau_steps": 600,
        },
    )
    model = gt.ManifoldSAE(cfg).double()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.as_tensor(train, dtype=torch.float64)

    for _ in range(600):
        out = model(x)
        loss = ((out.x_hat - x) ** 2).mean() + 1.0e-5 * model.regularization(out.gate)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.sparsity.advance_temperature()

    with torch.no_grad():
        train_out = model(x)
        test_out = model(torch.as_tensor(test, dtype=torch.float64))

    train_fit = train_out.x_hat.detach().numpy()
    test_fit = test_out.x_hat.detach().numpy()
    train_assign = train_out.assignments.detach().numpy()
    test_assign = test_out.assignments.detach().numpy()

    assert _r2(train, train_fit) >= 0.90
    assert _r2(test, test_fit) >= 0.85
    assert _best_label_flip_accuracy(train_assign, train_truth) >= 0.95
    assert _best_label_flip_accuracy(test_assign, test_truth) >= 0.90
    assert float(np.mean(np.count_nonzero(train_assign > 1.0e-8, axis=1))) == 1.0
