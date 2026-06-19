"""Fast iteration harness for issue #1282 torch ManifoldSAE two-circle routing.

This is a *development* harness, NOT a test (it lives under ``scripts/`` so
pytest does not collect it). It exercises the exact same DGP and routing code
path as ``tests/torch/test_manifold_sae_two_circle_routing_issue_1282.py`` —
the energy-degenerate sign-coupled circles ``(c, s, c, ±s)`` plus the
``softmax_topk`` top-1 reconstruction-residual / transferable-anchor router —
but at reduced ``n`` and fewer training steps so a routing-accuracy read takes
seconds instead of minutes. Use it to iterate on the routing logic.

The full, un-weakened regression test (real ``n=256/128``, accuracy bars
``0.90``/``0.85``, all seeds) remains the only green gate. The micro harness
trains fewer steps on smaller batches and therefore reports *optimistic*
routing numbers; passing the micro is necessary, never sufficient. Always run
the full test once before declaring #1282 green.

Run::

    python scripts/micro_two_circle_routing_1282.py [n_seeds]   # default 3
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import gamfit.torch as gt


def _energy_degenerate_sign_coupled_circles(
    n: int, *, seed: int, noise: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """The #1282 fixture: two circles with identical raw squared-energy features.

    Label 0 is the diagonal circle ``(c, s, c, s)`` and label 1 flips only the
    last signed channel ``(c, s, c, -s)``. Identical row-level squared-coordinate
    profile and shared ambient line directions, so only a router keyed on a
    manifold-correlated discriminant (the ``x_i * x_j`` union-of-subspaces split)
    can separate them.
    """
    rng = np.random.default_rng(seed)
    truth = np.arange(n, dtype=int) % 2
    rng.shuffle(truth)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    c = np.cos(theta)
    s = np.sin(theta)
    x = np.empty((n, 4), dtype=np.float64)
    x[:, 0] = c
    x[:, 1] = s
    x[:, 2] = c
    x[:, 3] = np.where(truth == 0, s, -s)
    x /= np.sqrt(2.0)
    x += noise * rng.standard_normal(x.shape)
    return x, truth


def _best_label_flip_accuracy(assignments: np.ndarray, truth: np.ndarray) -> float:
    pred = np.argmax(assignments, axis=1)
    return max(float(np.mean(pred == truth)), float(np.mean((1 - pred) == truth)))


def run_once(
    manual_seed: int,
    *,
    n_train: int = 128,
    n_test: int = 64,
    steps: int = 200,
) -> tuple[float, float, object]:
    """Train the micro SAE and return (train_acc, test_acc, anchor_rule)."""
    torch.manual_seed(manual_seed)
    train, train_truth = _energy_degenerate_sign_coupled_circles(n_train, seed=12820)
    test, test_truth = _energy_degenerate_sign_coupled_circles(n_test, seed=12821)
    cfg = gt.ManifoldSAEConfig(
        input_dim=4,
        n_atoms=2,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=9,
        sparsity={
            "kind": "softmax_topk",
            "target_k": 1,
            "tau_start": 1.0,
            "tau_min": 0.05,
            "tau_steps": steps,
        },
    )
    model = gt.ManifoldSAE(cfg).double()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    x = torch.as_tensor(train, dtype=torch.float64)
    for _ in range(steps):
        out = model(x)
        loss = ((out.x_hat - x) ** 2).mean() + 1.0e-5 * model.regularization(out.gate)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.sparsity.advance_temperature()
    rule = model.sparsity._global_anchor_rule
    with torch.no_grad():
        train_out = model(x)
        test_out = model(torch.as_tensor(test, dtype=torch.float64))
    train_acc = _best_label_flip_accuracy(
        train_out.assignments.detach().numpy(), train_truth
    )
    test_acc = _best_label_flip_accuracy(
        test_out.assignments.detach().numpy(), test_truth
    )
    return train_acc, test_acc, rule


def main() -> int:
    torch.set_num_threads(8)
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seeds = [12820, 2024, 7, 0, 99, 1][:n_seeds]
    t0 = time.time()
    all_pass = True
    for ms in seeds:
        train_acc, test_acc, rule = run_once(ms)
        ok = train_acc >= 0.90 and test_acc >= 0.85
        all_pass = all_pass and ok
        print(
            f"mseed={ms}: train={train_acc:.4f} test={test_acc:.4f} "
            f"anchor_rule={rule} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    print(f"ELAPSED={time.time() - t0:.1f}s  ALL_PASS={all_pass}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
