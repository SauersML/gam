"""TopK SAE-manifold demo."""

from __future__ import annotations

import numpy as np

import gamfit


def make_data(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    atom_a = np.c_[np.cos(t), np.sin(t), 0.25 * np.cos(2.0 * t)]
    atom_b = np.c_[0.2 * np.sin(3.0 * t), np.cos(0.5 * t), np.sin(0.5 * t)]
    mask = (np.arange(n) % 3 == 0)[:, None]
    x = np.where(mask, atom_a, atom_b)
    return x + 0.04 * rng.standard_normal(x.shape)


def main() -> None:
    x = make_data()
    fit = gamfit.sae_manifold_fit(
        x,
        K=4,
        d_atom=2,
        atom_topology="circle",
        assignment="topk",
        top_k=2,
        n_iter=8,
        random_state=2,
    )
    print("assignment=topk")
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")
    print("active mass:", np.round(fit.assignments.sum(axis=0), 2))


if __name__ == "__main__":
    main()
