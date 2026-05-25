"""TopK SAE-manifold demo."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(4)
    n = 120
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    atom_a = np.c_[np.cos(t), np.sin(t), 0.25 * np.cos(2.0 * t)]
    atom_b = np.c_[0.2 * np.sin(3.0 * t), np.cos(0.5 * t), np.sin(0.5 * t)]
    x = np.where((np.arange(n) % 3 == 0)[:, None], atom_a, atom_b)
    x += 0.04 * rng.standard_normal(x.shape)
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
    print(f"topk SAE: K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
