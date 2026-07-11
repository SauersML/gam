"""Smooth threshold-gated manifold-SAE demo."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(-1.0, 1.0, 100)
    x = np.c_[t, np.maximum(t - 0.15, 0.0), np.maximum(-t - 0.2, 0.0)]
    x += 0.03 * rng.standard_normal(x.shape)
    fit = gamfit.sae_manifold_fit(
        x,
        K=3,
        d_atom=2,
        atom_topology="euclidean",
        assignment="threshold_gate",
        threshold_gate_threshold=0.0,
        n_iter=8,
        random_state=9,
    )
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
