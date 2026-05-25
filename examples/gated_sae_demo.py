"""Gated SAE decoder demo."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(11)
    n = 120
    t = np.linspace(0.0, 1.0, n)
    x = np.c_[
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
        (t > 0.5).astype(float) * np.sin(6.0 * np.pi * t),
    ] + 0.03 * rng.standard_normal((n, 3))
    fit = gamfit.sae_manifold_fit(
        x,
        K=3,
        d_atom=2,
        atom_topology="circle",
        assignment="gated",
        n_iter=8,
        random_state=5,
    )
    print(f"assignment=gated K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
