"""Fit one SAE manifold atom to a synthetic curved feature."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 160
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    z = np.c_[
        np.sin(theta),
        np.cos(theta),
        0.35 * np.sin(2.0 * theta),
        0.35 * np.cos(2.0 * theta),
    ]
    z += 0.03 * rng.standard_normal(z.shape)

    fit = gamfit.sae_manifold_fit(
        z,
        K=1,
        d_atom=2,
        atom_topology="circle",
        assignment="ordered_beta_bernoulli",
        n_iter=8,
        learning_rate=0.04,
        random_state=7,
    )
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
