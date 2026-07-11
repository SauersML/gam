"""Fit an ordered independent Beta--Bernoulli SAE with temperature annealing."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(20260524)
    n = 120
    t = np.linspace(-1.0, 1.0, n)
    y = np.c_[
        np.sin(np.pi * t),
        (t > -0.2) * np.cos(2.0 * np.pi * t),
        (t > 0.35) * np.sin(3.0 * np.pi * t),
    ]
    y += 0.04 * rng.standard_normal(y.shape)

    schedule = gamfit.GumbelTemperatureSchedule(
        tau_start=1.0,
        tau_min=0.1,
        decay="geometric",
    )
    fit = gamfit.sae_manifold_fit(
        y,
        K=4,
        d_atom=1,
        atom_topology="euclidean",
        assignment="ordered_beta_bernoulli",
        schedule=schedule,
        n_iter=8,
        learning_rate=0.04,
        random_state=20260524,
    )
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
