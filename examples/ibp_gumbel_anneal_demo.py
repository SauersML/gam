#!/usr/bin/env python3
"""IBP-MAP binary-concrete annealing on sparse per-row atom activations."""

from __future__ import annotations

import numpy as np

import gamfit


N, K, P, SEED = 200, 8, 6, 20260524


def make_data() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    x = np.linspace(-1.0, 1.0, N)
    z = rng.random((N, K)) < rng.uniform(0.08, 0.22, size=K)
    z[:, 0] |= x < -0.35
    z[:, 1] |= (x >= -0.35) & (x < 0.20)
    z[:, 2] |= x >= 0.20
    atom_curves = np.column_stack(
        [
            np.sin((k + 1) * np.pi * x) + 0.35 * np.cos((k + 2) * np.pi * x)
            for k in range(K)
        ]
    )
    mixing = rng.normal(size=(K, P))
    signal = (z * atom_curves) @ mixing
    signal += 0.04 * rng.normal(size=signal.shape)
    return signal - signal.mean(axis=0, keepdims=True)


def main() -> None:
    y = make_data()
    schedule = gamfit.GumbelTemperatureSchedule(
        tau_start=1.0,
        tau_end=0.1,
        decay="exponential",
    )
    ibp = gamfit.IBPAssignmentPenalty(
        K,
        alpha=1.0,
        tau=1.0,
        temperature_schedule=schedule,
    )
    descriptor = ibp.to_rust_descriptor()
    assert descriptor["temperature_schedule"]["tau_min"] == 0.1

    fit = gamfit.sae_manifold_fit(
        y,
        n_atoms=K,
        atom_basis=["duchon"] * K,
        atom_dim=[1] * K,
        assignment_prior="ibp_map",
        alpha=ibp.alpha,
        tau=ibp.tau,
        gumbel_schedule=schedule,
        smoothness="auto",
        max_iter=14,
        learning_rate=0.04,
        random_state=SEED,
    )
    assignments = np.asarray(fit.assignments, dtype=float)
    near_binary = float(np.mean((assignments < 0.05) | (assignments > 0.95)))
    mean_active = float(assignments.sum(axis=1).mean())
    if near_binary <= 0.80:
        raise SystemExit(
            f"FAIL near_binary={near_binary:.3f}, mean_active={mean_active:.2f}"
        )
    print(f"PASS near_binary={near_binary:.3f}, mean_active={mean_active:.2f}")


if __name__ == "__main__":
    main()
