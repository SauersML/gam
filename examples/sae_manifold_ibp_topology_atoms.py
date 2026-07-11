"""Fit IBP-gated manifold SAE atoms with planted circle and torus topology.

This example builds a synthetic ambient representation from two topology-typed
atoms: a 1D circle atom and a 2D torus atom. The planted binary gates make some
observations use only one atom and some use both; the fit uses the finite-IBP
assignment path and heterogeneous per-atom bases.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def main() -> None:
    rng = np.random.default_rng(20260602)
    n = 240

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)

    planted_gates = np.zeros((n, 2), dtype=float)
    planted_gates[:80, 0] = 1.0
    planted_gates[80:160, 1] = 1.0
    planted_gates[160:, :] = 1.0
    rng.shuffle(planted_gates, axis=0)

    circle_atom = np.c_[
        np.cos(theta),
        np.sin(theta),
        0.35 * np.cos(2.0 * theta),
        0.35 * np.sin(2.0 * theta),
        np.zeros(n),
        np.zeros(n),
    ]
    torus_atom = np.c_[
        np.zeros(n),
        np.zeros(n),
        0.45 * np.cos(u),
        0.45 * np.sin(u),
        0.65 * np.cos(v),
        0.65 * np.sin(v),
    ]
    x = planted_gates[:, [0]] * circle_atom + planted_gates[:, [1]] * torus_atom
    x += 0.025 * rng.standard_normal(x.shape)

    fit = gamfit.sae_manifold_fit(
        x,
        K=2,
        atom_basis=["periodic", "torus"],
        d_atom=[1, 2],
        assignment="ordered_beta_bernoulli",
        schedule=gamfit.GumbelTemperatureSchedule(
            tau_start=1.0,
            tau_min=0.2,
            decay="geometric",
        ),
        a_init=_logit(0.1 + 0.8 * planted_gates),
        n_iter=20,
        random_state=20260602,
    )

    summary = fit.summary()
    print("IBP mixed-topology manifold SAE")
    print(f"basis_specs={fit.basis_specs}")
    print(f"atom_topologies={fit.atom_topologies}")
    print(f"assignment={fit.assignment} r2={fit.reconstruction_r2:.3f}")
    print(
        "mean planted gates vs recovered assignments:",
        np.round(planted_gates.mean(axis=0), 3),
        np.round(fit.assignments.mean(axis=0), 3),
    )
    print(f"avg_active_atoms={summary['avg_active_atoms']:.2f}")

    order = np.lexsort((planted_gates[:, 1], planted_gates[:, 0]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].imshow(planted_gates[order], aspect="auto", vmin=0.0, vmax=1.0)
    axes[0].set_title("planted gates")
    axes[0].set_xlabel("atom")
    axes[0].set_ylabel("observation")
    axes[1].imshow(fit.assignments[order], aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_title("IBP assignments")
    axes[1].set_xlabel("atom")
    axes[2].scatter(x[:, 0], x[:, 1], s=12, alpha=0.55, label="data")
    axes[2].scatter(fit.fitted[:, 0], fit.fitted[:, 1], s=8, alpha=0.55, label="fit")
    axes[2].set_title("circle channels")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
