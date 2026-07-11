"""Compare SCAD and MCP non-convex gate sparsity in manifold SAE fitting.

The synthetic data contain three local mechanisms but the model is given five
candidate atoms. ``coord_sparsity="scad"`` and ``"mcp"`` route the SAE row-block
ScadMcp penalty through ``sae_manifold_fit`` and should concentrate assignment
mass on the useful atoms while leaving redundant atoms mostly inactive.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def _active_fraction(assignments: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.mean(assignments >= threshold, axis=0)


def main() -> None:
    rng = np.random.default_rng(240)
    n = 240
    t = np.linspace(-1.0, 1.0, n)
    gates = np.c_[t < -0.25, (t >= -0.25) & (t <= 0.35), t > 0.35].astype(float)
    atoms = np.stack(
        [
            np.c_[np.sin(2.0 * np.pi * t), np.zeros(n), 0.25 * t, np.zeros(n)],
            np.c_[np.zeros(n), t**2, np.cos(np.pi * t), np.zeros(n)],
            np.c_[0.25 * t, np.zeros(n), np.zeros(n), np.sin(3.0 * np.pi * t)],
        ],
        axis=1,
    )
    x = np.sum(gates[:, :, None] * atoms, axis=1)
    x += 0.03 * rng.standard_normal(x.shape)

    fits = {
        "scad": gamfit.sae_manifold_fit(
            x,
            K=5,
            d_atom=1,
            atom_topology="euclidean",
            assignment="ordered_beta_bernoulli",
            coord_sparsity="scad",
            scad_mcp_gamma=3.7,
            sparsity_weight=1.2,
            n_iter=20,
            random_state=240,
        ),
        "mcp": gamfit.sae_manifold_fit(
            x,
            K=5,
            d_atom=1,
            atom_topology="euclidean",
            assignment="ordered_beta_bernoulli",
            coord_sparsity="mcp",
            scad_mcp_gamma=2.5,
            sparsity_weight=1.2,
            n_iter=20,
            random_state=240,
        ),
    }

    print("SCAD/MCP gate sparsity demo")
    for label, fit in fits.items():
        summary = fit.summary()
        print(
            f"{label}: r2={fit.reconstruction_r2:.3f} "
            f"avg_active_atoms={summary['avg_active_atoms']:.2f} "
            f"active_fraction={np.round(_active_fraction(fit.assignments), 3)}"
        )
        print(f"{label} primitives:", summary["primitives"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for ax, (label, fit) in zip(axes, fits.items()):
        sorted_mass = np.sort(fit.assignments, axis=0)
        ax.plot(sorted_mass)
        ax.set_title(f"{label.upper()} sorted assignment mass")
        ax.set_xlabel("observation rank")
        ax.set_ylabel("assignment")
        ax.set_ylim(-0.05, 1.05)
    plt.show()


if __name__ == "__main__":
    main()
