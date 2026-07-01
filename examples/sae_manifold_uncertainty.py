"""Read per-atom posterior shape uncertainty.

Fresh manifold-SAE fits expose per-atom decoder covariance and a posterior
shape band through ``shape_uncertainty``. This example prints the shape-band
summary, then plots the fitted mean curve with a pointwise mean +/- one
posterior standard deviation band.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def main() -> None:
    rng = np.random.default_rng(0)
    n = 400
    theta = rng.uniform(0.0, 1.0, n)
    x = np.column_stack([
        np.cos(2.0 * np.pi * theta),
        np.sin(2.0 * np.pi * theta),
    ])
    x += 0.18 * rng.standard_normal(x.shape)

    fit = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        n_iter=40,
        learning_rate=1.0,
        random_state=0,
    )

    atom = 0
    band = fit.shape_uncertainty(atom=atom, n_sd=1.0)
    covariance = fit.atoms[atom].decoder_covariance

    print("Per-atom posterior shape uncertainty")
    print(f"r2={fit.reconstruction_r2:.3f}")
    print(f"decoder_covariance_shape={covariance.shape}")
    print(f"shape band grid={band['coords'].shape} ambient={band['mean'].shape}")
    print("ambient posterior sd mean:", np.round(band["sd"].mean(axis=0), 4))

    order = np.argsort(band["coords"][:, 0])
    coord = band["coords"][order, 0]
    mean = band["mean"][order]
    lower = band["lower"][order]
    upper = band["upper"][order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    axes[0].scatter(x[:, 0], x[:, 1], s=14, alpha=0.4, label="data")
    axes[0].plot(mean[:, 0], mean[:, 1], color="black", lw=1.5, label="posterior mean")
    axes[0].plot(lower[:, 0], lower[:, 1], color="tab:blue", lw=1.0, alpha=0.7, label="-1 sd")
    axes[0].plot(upper[:, 0], upper[:, 1], color="tab:red", lw=1.0, alpha=0.7, label="+1 sd")
    axes[0].set_title("ambient shape band")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best")
    axes[1].plot(coord, mean[:, 0], color="black", label="mean channel 0")
    axes[1].fill_between(coord, lower[:, 0], upper[:, 0], color="tab:blue", alpha=0.2)
    axes[1].set_title("coordinate band")
    axes[1].set_xlabel("atom coordinate")
    axes[1].set_ylabel("ambient channel 0")
    axes[1].legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
