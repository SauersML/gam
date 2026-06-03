"""Read per-atom posterior shape uncertainty and typical coordinate ranges.

Fresh manifold-SAE fits expose per-atom decoder covariance and a posterior
shape band through ``shape_uncertainty``. This example prints the coordinate
range and typical-shape summaries, then plots the fitted mean curve with a
pointwise mean +/- one posterior standard deviation band.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def main() -> None:
    rng = np.random.default_rng(674)
    n = 180
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.c_[
        np.cos(theta),
        np.sin(theta),
        0.25 * np.cos(2.0 * theta),
    ]
    x += 0.035 * rng.standard_normal(x.shape)

    fit = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp",
        n_iter=18,
        random_state=674,
    )

    atom = 0
    coordinate_range = fit.coordinate_range(atom=atom)
    band = fit.shape_uncertainty(atom=atom, n_sd=1.0)
    typical = fit.typical_shape(atom=atom, quantile_range=(5.0, 95.0), n_sd=1.0)
    covariance = fit.atoms[atom].decoder_covariance

    print("Per-atom uncertainty and typical range")
    print(f"r2={fit.reconstruction_r2:.3f}")
    print(f"decoder_covariance_shape={covariance.shape}")
    print(
        "coordinate p05/p50/p95:",
        np.round(coordinate_range["p05"], 3),
        np.round(coordinate_range["p50"], 3),
        np.round(coordinate_range["p95"], 3),
    )
    print(f"shape band grid={band['coords'].shape} ambient={band['mean'].shape}")
    print("typical ambient mean:", np.round(typical["ambient_mean"], 3))
    print("typical posterior sd mean:", np.round(typical["posterior_sd_mean"], 4))

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
