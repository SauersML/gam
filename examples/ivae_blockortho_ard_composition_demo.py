#!/usr/bin/env python3
"""Fit a latent GAM with iVAE, block-orthogonality, and ARD penalties."""

import gamfit
import numpy as np


def main() -> None:
    rng = np.random.default_rng(38038)
    n = 96
    d = 6
    u = rng.normal(size=(n, 3))
    z = rng.normal(size=(n, 3)) * np.array([1.0, 0.7, 0.05])
    t0 = np.column_stack([u, z])
    y = np.sin(t0[:, 0]) + 0.4 * t0[:, 3] + 0.1 * rng.normal(size=n)

    gamfit.fit(
        {"y": y.tolist()},
        "y ~ s(t, type='duchon', centers=24)",
        latents={"t": gamfit.LatentCoord(n=n, d=d, init=t0)},
        penalties=[
            gamfit.IvaeRidgeMeanGauge(u, weight=4.0, n_eff=n, target="t"),
            gamfit.BlockOrthogonalityPenalty(
                [[0, 1, 2], [3, 4, 5]], weight=12.0, n_eff=n, target="t"
            ),
            gamfit.ARDPenalty(target="t"),
        ],
    )
    print("fit latent GAM with iVAE, block orthogonality, and ARD penalties")


if __name__ == "__main__":
    main()
