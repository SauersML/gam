#!/usr/bin/env python3
"""Fit a latent GAM with orthogonality and ARD penalties."""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    rng = np.random.default_rng(214)
    n = 96
    d = 6
    t0 = rng.normal(size=(n, d))
    y = np.sin(t0[:, 0]) + 0.35 * t0[:, 1] + 0.12 * rng.normal(size=n)
    data = pd.DataFrame(data={"y": y})

    gamfit.fit(
        data=data,
        formula="y ~ s(t, type='duchon', centers=32)",
        latents={"t": gamfit.LatentCoord(n=n, d=d, init=t0)},
        penalties=[
            gamfit.OrthogonalityPenalty(weight=1.0, n_eff=n, target="t"),
            gamfit.ARDPenalty(target="t"),
        ],
    )
    print(f"fit orthogonality+ARD latent GAM: n={n}, d={d}")


if __name__ == "__main__":
    main()
