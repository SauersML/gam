#!/usr/bin/env python3
"""Fit a GAM with a circle-constrained latent coordinate."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    data = {"y": (np.sin(theta) + 0.1 * rng.normal(size=n)).tolist()}

    model = gamfit.fit(
        data,
        f"y ~ s(t, type='periodic', k=16, period={2.0 * np.pi}, origin=0)",
        family="gaussian",
        latents={
            "t": gamfit.LatentCoord(
                n=n,
                d=1,
                init=theta[:, None],
                aux_prior={"u": theta[:, None], "family": "ridge", "strength": 1.0},
                manifold="circle",
                retraction="circle",
            )
        },
    )
    summary = model.summary()
    print(f"family={summary['family_name']} reml={summary['reml_score']:.6g}")


if __name__ == "__main__":
    main()
