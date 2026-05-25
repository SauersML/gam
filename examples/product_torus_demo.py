"""Fit a latent smooth with product-torus updates."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(11)
    n = 96
    theta = rng.uniform(-np.pi, np.pi, size=(n, 3))
    y = (
        np.sin(theta[:, 0])
        + 0.5 * np.cos(theta[:, 1] - theta[:, 2])
        + 0.05 * rng.normal(size=n)
    )
    manifold = gamfit.ProductManifold(
        gamfit.CircleManifold(),
        gamfit.CircleManifold(),
        gamfit.CircleManifold(),
    )
    model = gamfit.fit(
        {"y": y},
        "y ~ s(t, type='duchon', centers=24)",
        family="gaussian",
        latents={
            "t": gamfit.LatentCoord(
                n=n,
                d=3,
                init=theta,
                manifold=manifold.to_json(),
                retraction=manifold.to_json(),
                aux_prior={"u": theta},
            )
        },
    )
    print(f"product torus fit deviance={model.summary()['deviance']:.3f}")


if __name__ == "__main__":
    main()
