"""Fit a latent GAM with NuclearNormPenalty and ARDPenalty."""

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(515)
    n = 96
    t = rng.normal(size=(n, 4))
    t[:, 2:] *= 0.05
    y = np.sin(t[:, 0]) + 0.4 * t[:, 1] + 0.1 * rng.normal(size=n)

    model = gamfit.fit(
        {"y": y},
        "y ~ s(t, type='duchon', centers=24)",
        family="gaussian",
        latents={"t": gamfit.LatentCoord(n=n, d=4, init=t)},
        penalties=[
            gamfit.NuclearNormPenalty(1.0, n_eff=n, target="t"),
            gamfit.ARDPenalty(target="t"),
        ],
    )

    print(f"nuclear_norm_ard_demo: {type(model).__name__}")


if __name__ == "__main__":
    main()
