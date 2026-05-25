"""Fit a latent GAM with a mechanism sparsity penalty."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(802)
    n = 96
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    aux = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
    y = np.sin(theta) + 0.1 * rng.normal(size=n)

    penalty = gamfit.MechanismSparsityPenalty(
        [[0, 1], [2, 3]],
        weight=0.2,
        n_eff=n,
        target="t",
    )
    model = gamfit.fit(
        {"y": y},
        "y ~ s(t, type='duchon', centers=24)",
        latents={
            "t": gamfit.LatentCoord(n=n, d=2, init=aux[:, :2], aux_prior={"u": aux}),
        },
        penalties=[penalty],
    )
    print(f"fit={model.summary()['family_name']} penalty={penalty.to_rust_descriptor()['kind']}")


if __name__ == "__main__":
    main()
