"""TopologyAutoSelector on a synthetic ring."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 180
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = (
        1.4 * np.sin(theta)
        - 0.5 * np.cos(2.0 * theta)
        + 0.08 * rng.standard_normal(n)
    )
    data = {"y": y}

    latent = gamfit.LatentCoord(
        n=n,
        d=1,
        init=((theta + 0.05 * rng.standard_normal(n)) % (2.0 * np.pi)).reshape(-1, 1),
        aux_prior={"u": theta.reshape(-1, 1), "family": "ridge", "strength": "auto"},
    )
    selector = gamfit.TopologyAutoSelector(candidates=["euclidean", "circle"])
    result = selector.fit(
        data,
        "y ~ s(theta, type=AUTO, k=32)",
        latents={"theta": latent},
        family="gaussian",
    )

    print("winner:", result.winner[0])
    for topology_name, tk_score, raw_reml, effective_dim, n_obs, _fit in result.ranked:
        print(
            f"{topology_name:9s} tk={tk_score:.6g} "
            f"raw_reml={raw_reml:.6g} edf={effective_dim:.3f} n={n_obs}"
        )


if __name__ == "__main__":
    main()
