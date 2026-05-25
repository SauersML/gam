"""Fit a latent Duchon composition with analytic penalties."""

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    rng = np.random.default_rng(23)
    n, d = 96, 2
    t = rng.normal(size=(n, d))
    y = np.sin(t[:, 0]) + 0.4 * t[:, 1] + 0.1 * rng.normal(size=n)
    data = pd.DataFrame({"y": y})
    lambda_per_row = np.broadcast_to(np.eye(d), (n, d, d))
    penalties = [
        gamfit.AuxConditionalPriorPenalty(lambda_per_row, weight=1.0, n_eff=n),
        gamfit.ScadMcpPenalty(weight=0.1, n_eff=n, variant="mcp"),
        gamfit.OrthogonalityPenalty(weight=0.1, n_eff=n),
    ]

    model = gamfit.fit(
        data,
        "y ~ s(t, type='duchon', centers=24)",
        family="gaussian",
        latents={"t": gamfit.LatentCoord(n=n, d=d, init=t)},
        penalties=penalties,
    )
    print(f"composition fit: {type(model).__name__} with {len(penalties)} penalties")


if __name__ == "__main__":
    main()
