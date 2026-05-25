"""Fit a latent GAM with an auxiliary-conditional prior penalty."""

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    rng = np.random.default_rng(902)
    n = 96
    d = 2
    aux = rng.normal(size=(n, 2))
    scales = np.column_stack(
        [
            0.35 + 0.45 / (1.0 + np.exp(-aux[:, 0])),
            0.30 + 0.40 / (1.0 + np.exp(aux[:, 1])),
        ]
    )
    t_init = rng.normal(size=(n, d)) * scales
    y = np.sin(t_init[:, 0]) + 0.25 * t_init[:, 1] + 0.10 * rng.normal(size=n)

    lambda_per_row = np.zeros((n, d, d))
    lambda_per_row[:, np.arange(d), np.arange(d)] = 1.0 / scales**2
    data = pd.DataFrame({"y": y})

    model = gamfit.fit(
        data,
        "y ~ s(t, type='duchon', centers=32)",
        family="gaussian",
        latents={"t": gamfit.LatentCoord(n=n, d=d, init=t_init)},
        penalties=[
            gamfit.AuxConditionalPriorPenalty(
                lambda_per_row,
                weight=2.0,
                n_eff=n,
                target="t",
            )
        ],
    )
    summary = model.summary()
    print(f"family={summary['family_name']} reml={summary['reml_score']:.6g}")


if __name__ == "__main__":
    main()
