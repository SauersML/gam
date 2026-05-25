"""Fit a standard GAM with a learned latent coordinate."""

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 96
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rgb = np.column_stack(
        [
            0.5 + 0.45 * np.cos(theta),
            0.5 + 0.45 * np.sin(theta),
            rng.uniform(0.0, 1.0, n),
        ]
    )
    y = np.sin(theta) + 0.15 * rng.normal(size=n)
    data = pd.DataFrame({"y": y})
    data.rgb_array = rgb

    model = gamfit.fit(
        data,
        "y ~ s(t, type='duchon', centers=32)",
        latents={
            "t": gamfit.LatentCoord(
                n=len(data),
                d=2,
                init="pca",
                aux_prior={"u": data.rgb_array},
            )
        },
    )
    summary = model.summary()
    print(f"family={summary['family_name']} reml={summary['reml_score']:.6g}")


if __name__ == "__main__":
    main()
