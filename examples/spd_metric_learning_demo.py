"""Fit a response-geometry GAM with an SPD Fisher-Rao metric."""

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    n = 64
    x = np.linspace(0.0, 1.0, n)
    angle = 2.0 * np.pi * x
    data = pd.DataFrame(
        {
            "x": x,
            "sx": 0.6 * np.cos(angle),
            "sy": 0.6 * np.sin(angle),
            "sz": np.full(n, 0.8),
        }
    )
    model = gamfit.fit(
        data,
        "sphere ~ s(x, type='duchon', centers=16)",
        response_geometry="spherical",
        response_columns=["sx", "sy", "sz"],
        fisher_rao_w=np.diag([1.0, 1.5, 2.0]),
    )
    print("tangent_dimension", model.summary()["tangent_dimension"])


if __name__ == "__main__":
    main()
