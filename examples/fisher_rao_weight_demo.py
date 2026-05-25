"""Fit a Gaussian GAM with Fisher-Rao observation weights."""

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(4)
    n = 96
    x = np.linspace(0.0, 1.0, n)
    fisher_rao_w = 1.0 + 3.0 * (np.abs(x - 0.55) < 0.18)
    y = np.sin(2.0 * np.pi * x) + rng.normal(scale=0.12 / np.sqrt(fisher_rao_w), size=n)
    data = {"x": x, "y": y}

    model = gamfit.fit(
        data,
        "y ~ s(x, k=18)",
        family="gaussian",
        fisher_rao_w=fisher_rao_w,
    )

    summary = model.summary()
    print(f"Fisher-Rao weighted fit: deviance={summary['deviance']:.4f}")


if __name__ == "__main__":
    main()
