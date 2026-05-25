"""Fit a small Matérn smooth with automatic centers."""

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(128, 4))
    y = np.sin(x[:, 0]) + 0.4 * x[:, 1] + 0.1 * rng.normal(size=x.shape[0])
    data = {"y": y} | {f"x{i}": x[:, i] for i in range(x.shape[1])}

    model = gamfit.fit(
        data,
        "y ~ matern(x0, x1, x2, x3, centers=24, nu=2.5, length_scale=1.0)",
        family="gaussian",
    )
    print(f"matern fit: deviance={model.summary()['deviance']:.3f}")


if __name__ == "__main__":
    main()
