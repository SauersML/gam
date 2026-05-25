"""Fit a periodic B-spline smooth."""

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(31)
    n = 256
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.sin(t) + 0.1 * rng.normal(size=n)

    model = gamfit.fit(
        {"y": y, "t": t},
        "y ~ bspline(t, k=24, periodic=true, period=2*pi, origin=0)",
        family="gaussian",
    )
    summary = model.summary()
    print(f"{summary['family_name']} B-spline fit: deviance={summary['deviance']:.3f}")


if __name__ == "__main__":
    main()
