"""Fit a spherical smooth to synthetic latitude/longitude data."""

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(23)
    n = 96
    lat = rng.uniform(-70.0, 70.0, size=n)
    lon = rng.uniform(-180.0, 180.0, size=n)
    y = (
        np.sin(np.deg2rad(lat))
        + 0.5 * np.cos(np.deg2rad(lon))
        + rng.normal(scale=0.05, size=n)
    )

    model = gamfit.fit(
        {"y": y, "lat": lat, "lon": lon},
        "y ~ sphere(lat, lon, k=24)",
        family="gaussian",
    )
    summary = model.summary()
    print(f"sphere fit: {summary['family_name']} deviance={summary['deviance']:.3f}")


if __name__ == "__main__":
    main()
