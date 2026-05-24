"""Streaming Sphere basis demo.

The Rust core auto-activates row-chunked streaming whenever the would-be
dense basis buffer exceeds ~1 GiB, so this demo sizes n × k above that
threshold and observes that peak RSS stays bounded — no streaming opt-in
arg is needed (and none exists).
"""

from __future__ import annotations

import resource
import sys

import numpy as np

import gamfit


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def cos_gamma(lat_lon: np.ndarray, centers: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_lon[:, 0])[:, None]
    lon = np.deg2rad(lat_lon[:, 1])[:, None]
    clat = np.deg2rad(centers[:, 0])[None, :]
    clon = np.deg2rad(centers[:, 1])[None, :]
    return np.sin(lat) * np.sin(clat) + np.cos(lat) * np.cos(clat) * np.cos(lon - clon)


def main() -> None:
    k = 64
    # Size n so that n * k * 8 bytes exceeds the 1 GiB auto-stream threshold,
    # exercising the auto-streaming path.
    n = max(20_000, (1024 * 1024 * 1024) // (k * 8) + 1)
    rng = np.random.default_rng(23)
    baseline = rss_mb()

    lat = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    x = np.column_stack([lat, lon])
    centers = x[rng.choice(n, size=32, replace=False)]
    coef = rng.normal(scale=0.5, size=centers.shape[0])

    y = np.zeros(n)
    block = 2048
    for start in range(0, n, block):
        end = min(start + block, n)
        kmat = np.exp(4.0 * (cos_gamma(x[start:end], centers) - 1.0))
        y[start:end] = kmat @ coef
    y += rng.normal(scale=0.05, size=n)

    data = {"y": y, "lat": lat, "lon": lon}
    model = gamfit.fit(
        data,
        f"y ~ sphere(lat, lon, k={k}, kernel=sobolev)",
        family="gaussian",
    )
    print(model.summary())

    delta = rss_mb() - baseline
    print(f"peak RSS delta: {delta:.1f} MB")
    assert delta < 2048.0


if __name__ == "__main__":
    main()
