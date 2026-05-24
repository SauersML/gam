"""Streaming B-spline basis demo."""

from __future__ import annotations

import resource
import sys

import numpy as np

import gamfit


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def main() -> None:
    n = 20_000
    chunk = 2048
    rng = np.random.default_rng(31)
    baseline = rss_mb()

    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    y = (
        1.2 * np.sin(t)
        - 0.7 * np.cos(2.0 * t)
        + 0.35 * np.sin(3.0 * t + 0.4)
        + rng.normal(scale=0.08, size=n)
    )

    data = {"y": y, "t": t}
    model = gamfit.fit(
        data,
        (
            "y ~ bspline(t, k=64, periodic=true, period=2*pi, origin=0, "
            f"streaming_chunk_size={chunk})"
        ),
        family="gaussian",
    )
    print(model.summary())

    delta = rss_mb() - baseline
    print(f"peak RSS delta: {delta:.1f} MB")
    assert delta < 512.0


if __name__ == "__main__":
    main()
