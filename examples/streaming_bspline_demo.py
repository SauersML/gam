"""Streaming B-spline basis demo.

The Rust core auto-activates row-chunked streaming whenever the would-be
dense basis buffer exceeds ~1 GiB, so this demo simply constructs a
problem large enough to cross that threshold and shows that peak RSS stays
bounded — no streaming opt-in arg is needed (and none exists).
"""

from __future__ import annotations

import resource
import sys

import numpy as np

import gamfit


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def main() -> None:
    # Pick n large enough that n * k * 8 bytes exceeds the 1 GiB auto-stream
    # threshold so that streaming is selected automatically by the core.
    k = 64
    n = max(20_000, (1024 * 1024 * 1024) // (k * 8) + 1)
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
        f"y ~ bspline(t, k={k}, periodic=true, period=2*pi, origin=0)",
        family="gaussian",
    )
    print(model.summary())

    delta = rss_mb() - baseline
    print(f"peak RSS delta: {delta:.1f} MB")
    assert delta < 2048.0


if __name__ == "__main__":
    main()
