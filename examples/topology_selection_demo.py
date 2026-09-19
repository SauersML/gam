"""Fit a synthetic torus signal with topology auto-selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def main() -> None:
    rng = np.random.default_rng(17)
    n = 256
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    y = np.cos(u + v) + 0.45 * np.sin(2.0 * u) - 0.35 * np.cos(v)
    df = pd.DataFrame({"u": u, "v": v, "y": y + rng.normal(scale=0.12, size=n)})
    # select_topology takes the response column and races candidate topologies
    # for one smooth over every other column: here `y ~ s(u, v, type=AUTO)`.
    result = gamfit.topology.select_topology(df, "y", return_fits=False)

    print(f"selected topology: {result.winner_name}")


if __name__ == "__main__":
    main()
