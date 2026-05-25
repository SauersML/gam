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
    z = rng.normal(size=n)
    y = np.cos(u + v) + 0.45 * np.sin(2.0 * u) - 0.35 * np.cos(v) + 0.18 * z
    df = pd.DataFrame({"u": u, "v": v, "z": z, "y": y + rng.normal(scale=0.12, size=n)})
    result = gamfit.select_topology(
        df,
        "y ~ s(u, v, type=AUTO) + z",
        return_fits=False,
    )

    print(f"selected topology: {result.winner_name}")


if __name__ == "__main__":
    main()
