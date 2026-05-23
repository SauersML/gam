"""Synthetic topology auto-selection demo.

Do not run as part of CI: this is a small interactive experiment showing
``gamfit.select_topology`` on a toroidal signal with Euclidean nuisance noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def make_torus_data(n: int = 700, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.normal(size=n)

    signal = np.cos(u + v) + 0.45 * np.sin(2.0 * u) - 0.35 * np.cos(v)
    y = signal + 0.18 * z + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"u": u, "v": v, "z": z, "y": y})


def main() -> None:
    df = make_torus_data()
    result = gamfit.select_topology(
        df,
        "y ~ s(u, v, type=AUTO) + z",
        return_fits=False,
    )
    assert result["winner"] == "Torus", result["evidence_summary"]

    print(result["evidence_summary"])
    for name, reml, delta, bf, _cv_r2 in result["ranking"]:
        print(f"{name:15s} REML={reml:12.4f} delta={delta:10.4f} BF_vs_best={bf:.3g}")


if __name__ == "__main__":
    main()
