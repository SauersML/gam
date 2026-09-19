"""Draw the README's mcycle figure from a gamfit location-scale fit.

The figure shows the README example: the posterior mean of head acceleration,
its 95% credible band, and the 95% observation interval of a fit whose noise
level is itself a smooth of time. Every plotted number comes from
``Model.predict``; this script only draws it.

    python scripts/gen_mcycle_figure.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gamfit

MCYCLE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "mcycle_location_scale.png"


def main(output: Path) -> None:
    mcycle = pd.read_csv(MCYCLE_URL)
    model = gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")
    grid = pd.DataFrame({"times": np.linspace(mcycle["times"].min(), mcycle["times"].max(), 400)})
    bands = model.predict(grid, interval=0.95, observation_interval=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.fill_between(grid["times"], bands["observation_lower"], bands["observation_upper"],
                    color="tab:blue", alpha=0.15, linewidth=0, label="95% observation interval")
    ax.fill_between(grid["times"], bands["posterior_mean_lower"], bands["posterior_mean_upper"],
                    color="tab:blue", alpha=0.35, linewidth=0, label="95% credible band for the mean")
    ax.plot(grid["times"], bands["posterior_mean"], color="tab:blue", linewidth=1.8, label="posterior mean")
    ax.scatter(mcycle["times"], mcycle["accel"], s=9, color="black", alpha=0.7, label="mcycle data")
    ax.set_xlabel("time after impact (ms)")
    ax.set_ylabel("head acceleration (g)")
    ax.set_title('gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")', fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
