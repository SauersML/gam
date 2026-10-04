"""Draw the README's mcycle figure from a gamfit location-scale fit.

The figure shows the README example: the posterior mean of head acceleration,
its 95% credible band, and the 95% observation interval of a fit whose noise
level is itself a smooth of time. Every plotted number comes from
``Model.predict``; this script only draws it.

It writes a light and a dark variant on transparent backgrounds. The README
picks one through ``<picture>`` and ``prefers-color-scheme``.

    python scripts/gen_mcycle_figure.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import gamfit

MCYCLE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"
STEM = "mcycle_location_scale"

SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
MONO = ["JetBrainsMono NF", "Menlo", "DejaVu Sans Mono"]

# One hue (blue) for everything the model says, ink for the data. The surface
# colors are GitHub's page backgrounds; they only colour the ring around each
# data point, because the figure itself is transparent.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "ink_2": "#59636e",
        "ink_3": "#818b98",
        "grid": "#e6e8eb",
        "zero": "#c9ced4",
        "hue": "#2a78d6",
        "mean": "#184f95",
        "obs_alpha": 0.13,
        "cred_alpha": 0.30,
        "dot": "#1f2328",
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#f0f6fc",
        "ink_2": "#9198a1",
        "ink_3": "#6e7681",
        "grid": "#21262d",
        "zero": "#3d444d",
        "hue": "#3987e5",
        "mean": "#86b6ef",
        "obs_alpha": 0.17,
        "cred_alpha": 0.36,
        "dot": "#e6edf3",
    },
}


def fit():
    mcycle = pd.read_csv(MCYCLE_URL)
    model = gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")
    grid = pd.DataFrame({"times": np.linspace(mcycle["times"].min(), mcycle["times"].max(), 600)})
    bands = model.predict(grid, interval=0.95, observation_interval=True)
    at_data = model.predict(mcycle, interval=0.95, observation_interval=True)
    covered = (mcycle["accel"] >= at_data["observation_lower"]) & (mcycle["accel"] <= at_data["observation_upper"])
    return mcycle, grid, bands, float(covered.mean())


def draw(mcycle, grid, bands, coverage, theme: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": 9,
        "axes.unicode_minus": True,
    })
    t = grid["times"].to_numpy()
    fig = plt.figure(figsize=(8.0, 4.6), dpi=220)
    ax = fig.add_axes((0.085, 0.13, 0.885, 0.62))

    ax.fill_between(t, bands["observation_lower"], bands["observation_upper"],
                    color=theme["hue"], alpha=theme["obs_alpha"], linewidth=0, zorder=1)
    ax.fill_between(t, bands["posterior_mean_lower"], bands["posterior_mean_upper"],
                    color=theme["hue"], alpha=theme["cred_alpha"], linewidth=0, zorder=2)
    ax.plot(t, bands["posterior_mean"], color=theme["mean"], linewidth=1.9,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(mcycle["times"], mcycle["accel"], s=15, color=theme["dot"], alpha=0.9,
               edgecolors=theme["surface"], linewidths=0.7, zorder=4)

    ax.axhline(0, color=theme["zero"], linewidth=0.8, zorder=0.5)
    ax.set_xlim(0, 60)
    ax.set_ylim(-180, 112)
    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks(range(-150, 101, 50))
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=6, labelsize=8.5, labelcolor=theme["ink_2"])
    ax.patch.set_alpha(0)

    # Axis titles sit at the ends of the axes instead of running along them.
    ax.text(0, 1.02, "head acceleration (g)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=theme["ink_2"])
    ax.text(1, -0.1, "milliseconds after impact", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=theme["ink_2"])

    # One annotation, on what a plain smoother cannot show.
    upper = bands["observation_upper"].to_numpy()
    target = int(np.argmin(np.abs(t - 36.5)))
    ax.annotate("the noise is fitted too: the interval is narrow\nwhile the head is still and widens once it moves",
                xy=(t[target], upper[target]), xytext=(40.0, 76), fontsize=8, color=theme["ink_2"],
                ha="left", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=theme["ink_3"], linewidth=0.7, relpos=(0, 0.5),
                                shrinkA=4, shrinkB=1, connectionstyle="arc3,rad=-0.3"))

    left = ax.get_position().x0
    fig.text(left, 0.945, "Both the mean and the noise are smooth functions of time",
             fontsize=13, fontweight="bold", color=theme["ink"], ha="left", va="top")
    fig.text(left, 0.885, 'gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")',
             family=MONO, fontsize=8.5, color=theme["ink_2"], ha="left", va="top")

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.2, color=theme["dot"],
               markeredgecolor=theme["surface"], markeredgewidth=0.6),
        Line2D([], [], color=theme["mean"], linewidth=1.9),
        Patch(facecolor=theme["hue"], alpha=theme["cred_alpha"] + theme["obs_alpha"], linewidth=0),
        Patch(facecolor=theme["hue"], alpha=theme["obs_alpha"], linewidth=0),
    ]
    labels = [
        f"{len(mcycle)} observations",
        "posterior mean",
        "95% credible band",
        f"95% observation interval (covers {coverage:.0%})",
    ]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left - 0.006, 0.835),
               ncol=4, frameon=False, fontsize=8.5, labelcolor=theme["ink"],
               handlelength=1.4, handleheight=0.9, handletextpad=0.55, columnspacing=1.6,
               borderaxespad=0, borderpad=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    plt.close(fig)


def main(output_dir: Path) -> None:
    mcycle, grid, bands, coverage = fit()
    draw(mcycle, grid, bands, coverage, THEMES["light"], output_dir / f"{STEM}.png")
    draw(mcycle, grid, bands, coverage, THEMES["dark"], output_dir / f"{STEM}_dark.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR)
