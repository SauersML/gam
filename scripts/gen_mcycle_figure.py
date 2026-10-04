"""Render the README's full-width, word-free location-scale figure.

All curves and interval boundaries come from the README's real gamfit fit.
Nested observation intervals (10–95%) make the changing noise visible; the
inner ribbon is the 95% credible band of the mean. Gold dots are the 133
unmodified measurements. Only short axis labels and numeric ticks are drawn.

    python -m scripts.gen_mcycle_figure [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import gamfit

MCYCLE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"
STEM = "mcycle_location_scale"
LEVELS = np.linspace(0.95, 0.10, 32)
SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

THEMES = {
    "light": {
        "surface": "#f7f9fc",
        "ink": "#47566c",
        "grid": "#dce3ed",
        "colors": ["#e4e7f6", "#aaa9e4", "#648dd9", "#32a6bd", "#87d9da"],
        "edge": "#6c7dbe",
        "credible": "#e9ffff",
        "mean": "#124f68",
        "dot": "#c97822",
        "dot_edge": "#fff3dc",
    },
    "dark": {
        "surface": "#0b1220",
        "ink": "#9cacc5",
        "grid": "#253148",
        "colors": ["#20213f", "#48408b", "#356aa7", "#238eaa", "#66cbd0"],
        "edge": "#6b64b3",
        "credible": "#c4f4f0",
        "mean": "#f1fffc",
        "dot": "#f2bc72",
        "dot_edge": "#513c2f",
    },
}


def fit():
    mcycle = pd.read_csv(MCYCLE_URL)
    model = gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")
    grid = pd.DataFrame({
        "times": np.linspace(mcycle["times"].min(), mcycle["times"].max(), 1200)
    })
    intervals = [
        model.predict(grid, interval=float(level), observation_interval=True)
        for level in LEVELS
    ]
    return mcycle, grid, intervals


def draw(mcycle, grid, intervals, theme: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": SANS, "font.size": 12})
    fig = plt.figure(figsize=(15, 9), dpi=240, facecolor=theme["surface"])
    ax = fig.add_axes((0.075, 0.105, 0.895, 0.86), facecolor=theme["surface"])
    t = grid["times"].to_numpy()
    bands = intervals[0]
    palette = LinearSegmentedColormap.from_list("intervals", theme["colors"])

    # Each layer is an actual prediction interval, not a decorative offset.
    for index, interval in enumerate(intervals):
        color = palette(index / (len(intervals) - 1))
        ax.fill_between(t, interval["observation_lower"], interval["observation_upper"],
                        color=color, linewidth=0, zorder=2)
        if index % 3 == 0:
            for bound in ("observation_lower", "observation_upper"):
                ax.plot(t, interval[bound], color=theme["surface"],
                        alpha=0.18, linewidth=0.6, zorder=2.1)

    for bound in ("observation_lower", "observation_upper"):
        ax.plot(t, bands[bound], color=theme["edge"], alpha=0.7,
                linewidth=1.0, zorder=3)

    ax.fill_between(t, bands["posterior_mean_lower"], bands["posterior_mean_upper"],
                    color=theme["credible"], alpha=0.58, linewidth=0, zorder=4)
    for bound in ("posterior_mean_lower", "posterior_mean_upper"):
        ax.plot(t, bands[bound], color=theme["credible"], alpha=0.8,
                linewidth=0.7, zorder=5)
    ax.plot(t, bands["posterior_mean"], color=theme["mean"], linewidth=2.6,
            solid_capstyle="round", solid_joinstyle="round", zorder=6)

    ax.scatter(mcycle["times"], mcycle["accel"], s=43,
               facecolors=theme["dot"], edgecolors=theme["dot_edge"],
               linewidths=0.8, zorder=7)

    # Preserve quantitative scale; leave generous breathing room at both ends.
    ax.set_xlim(0, 60)
    lower = min(mcycle["accel"].min(), bands["observation_lower"].min())
    upper = max(mcycle["accel"].max(), bands["observation_upper"].max())
    margin = (upper - lower) * 0.09
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_yticks(np.arange(-150, 101, 50))
    ax.grid(axis="y", color=theme["grid"], linewidth=0.65, alpha=0.65)
    ax.axhline(0, color=theme["grid"], linewidth=1, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=12, labelsize=11, labelcolor=theme["ink"])
    ax.set_xlabel("Time (ms)", color=theme["ink"], fontsize=12, labelpad=16)
    ax.set_ylabel("Acceleration (g)", color=theme["ink"], fontsize=12, labelpad=18)

    # Assert the text contract on the figure itself, including hidden titles.
    assert not ax.get_title() and not fig.texts and ax.get_legend() is None
    assert [ax.get_xlabel(), ax.get_ylabel()] == ["Time (ms)", "Acceleration (g)"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=theme["surface"])
    plt.close(fig)


def main(output_dir: Path) -> None:
    mcycle, grid, intervals = fit()
    draw(mcycle, grid, intervals, THEMES["light"], output_dir / f"{STEM}.png")
    draw(mcycle, grid, intervals, THEMES["dark"], output_dir / f"{STEM}_dark.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR)
