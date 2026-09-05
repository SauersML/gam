"""Plot the bounded, matched node-reduction measurements for #2827."""

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source, destination = map(Path, sys.argv[1:])
    with source.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    sizes = sorted({int(row["rows"]) for row in records})
    fig, axes = plt.subplots(1, len(sizes), figsize=(9, 4.7))
    fig.patch.set_facecolor("#f7f8fa")
    colors = ["#b46c3f", "#28796b"]
    for axis, size in zip(axes, sizes):
        subset = {row["variant"]: row for row in records if int(row["rows"]) == size}
        before, after = (float(subset[key]["seconds"]) for key in ["indexed", "iterators"])
        bars = axis.bar([0, 1], [before, after], color=colors, width=0.58)
        axis.set_facecolor("#f7f8fa")
        axis.set_xticks([0, 1], ["Indexed loop", "Row iterators"])
        axis.set_title(f"{size:,} rows · 100 columns", loc="left", fontweight="bold", pad=18)
        axis.set_ylabel("Seconds per completed node reduction")
        axis.set_ylim(0, before * 1.33)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#c6ccd2")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#dfe3e7", linewidth=0.7)
        for bar, value in zip(bars, [before, after]):
            axis.text(bar.get_x() + bar.get_width() / 2, value + before * 0.025,
                      f"{value:.3f} s", ha="center", fontsize=11, fontweight="bold")
        axis.text(0.96, 0.96, f"{before / after:.2f}× faster", transform=axis.transAxes,
                  ha="right", va="top", color=colors[1], fontweight="bold", fontsize=12)
    fig.suptitle("Same compensated arithmetic, less indexing work", x=0.075, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.075, 0.035,
             "Single matched measurements · warm opt-level-2 test profile · four-thread setting\n"
             "Synthetic row-major inputs; excludes design construction. Not an end-to-end fit benchmark.",
             color="#535f6c", fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.23, top=0.80, wspace=0.38)
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
