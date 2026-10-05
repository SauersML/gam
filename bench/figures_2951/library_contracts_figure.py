"""The induction circuit's contracts tested by patching (mpd_library_contracts_2951 JSON).

python library_contracts_figure.py CONTRACTS.json OUT.png

Three heads (previous-token, induction, copy) left to right; the arrows are the reads the
contracts name (the previous-token head into the induction head's key, the induction head into the
copy head's value). Under each head, its predictions with the fraction of targets where each held;
a prediction counts as held when it holds at a majority of targets (filled marker) and failed
otherwise (open marker).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SHORT = {
    "previous selects t-1": "selects t−1",
    "induction selects j, j+1": "selects j, j+1",
    "induction key reads previous": "key reads L1 head",
    "induction writes next": "writes x(j+1)",
    "copy value reads induction": "value reads L2 head",
    "copy writes next": "writes x(j+1)",
}
OWNER = {
    "previous selects t-1": "previous",
    "induction selects j, j+1": "induction",
    "induction key reads previous": "induction",
    "induction writes next": "induction",
    "copy value reads induction": "copy",
    "copy writes next": "copy",
}


def main():
    report = json.load(open(sys.argv[1]))
    heads = report["heads"]
    x = {"previous": 0.0, "induction": 4.2, "copy": 8.4}
    role = {"previous": "previous token", "induction": "induction", "copy": "copy"}
    fig, ax = plt.subplots(figsize=(14, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for key, cx in x.items():
        ax.add_patch(FancyBboxPatch((cx - 1.15, 2.3), 2.3, 1.1, boxstyle="round,pad=0.08", facecolor="#eef2f7", edgecolor="#2c3e50", linewidth=2))
        ax.text(cx, 3.02, heads[key], ha="center", va="center", fontsize=20, fontweight="bold")
        ax.text(cx, 2.6, role[key], ha="center", va="center", fontsize=15)
    for (a, b, label) in [("previous", "induction", "key"), ("induction", "copy", "value")]:
        ax.add_patch(FancyArrowPatch((x[a] + 1.25, 2.85), (x[b] - 1.25, 2.85), arrowstyle="-|>", mutation_scale=24, linewidth=3, color="#3b6ea5"))
        ax.text((x[a] + x[b]) / 2, 3.05, label, ha="center", va="bottom", fontsize=15, color="#3b6ea5")
    rows = {k: 0 for k in x}
    for p in report["predictions"]:
        owner = OWNER[p["name"]]
        held = p["fraction"] > 0.5
        y = 1.75 - 0.55 * rows[owner]
        rows[owner] += 1
        color = "#2e7d32" if held else "#c62828"
        ax.scatter([x[owner] - 1.05], [y], s=170, marker="o", facecolors=color if held else "white", edgecolors=color, linewidths=2.5, zorder=3)
        ax.text(x[owner] - 0.8, y, f"{SHORT[p['name']]}  {p['held']}/{p['targets']}", ha="left", va="center", fontsize=15, color=color)
    ax.set_xlim(-1.6, 10.2)
    ax.set_ylim(-0.2, 3.7)
    ax.set_axis_off()
    fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
