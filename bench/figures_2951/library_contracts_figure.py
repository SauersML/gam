"""The induction circuit's contracts tested by patching (mpd_library_contracts_2951 JSON).

python library_contracts_figure.py CONTRACTS.json OUT.png

Three heads (previous-token, induction, copy) left to right; the arrows are the reads the
contracts name (the previous-token head into the induction head's key, the induction head into the
copy head's value). Under each head, its predictions with the number of targets where each held. A
target is a position in the second copy of a repeated passage; its earlier copy is the same position
in the first copy. A prediction counts as held when it holds at a majority of targets (filled
marker) and failed otherwise (open marker).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LABEL = {
    "previous selects t-1": "depends on the previous token",
    "induction selects j, j+1": "depends on the earlier copy\nand the token after it",
    "induction key reads previous": "its key reads L1's head\nmore than its query does",
    "induction writes next": "raises the token that\nfollowed the earlier copy",
    "copy value reads induction": "its value reads L2's head\nmore than its query does",
    "copy writes next": "raises the token that\nfollowed the earlier copy",
}
OWNER = {
    "previous selects t-1": "previous",
    "induction selects j, j+1": "induction",
    "induction key reads previous": "induction",
    "induction writes next": "induction",
    "copy value reads induction": "copy",
    "copy writes next": "copy",
}
INK, MUTED, BOX, ARROW = "#1f1f1e", "#6b6b68", "#eef2f7", "#3b6ea5"
HELD, FAILED = "#2e7d32", "#c62828"


def main():
    report = json.load(open(sys.argv[1]))
    heads = report["heads"]
    x = {"previous": 0.0, "induction": 4.6, "copy": 9.2}
    role = {"previous": "previous token", "induction": "induction", "copy": "copy"}
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(16, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for key, cx in x.items():
        ax.add_patch(FancyBboxPatch((cx - 1.2, 2.3), 2.4, 1.1, boxstyle="round,pad=0.08", facecolor=BOX, edgecolor=INK, linewidth=2))
        ax.text(cx, 3.02, heads[key], ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
        ax.text(cx, 2.6, role[key], ha="center", va="center", fontsize=17, color=INK)
    for (a, b, label) in [("previous", "induction", "key"), ("induction", "copy", "value")]:
        ax.add_patch(FancyArrowPatch((x[a] + 1.3, 2.85), (x[b] - 1.3, 2.85), arrowstyle="-|>", mutation_scale=26, linewidth=3, color=ARROW))
        ax.text((x[a] + x[b]) / 2, 3.05, label, ha="center", va="bottom", fontsize=17, color=ARROW)
    rows = {k: 0 for k in x}
    for p in report["predictions"]:
        owner = OWNER[p["name"]]
        held = p["fraction"] > 0.5
        y = 1.95 - 0.8 * rows[owner]
        rows[owner] += 1
        color = HELD if held else FAILED
        lines = LABEL[p["name"]].count("\n") + 1
        ax.scatter([x[owner] - 1.15], [y - 0.09], s=200, marker="o", facecolors=color if held else "white", edgecolors=color, linewidths=2.5, zorder=3)
        ax.text(x[owner] - 0.9, y, LABEL[p["name"]], ha="left", va="top", fontsize=16, color=INK, linespacing=1.15)
        ax.text(x[owner] - 0.9, y - 0.19 * lines - 0.04, f"held at {p['held']} of {p['targets']} targets", ha="left", va="top", fontsize=15, color=color)
    ax.set_xlim(-1.6, 11.6)
    ax.set_ylim(-0.6, 3.6)
    ax.set_axis_off()
    fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
