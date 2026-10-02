"""Shared-structure census: bits each kind of shared structure saves, as a share of the model (#2951)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

F = Path.home() / "mpd-data/frontier"
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, AQUA = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})

ROWS = [("each matrix is low-rank", lambda r: r["per_matrix"]["saved_vs_dense"], False),
        ("matrices share directions", lambda r: sum(v["saved"] for v in r["subspace"].values()), False),
        ("neurons are copies of each other", lambda r: max(x["saved"] for v in r["clusters"].values() for x in v), False),
        ("heads are copies, up to rotation", lambda r: r["gauge"]["heads_up_to_gauge_saved_bits"], False),
        ("neurons and heads can be reordered", lambda r: r["gauge"]["permutation_bits"], False),
        ("heads can be rotated", lambda r: r["gauge"]["orbit_bound_bits"], True)]
MODELS = [("vpd4l", "VPD's 4-layer model", BLUE), ("pythia70m", "Pythia-70m", AQUA)]
LO = 3e-4  # percent; zeros sit here as "none"

fig, ax = plt.subplots(figsize=(14, 7.2), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for i in range(len(ROWS)):
    ax.axhline(i, color=GRID, lw=0.8, zorder=0)
for j, (m, label, color) in enumerate(MODELS):
    r = json.load(open(F / f"census_{m}.json"))
    for i, (name, f, bound) in enumerate(ROWS):
        y = len(ROWS) - 1 - i + (0.14 if j == 0 else -0.14)
        pct = 100 * f(r) / r["dense_bits"]
        if pct <= 0:
            ax.text(LO, y, "none", color=MUTED, fontsize=12, va="center", ha="left")
            continue
        ax.scatter(pct, y, s=90, zorder=4, linewidth=2,
                   facecolor=SURF if bound else color, edgecolor=color)
        ax.annotate(("at most " if bound else "") + (f"{pct:.2f}%" if pct >= 0.01 else f"{pct:.3f}%"),
                    (pct, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=12, color=INK2)
    ax.scatter([], [], s=90, color=color, label=label)
ax.axvline(100, color=INK, lw=1.4, zorder=1)
ax.annotate("the whole\nmodel", (100, len(ROWS) - 0.45), xytext=(-8, 0), textcoords="offset points",
            ha="right", va="bottom", fontsize=13, color=INK, multialignment="right")
ax.set_xscale("log")
ax.set_xlim(LO / 1.3, 160)
ax.set_ylim(-0.6, len(ROWS) - 0.1)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([n for n, _, _ in ROWS][::-1], fontsize=14.5, color=INK)
ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
ax.set_xticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])
ax.set_xlabel("bits saved, as a share of the whole model", color=INK, labelpad=10)
ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", colors=INK2, labelsize=12.5)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
fig.legend(loc="upper right", bbox_to_anchor=(0.97, 0.93), ncol=2, frameon=False, fontsize=13.5, labelcolor=INK, columnspacing=2)
fig.suptitle("Real language models share almost nothing", color=INK, fontsize=21, fontweight="bold",
             x=0.025, ha="left", y=0.975)
fig.text(0.025, 0.905, "every kind of shared structure we searched for, measured in bits the weights would save",
         color=INK2, fontsize=14.5, ha="left")
fig.subplots_adjust(left=0.28, right=0.97, top=0.83, bottom=0.12)
out = Path.home() / "mpd-data/figures/census_shared.png"
fig.savefig(out, facecolor=SURF)
print(out)
