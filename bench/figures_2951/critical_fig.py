"""VPD pieces switched on per word against how hard the word is for the model, and how badly VPD
explains it (#2951). Data from critical_data.py."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

d = np.load(Path.home() / "mpd-data/figures/data/critical.npz")
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
NB = 6
keep = np.ones(d["pieces"].shape, bool)
keep[:, 0] = False  # the first word of a chunk is an outlier of its own (535 pieces)
pieces = d["pieces"][keep].astype(float)
mean = pieces.mean()
panels = [("entropy", "how unsure the model is\nabout the next word", ("sure", "very unsure")),
          ("loss", "how surprising the real\nnext word was to the model", ("expected", "very surprising")),
          ("kl", "how far VPD's explanation\nis from the model at this word", ("close", "far off"))]

fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, (key, title, (lo, hi)) in zip(axes, panels):
    x = d[key][keep]
    edges = np.quantile(x, np.linspace(0, 1, NB + 1))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, NB - 1)
    h = [pieces[b == i].mean() for i in range(NB)]
    ax.set_facecolor(SURF)
    ax.bar(range(NB), h, width=0.62, color=BLUE, zorder=3)
    for i, v in enumerate(h):
        ax.annotate(f"{v:.0f}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12.5, color=INK, zorder=5)
    ax.axhline(mean, color=INK2, lw=1.1, zorder=4)
    ax.set_xticks(range(NB))
    ax.set_xticklabels([lo] + [""] * (NB - 2) + [hi], fontsize=13.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", colors=INK2, labelsize=12.5, length=0)
    ax.set_title(title, color=INK, fontsize=15, loc="left", pad=14)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
axes[0].set_ylabel("VPD pieces switched on, per word", color=INK, labelpad=10)
axes[0].set_ylim(0, 320)
fig.suptitle("Only the easiest words get fewer VPD pieces; the words VPD explains worst get the most",
             color=INK, fontsize=19.5, fontweight="bold", x=0.025, ha="left", y=0.975)
fig.text(0.025, 0.885, f"{len(pieces):,} words of web text through VPD's 4-layer model; each bar is one sixth of the words; "
         f"the line is the average word ({mean:.0f} pieces)",
         color=INK2, fontsize=14, ha="left")
fig.subplots_adjust(left=0.065, right=0.985, top=0.74, bottom=0.1, wspace=0.12)
out = Path.home() / "mpd-data/figures/critical_words.png"
fig.savefig(out, facecolor=SURF)
print(out)
