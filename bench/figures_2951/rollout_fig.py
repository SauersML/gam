"""VPD's error along 32-token rollouts the real 4L model wrote itself: per word, and summed (#2951)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

d = json.load(open(Path.home() / "mpd-data/figures/data/rollout_kl.json"))
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
ORANGE, BLUE = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
H = d["horizon"]
t = np.arange(1, H + 1)
series = [("rounded_whole", ORANGE, "VPD, on/off masks"), ("ci_whole", BLUE, "VPD, graded masks")]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), dpi=200)
fig.patch.set_facecolor(SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=12.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(0.5, H + 0.5)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_xlabel("token of the reply", color=INK, labelpad=8)

ax = axes[0]
for k, color, label in series:
    x = np.array(d[k])
    if k == "rounded_whole":
        for row in x:
            ax.plot(t, row, color=color, lw=0.9, alpha=0.18, zorder=2)
    ax.plot(t, x.mean(0), color=color, lw=2.4, zorder=4, solid_capstyle="round", label=label)
ax.set_ylim(0, 2.8)
ax.set_title("difference from the real model at each token (KL)", color=INK, fontsize=15.5, loc="left", pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=13, labelcolor=INK)
ax.text(H, 1.7, "bold: average of 8 replies\nfaint: each reply, on/off masks", color=MUTED, fontsize=12, ha="right")

ax = axes[1]
for k, color, label in series:
    x = np.cumsum(np.array(d[k]), 1)
    if k == "rounded_whole":
        for row in x:
            ax.plot(t, row, color=color, lw=0.9, alpha=0.18, zorder=2)
    m = x.mean(0)
    ax.plot(t, m, color=color, lw=2.4, zorder=4, solid_capstyle="round")
    ax.scatter([t[-1]], [m[-1]], s=60, color=color, edgecolor=SURF, linewidth=2, zorder=5)
    ax.annotate(f"{m[-1]:.1f}", (t[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=13.5, color=INK)
ax.set_ylim(0, 31)
ax.set_title("added up over the reply so far (KL)", color=INK, fontsize=15.5, loc="left", pad=12)

fig.suptitle("VPD's errors add up token by token as the model writes, but they don't snowball",
             color=INK, fontsize=20, fontweight="bold", x=0.035, ha="left", y=0.975)
fig.text(0.035, 0.895, "8 replies of 32 tokens that VPD's 4-layer target model wrote itself, each after 16 tokens of real text",
         color=INK2, fontsize=14, ha="left")
fig.subplots_adjust(left=0.05, right=0.965, top=0.79, bottom=0.12, wspace=0.18)
out = Path.home() / "mpd-data/figures/rollout_drift.png"
fig.savefig(out, facecolor=SURF)
print(out)
