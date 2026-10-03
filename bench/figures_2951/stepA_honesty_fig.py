"""Our search's choice of VPD's subcomponents against VPD's adversary, layer by layer, and under random
leak-back (#2951). Data: robust's ~/mpd-data/frontier/stepA_honesty.json (32 frontier passages of VPD's
4-layer Pile model)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

d = json.load(open(Path.home() / "mpd-data/frontier/stepA_honesty.json"))
box, layers = d["box"], d["layers"]
INK, INK2, MUTED, SURF, AXIS = "#0b0b0b", "#52514e", "#898781", "#ffffff", "#c3c2b7"
ORANGE, BLACK = "#eb6834", "#0b0b0b"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 24})
FAM = [("vpd_rounded", "VPD's choice", ORANGE), ("ours_corner", "our search's choice (on VPD's subcomponents)", BLACK)]
VAL = 20  # value labels


def frame(ax, title):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=20)
    ax.set_title(title, color=INK, fontsize=28, loc="left", pad=18)


fig, axes = plt.subplots(1, 3, figsize=(24, 10.5), dpi=150, gridspec_kw={"width_ratios": [1.3, 1.15, 0.85]})
fig.patch.set_facecolor(SURF)

# 1: VPD's adversary, step by step
ax = axes[0]
frame(ax, "VPD's adversary")
steps = [0, 20, 40, 80]
for fam, label, color in FAM:
    for mode, ls in (("delta_off", "-"), ("delta_adversarial", (0, (4, 3)))):
        ladder = box["pgd_shared"][f"{fam}/{mode}"]
        y = [box["fixed_kl"][fam]["mean"]] + [ladder[str(s)] for s in steps[1:]]
        ax.plot(steps, y, color=color, lw=3, ls=ls, zorder=3)
        ax.scatter(steps, y, s=70, color=color, edgecolor=SURF, linewidth=2, zorder=4)
    end = box["pgd_shared"][f"{fam}/delta_off"]["80"]
    ax.annotate(f"{end:.3g}", (80, end), xytext=(12, -16 if end > 10 else 0), textcoords="offset points",
                va="center", fontsize=VAL, color=INK)
off = box["fixed_kl"]["all_off"]["mean"]
ax.plot([0, 80], [off, off], color=MUTED, lw=1.6, zorder=2)
ax.annotate(f"all off: {off:.1f}", (80, off), xytext=(12, 10), textcoords="offset points", va="center",
            fontsize=VAL, color=INK2)
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30, 100]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(0.2, 120)
ax.set_xticks(steps)
ax.set_xticklabels(["none", "20", "40", "80"])
ax.set_xlim(-5, 120)
ax.set_xlabel("adversary steps", color=INK, labelpad=10)
ax.set_ylabel("KL from the model, per word", color=INK, labelpad=12)
ax.legend(handles=[Line2D([], [], color=INK2, lw=3, label="VPD's leftover kept off"),
                   Line2D([], [], color=INK2, lw=3, ls=(0, (4, 3)), label="adversary sets the leftover too")],
          loc="lower right", frameon=False, fontsize=19, labelcolor=INK)

# 2: one layer at a time, against all four at once
ax = axes[1]
frame(ax, "One layer at a time")
W = 0.32
for j, (fam, label, color) in enumerate(FAM):
    v = layers[fam]["kl_only_layer"]
    x = np.arange(4) + (j - 0.5) * (W + 0.1)
    ax.bar(x, v, width=W, color=color, zorder=3)
    for xi, vi in zip(x, v):
        ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=VAL - 3, color=INK, zorder=6, bbox=dict(boxstyle="square,pad=0.02", fc=SURF, ec="none"))
    joint = layers[fam]["kl_joint"]
    ax.plot([-0.5, 3.55], [joint, joint], color=color, lw=2.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"all four: {joint:.2f}", (3.55, joint), xytext=(8, 7 if j == 0 else -7), textcoords="offset points",
                ha="left", va="bottom" if j == 0 else "top", fontsize=VAL - 1, color=INK)
ax.set_xticks(range(4))
ax.set_xticklabels([f"layer {L + 1}" for L in range(4)], fontsize=21, color=INK)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, 5.0)
ax.set_ylim(0, 1.0)
ax.set_ylabel("KL, per word", color=INK, labelpad=10)

# 3: random leak-back
ax = axes[2]
n = box["draws_per_word"]
frame(ax, f"Random leak-back ({n} tries)")
W = 0.34
for j, (fam, label, color) in enumerate(FAM):
    dd = box["draws"][f"{fam}/delta_off"]
    v = [box["fixed_kl"][fam]["mean"], dd["all_draws"]["mean"], dd["worst_of_draws_per_word"]["mean"]]
    x = np.arange(3) + (j - 0.5) * (W + 0.04)
    ax.bar(x, v, width=W, color=color, zorder=3)
    for xi, vi in zip(x, v):
        ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=VAL - 1, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["off means\noff", "average\ntry", "worst try\nper word"], fontsize=20, color=INK)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 0.52)
ax.set_ylabel("KL, per word", color=INK, labelpad=10)

fig.legend(handles=[Line2D([], [], color=c, lw=9, label=l) for _, l, c in FAM], loc="upper left",
           bbox_to_anchor=(0.035, 0.905), ncol=2, frameon=False, fontsize=23, labelcolor=INK, columnspacing=3)
fig.suptitle("Our search's choice is closer to the model than VPD's when off means off, but VPD's adversary breaks it",
             color=INK, fontsize=29, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.subplots_adjust(left=0.055, right=0.985, top=0.74, bottom=0.15, wspace=0.3)
out = Path.home() / "mpd-data/figures/stepA_honesty.png"
fig.savefig(out, facecolor=SURF)
print(out)
