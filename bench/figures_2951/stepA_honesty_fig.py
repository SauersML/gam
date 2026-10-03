"""Step A's corner win against VPD's adversary, layer by layer, and under random leak-back (#2951).
Data: robust's ~/mpd-data/frontier/stepA_honesty.json (32 frontier passages of VPD's 4-layer Pile model)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

d = json.load(open(Path.home() / "mpd-data/frontier/stepA_honesty.json"))
box, layers = d["box"], d["layers"]
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
ORANGE, BLACK = "#eb6834", "#0b0b0b"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
FAM = [("vpd_rounded", "VPD's choice of subcomponents", ORANGE), ("ours_corner", "our choice (Step A)", BLACK)]
words = layers["rows"] * 512


def frame(ax, title):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=12)
    ax.set_title(title, color=INK, fontsize=14.5, loc="left", pad=12)


def plain_log(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.8, zorder=0)


fig, axes = plt.subplots(1, 3, figsize=(19, 7.6), dpi=200, gridspec_kw={"width_ratios": [1.25, 1.1, 0.8]})
fig.patch.set_facecolor(SURF)

# 1: VPD's adversary, step by step
ax = axes[0]
frame(ax, "under VPD's adversary, which partly turns the\nswitched-off subcomponents back on, in the worst way")
steps = [0, 20, 40, 80]
for fam, label, color in FAM:
    for mode, ls in (("delta_off", "-"), ("delta_adversarial", (0, (4, 3)))):
        ladder = box["pgd_shared"][f"{fam}/{mode}"]
        y = [box["fixed_kl"][fam]["mean"]] + [ladder[str(s)] for s in steps[1:]]
        ax.plot(steps, y, color=color, lw=2.3, ls=ls, zorder=3)
        ax.scatter(steps, y, s=45, color=color, edgecolor=SURF, linewidth=1.6, zorder=4)
    end = box["pgd_shared"][f"{fam}/delta_off"]["80"]
    ax.annotate(f"{end:.3g}", (80, end), xytext=(8, 0), textcoords="offset points", va="center", fontsize=12.5, color=INK)
off = box["fixed_kl"]["all_off"]["mean"]
ax.axhline(off, color=MUTED, lw=1.1, zorder=2)
ax.text(1, off * 1.12, f"every subcomponent off: {off:.1f}", fontsize=12, color=INK2, va="bottom")
plain_log(ax)
ax.set_ylim(0.2, 120)
ax.set_xticks(steps)
ax.set_xticklabels(["none\n(off means off)", "20", "40", "80"])
ax.set_xlim(-4, 92)
ax.set_xlabel("steps the adversary takes", color=INK, labelpad=8)
ax.set_ylabel("difference from the model's outputs (KL, per word)", color=INK, labelpad=10)
ax.legend(handles=[Line2D([], [], color=INK2, lw=2.3, label="VPD's leftover part kept off"),
                   Line2D([], [], color=INK2, lw=2.3, ls=(0, (4, 3)), label="adversary also sets the leftover part")],
          loc="lower right", frameon=False, fontsize=12, labelcolor=INK)

# 2: one layer at a time
ax = axes[1]
frame(ax, "applying the choice in one layer only\n(the other three layers run in full)")
W = 0.36
for j, (fam, label, color) in enumerate(FAM):
    v = layers[fam]["kl_only_layer"]
    x = np.arange(4) + (j - 0.5) * (W + 0.04)
    ax.bar(x, v, width=W, color=color, zorder=3)
    for xi, vi in zip(x, v):
        ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color=INK, zorder=6, bbox=dict(boxstyle="square,pad=0.08", fc=SURF, ec="none"))
    joint = layers[fam]["kl_joint"]
    ax.axhline(joint, color=color, lw=1.6, ls=(0, (4, 3)), zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels([f"layer {L + 1}" for L in range(4)], fontsize=13, color=INK)
ax.text(0.02, 0.98, "dashed: all four layers at once — VPD's {:.2f}, ours {:.2f}".format(
    layers["vpd_rounded"]["kl_joint"], layers["ours_corner"]["kl_joint"]), transform=ax.transAxes, fontsize=12,
    color=INK, va="top")
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_ylabel("KL, per word", color=INK, labelpad=8)

# 3: random leak-back
ax = axes[2]
n = box["draws_per_word"]
frame(ax, f"random leak-back, {n} tries per word")
W = 0.34
for j, (fam, label, color) in enumerate(FAM):
    dd = box["draws"][f"{fam}/delta_off"]
    v = [box["fixed_kl"][fam]["mean"], dd["all_draws"]["mean"], dd["worst_of_draws_per_word"]["mean"]]
    x = np.arange(3) + (j - 0.5) * (W + 0.04)
    ax.bar(x, v, width=W, color=color, zorder=3)
    for xi, vi in zip(x, v):
        ax.annotate(f"{vi:.2f}", (xi, vi), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(["off means\noff", "average\ntry", f"worst of {n}\nper word"], fontsize=12.5, color=INK)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 0.55)
ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_ylabel("KL, per word", color=INK, labelpad=8)

fig.legend(handles=[Line2D([], [], color=c, lw=6, label=l) for _, l, c in FAM], loc="upper left",
           bbox_to_anchor=(0.04, 0.875), ncol=2, frameon=False, fontsize=13.5, labelcolor=INK, columnspacing=2.5)
o, v = box["fixed_kl"]["ours_corner"]["mean"], box["fixed_kl"]["vpd_rounded"]["mean"]
fig.suptitle("Our Step A choice is closer to the model than VPD's when off means off, but VPD's adversary breaks it",
             color=INK, fontsize=19, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.text(0.02, 0.905, f"VPD's 4-layer Pile model, the {layers['rows']} frontier passages ({words:,} words); "
         f"with every chosen subcomponent exactly on and the rest exactly off, KL is {o:.3f} for ours and {v:.3f} for VPD's",
         color=INK2, fontsize=13.5, ha="left")
fig.subplots_adjust(left=0.05, right=0.985, top=0.72, bottom=0.13, wspace=0.25)
out = Path.home() / "mpd-data/figures/stepA_honesty.png"
fig.savefig(out, facecolor=SURF)
print(out)
