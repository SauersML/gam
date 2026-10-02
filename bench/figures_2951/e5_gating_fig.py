"""E5: how much of each VPD component's on/off a short rule over upstream components explains (#2951)."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

e = json.load(open(Path.home() / "mpd-data/frontier/e5_gating_rules.json"))
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, GRAY = "#2a78d6", "#b9b8b1"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
r = e["records"]
half_log2n = 0.5 * math.log2(e["coded_tokens"])
k = np.array([len(x["features"]) for x in r])
# share of the component's bits saved, each code charged its own model bits (the rate alone costs one coefficient)
saved = np.array([1 - (x["test_bits_rule"] + x["model_bits"]) / (x["test_bits_marginal"] + half_log2n) for x in r])
n, none = len(r), int((k == 0).sum())


def frame(ax):
    ax.set_facecolor(SURF)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", colors=INK2, labelsize=12.5, length=0)
    ax.tick_params(axis="x", colors=INK2, labelsize=12.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=200, gridspec_kw={"width_ratios": [2.1, 1]})
fig.patch.set_facecolor(SURF)
frame(ax)
frame(bx)

# left: distribution of savings for components that got a rule; the no-rule components as one bar at zero
W = 0.025
bins = np.arange(0, 0.7 + W, W)
h, _ = np.histogram(np.clip(saved[k > 0], 0, None), bins)
ax.bar(bins[:-1] + W / 2, h, width=W * 0.86, color=BLUE, zorder=3)
TOP = h.max() * 1.45  # the no-rule bar is cut here and labelled with its true height
ax.bar(-0.045, TOP * 0.97, width=W * 1.4, color=GRAY, zorder=3)
for dy in (0.80, 0.84):
    ax.plot([-0.045 - W, -0.045 + W], [TOP * dy, TOP * (dy + 0.03)], color=SURF, lw=3, zorder=4)
ax.annotate(f"no rule found:\n{none:,} components ({none / n:.0%})", (-0.045 + W, TOP * 0.93), xytext=(8, 0),
            textcoords="offset points", ha="left", va="top", fontsize=13, color=INK)
ax.set_ylim(0, TOP)
med = float(np.median(saved[k > 0]))
mb = int(med // W)
ax.annotate(f"typical component with a rule: {med:.0%}", (bins[mb] + W / 2, h[mb]), xytext=(30, 34), textcoords="offset points",
            ha="left", va="bottom", fontsize=13, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=2, shrinkB=3))
big = int((saved >= 0.5).sum())
ax.axvline(0.5, color=INK2, lw=1, zorder=2)
ax.annotate(f"half or more: {big} components", (0.5, h.max() * 0.45), xytext=(8, 0), textcoords="offset points",
            ha="left", va="center", fontsize=13, color=INK)
ax.set_xlim(-0.08, 0.72)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%"])
ax.set_xlabel("how much of a component's switching a small rule predicts", color=INK, labelpad=10)
ax.set_title("number of components", color=INK, fontsize=14.5, loc="left", pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

# right: how many inputs the rules use
fan = e["fan_in_histogram"]
ks = np.array(sorted(int(a) for a in fan))
cnt = np.array([fan[str(a)] for a in ks])
bx.bar(ks[ks > 0], cnt[ks > 0], width=0.7, color=BLUE, zorder=3)
bx.bar([0], cnt[ks == 0], width=0.7, color=GRAY, zorder=3)
bx.set_xticks([0, 2, 4, 6, 8, 10, 12])
bx.set_xlabel("how many other components the rule looks at", color=INK, labelpad=10)
bx.set_title("number of components", color=INK, fontsize=14.5, loc="left", pad=12)
bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

fig.suptitle("Can you predict when a VPD component switches on from the components below it? Mostly no", color=INK, fontsize=21,
             fontweight="bold", x=0.03, ha="left", y=0.975)
fig.text(0.03, 0.895, f"for each of {n:,} VPD components, the best small rule we could find from the components below it and the "
         "previous word, checked on new text",
         color=INK2, fontsize=13.5, ha="left")
fig.subplots_adjust(left=0.05, right=0.98, top=0.79, bottom=0.12, wspace=0.14)
out = Path.home() / "mpd-data/figures/e5_gating_rules.png"
fig.savefig(out, facecolor=SURF)
print(out, "median", med, "big", big, "max", saved.max())
