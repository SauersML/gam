"""Per-token explanation size against faithfulness: VPD's gates vs fixed bases (#2951).

usage: pertoken_frontier_fig.py {vpd4l|pythia70m}
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

MODEL = sys.argv[1] if len(sys.argv) > 1 else "vpd4l"
F = Path.home() / "mpd-data/frontier"
bases = json.load(open(F / f"pertoken_{MODEL}_bases.json"))
vpd_p = F / f"pertoken_{MODEL}_vpd.json"
vpd = json.load(open(vpd_p)) if vpd_p.exists() else None
eng_p = F / f"pertoken_{MODEL}_engine.json"  # the engine's slot: {"points": [{"l0", "bits", "kl"}]}
eng = json.load(open(eng_p)) if eng_p.exists() else None

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 15})

series = [(bases["native"]["points"], BLUE, "the model's own neurons\nand head dimensions", "-"),
          (bases["svd"]["points"], YELLOW, "each matrix's SVD", "-"),
          (bases["wsvd"]["points"], AQUA, "each matrix's SVD, weighted\nby what the output feels", "-")]
if vpd:
    series += [(vpd["rounded"], ORANGE, "VPD", "-"), (vpd["ci_p4"], ORANGE, "VPD, graded gates", (0, (4, 3)))]
if eng:
    series.append((eng["points"], INK, "our engine", "-"))


def xy(ps, x):
    """Points in the order the sweep made them (pieces increase along it; bits need not)."""
    ps = sorted((p for p in ps if p["kl"] > 1e-6 and p[x] > 0), key=lambda p: p["l0"])
    return [(p[x], p["kl"]) for p in ps]


def x_at(curve, target):
    """log-log interpolation of x where a curve crosses a KL target (first crossing from the right)."""
    for (x0, y0), (x1, y1) in zip(curve[::-1][1:], curve[::-1]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            t = (np.log(target) - np.log(y0)) / (np.log(y1) - np.log(y0))
            return float(np.exp(np.log(x0) + t * (np.log(x1) - np.log(x0))))
    return None


fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, x, xlabel in ((axes[0], "l0", "pieces switched on per token"),
                      (axes[1], "bits", "bits per token to name them")):
    ax.set_facecolor(SURF)
    for ps, color, label, ls in series:
        c = xy(ps, x)
        xs, ys = zip(*c)
        ax.plot(xs, ys, color=color, lw=2.2, ls=ls, zorder=3, solid_capstyle="round")
        ax.scatter(xs, ys, s=55 if ls == "-" else 30, color=color, edgecolor=SURF, linewidth=1.8, zorder=4)
        if ax is axes[0]:
            ax.plot([], [], color=color, lw=2.2, ls=ls, marker="o", ms=7, mec=SURF, label=label.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK, labelpad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=12)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
axes[0].set_ylabel("difference from the model's outputs (KL)", color=INK, labelpad=10)
axes[0].set_ylim(1e-5, 150)
leg = axes[0].legend(loc="lower left", frameon=False, fontsize=13, labelcolor=INK, handlelength=2.6)
if vpd:
    v = xy(vpd["rounded"], "l0")[-1]
    w = x_at(xy(bases["wsvd"]["points"], "l0"), v[1])
    axes[0].annotate("", xy=(v[0] * 1.08, v[1]), xytext=(w / 1.08, v[1]),
                     arrowprops=dict(arrowstyle="<-", color=INK2, lw=1.2))
    axes[0].annotate(f"{w / v[0]:.0f}× fewer", xy=(np.sqrt(v[0] * w), v[1] * 1.35), color=INK2, fontsize=13, ha="center")
    title = "Per token, VPD switches on far fewer pieces than any fixed basis"
else:
    title = "Per token, how many pieces a fixed basis needs"
if vpd:
    for ax, x in ((axes[0], "l0"), (axes[1], "bits")):
        c = xy(vpd["rounded"], x)
        ax.annotate("VPD", xy=(c[0][0] / 1.15, c[0][1]), color=INK, fontsize=15, ha="right", va="center", fontweight="medium")
fig.suptitle(title, color=INK, fontsize=19, fontweight="bold", x=0.06, ha="left", y=0.98)
fig.tight_layout()
out = Path.home() / f"mpd-data/figures/pertoken_frontier_{MODEL}.png"
fig.savefig(out, facecolor=SURF)
print(out)
