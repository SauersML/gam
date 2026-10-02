"""Whole-model size against faithfulness: structure-blind compression vs VPD (#2951).

usage: mpd_wholemodel_frontier_fig_2951.py {vpd4l|pythia70m}   (prints bits at KL 0.01 / 0.1 / 0.35 per method)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

MODEL = sys.argv[1] if len(sys.argv) > 1 else "vpd4l"
F = Path.home() / "mpd-data/frontier"
wm = json.load(open(F / f"wholemodel_{MODEL}.json"))

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 15})


def pareto(pts):
    """Lower-left envelope of (bits, kl) points."""
    out = []
    for b, k in sorted(pts):
        if not out or k < out[-1][1]:
            out.append((b, k))
    return out


def bits_at(env, target):
    for (b0, k0), (b1, k1) in zip(env, env[1:]):
        if k0 >= target >= k1:
            t = (np.log(target) - np.log(k0)) / (np.log(k1) - np.log(k0))
            return float(np.exp(np.log(b0) + t * (np.log(b1) - np.log(b0))))
    return None


series = [("dense", BLUE, "the model itself, rounded"),
          ("svd", YELLOW, "each matrix's SVD, rounded"),
          ("prune", MAGENTA, "smallest weights removed, rounded"),
          ("combo", AQUA, "SVD plus the largest leftovers, rounded")]
curves = {m: pareto([(p["bits"], p["kl"]) for p in wm[m] if p["kl"] > 0]) for m, _, _ in series if m in wm}
if MODEL == "vpd4l":
    rows = dict(json.load(open(Path.home() / "mpd-data/vpd/scoreboard_compact.json")))
    vb = ["2", "3", "4", "5", "6", "8", "12", "fp32"]
    vpd_uv = [(rows[f"S1 VPD b={b}: bits U,V | Gamma | total"][0], rows[f"S1 VPD b={b}: KL ci | rounded | PGD-20 (no delta)"][1]) for b in vb]
    vpd_all = [(rows[f"S1 VPD b={b}: bits U,V | Gamma | total"][2], rows[f"S1 VPD b={b}: KL ci | rounded | PGD-20 (no delta)"][1]) for b in vb]

table = {}
for m, env in curves.items():
    table[m] = {t: bits_at(env, t) for t in (0.01, 0.1, 0.35)}
    print(m, {t: (f"{v:.3g}" if v else None) for t, v in table[m].items()})
json.dump(table, open(F / f"wholemodel_{MODEL}_table.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for m, color, label in series:
    if m not in curves:
        continue
    xs, ys = zip(*curves[m])
    wide = m == "combo"  # drawn under the others: where it coincides with plain rounding, both stay visible
    ax.plot(xs, ys, color=color, lw=6 if wide else 2.2, alpha=0.55 if wide else 1, zorder=2 if wide else (6 if m == "dense" else 4),
            solid_capstyle="round", label=label, marker=None if wide else "o", ms=7, mec=SURF, mew=1.8)
if MODEL == "vpd4l":
    for pts, ls, label in ((vpd_uv, "-", "VPD components only"), (vpd_all, (0, (4, 3)), "VPD components + the network that switches them on")):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ORANGE, lw=2.2, ls=ls, marker="o", ms=7, mec=SURF, mew=1.8, zorder=3, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
ax.set_xlabel("size of the description (bits)", color=INK, labelpad=10)
ax.set_ylabel("difference from the model's outputs (KL)", color=INK, labelpad=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=12)
ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
ax.legend(loc="lower left", frameon=False, fontsize=13, labelcolor=INK, handlelength=2.6)
ax.set_title("Plain compression makes the model far smaller than any decomposition so far" if MODEL == "vpd4l"
             else "Plain compression of Pythia-70m", color=INK, fontsize=19, fontweight="bold", loc="left", pad=16)
fig.tight_layout()
out = Path.home() / f"mpd-data/figures/wholemodel_frontier_{MODEL}.png"
fig.savefig(out, facecolor=SURF)
print(out)
