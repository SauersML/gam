"""Causally important units per target token: the model's own neurons and heads against VPD's
subcomponents (#2951).

Reads the battery's importance run (examples/mpd_battery_2951 importance): per target token t on
held-out rows 1024-1055 of vpd4l_clean4096, a unit counts when its RelP attribution of the
predicted token's centred logit exceeds a fraction tau of that logit; VPD's own count (subcomponents
with a positive causal importance) is the dashed line.

    python bench/figures_2951/importance_counts_fig.py IMPORTANCE.json OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out = sys.argv[1:3]
run = json.load(open(path))["importance"]["model_and_vpd"]

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, ax = plt.subplots(figsize=(11, 7), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for kind, color in [("neurons_and_heads", BLUE), ("vpd", AQUA)]:
    taus = [x["tau"] for x in run[kind]]
    means = [x["per_token"]["mean"] for x in run[kind]]
    ax.plot(taus, means, color=color, lw=3.5, marker="o", ms=11, markeredgecolor=SURFACE, markeredgewidth=2)
ax.text(0.0095, 12, "the model's own neurons and heads", color=BLUE, fontsize=18, va="center")
ax.text(0.034, 95, "VPD's subcomponents", color=AQUA, fontsize=18, va="center")
own = run["vpd_positive_importance"]["mean"]
ax.axhline(own, color=ORANGE, lw=2.5, ls=(0, (5, 3)))
ax.annotate("VPD's own count of active subcomponents", (0.1, own), xytext=(0, 10), textcoords="offset points", ha="right", color=ORANGE, fontsize=17)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.01, 0.03, 0.1])
ax.set_xticklabels(["1%", "3%", "10%"])
ax.set_xlim(0.0085, 0.12)
ax.set_xlabel("share of the prediction a unit must carry")
ax.set_ylabel("causally important units per token")
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
