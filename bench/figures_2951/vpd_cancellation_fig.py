"""Why VPD's layer errors cancel (#2951), from compare's held-out battery and cancellation test.

Left: KL(M || VPD) per token (nats, held-out rows 1024-1055, causal-importance masks) when VPD replaces
layer l alone, and layers l and 3 together. Right: what layer 3's dropped part does when layer l is
masked upstream: per token, the cosine between the change in that dropped part and layer l's own
deviation at the final residual; dots are medians, bars run from the 1st to the 10th percentile up to
the median (90% of tokens lie to the right of each bar's inner end).

    python bench/figures_2951/vpd_cancellation_fig.py BATTERY.json CANCELLATION.json OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

battery, cancellation, out = sys.argv[1:4]
rows = json.load(open(battery))["held_out"]["rows"]
canc = json.load(open(cancellation))["cancellation"]
kl = lambda name: rows[f"ci/{name}"]["kl_nats"]["mean"]

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7.6), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.35})

# Left: layer l alone against layers l and 3.
layers = [0, 1, 2]
alone = [kl(f"layers_{l}") for l in layers]
with3 = [kl(f"layers_{l}3") for l in layers]
x = np.arange(len(layers))
w = 0.36
left.bar(x - w / 2 - 0.02, alone, w, color=BLUE, edgecolor=SURFACE, linewidth=2)
left.bar(x + w / 2 + 0.02, with3, w, color=ORANGE, edgecolor=SURFACE, linewidth=2)
for xi, a, b in zip(x, alone, with3):
    left.annotate(f"−{100 * (a - b) / a:.0f}%", (xi + w / 2 + 0.02, b), xytext=(0, 8),
                  textcoords="offset points", ha="center", fontsize=17, color=INK)
left.set_xticks(x)
left.set_xticklabels([f"layer {l}" for l in layers])
left.tick_params(axis="x", length=0)
left.set_ylabel("KL to the model (nats per token)")
left.set_ylim(0, 0.9)
from matplotlib.patches import Patch
left.legend(handles=[Patch(color=BLUE, label="VPD replaces only this layer"),
                     Patch(color=ORANGE, label="VPD replaces this layer and layer 3")],
            loc="upper right", frameon=False, fontsize=16, handlelength=1.2)
left.set_title("Replacing layer 3 too makes VPD's error smaller", loc="left", pad=16, fontsize=21)

# Right: direction of layer 3's dropped part's change, relative to the upstream deviation.
ys = np.arange(len(layers))[::-1]
for yi, l in zip(ys, layers):
    c = canc[f"pair_{l}_3"]["cos_change_minus_D"]
    med, p10, p1 = -c["q50"], -c["q90"], -c["q99"]
    right.plot([p10, med], [yi, yi], color=BLUE, lw=10, solid_capstyle="round")
    right.plot([med], [yi], marker="o", ms=15, color=INK, markeredgecolor=SURFACE, markeredgewidth=2.5)
    right.annotate(f"{med:.2f}", (med, yi), xytext=(14, 0), textcoords="offset points", va="center",
                   fontsize=17)
right.axvline(0, color="#d9d9d6", lw=1.5, zorder=0)
right.set_yticks(ys)
right.set_yticklabels([f"layer {l} replaced" for l in layers])
right.tick_params(axis="y", length=0)
right.set_xlim(-0.15, 1.0)
right.set_xlabel("cosine with the upstream error (dot: median; bar: from 10th percentile)")
right.set_ylim(-0.6, 2.6)
right.set_title("because the part of layer 3 that VPD drops\ncarries the upstream error forward", loc="left", pad=16, fontsize=21)

fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
