"""VPD's error falls as more of the model is replaced (#2951).

Reads compare's battery (bench/vpd_2951/vpd_battery.py output): KL(M || E) per token on held-out
rows 1024-1055 of vpd4l_clean4096, where E is M with VPD's subcomponents (causal-importance masks,
VPD's intended setting) substituted in a subset of the four layers and M's own weights elsewhere.

    python bench/figures_2951/vpd_layer_cancellation_fig.py BATTERY.json OUT.png
"""
import json
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

battery, out = sys.argv[1], sys.argv[2]
rows = json.load(open(battery))["held_out"]["rows"]
layers = range(4)
plt.rcParams.update({"font.size": 20, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), facecolor="white")
singles_of = {}
for ax, (stat, title) in zip(axes, [("mean", "Mean over tokens"), ("q99", "99th percentile over tokens")]):
    ax.set_facecolor("white")
    means = []
    for k in range(1, 5):
        subsets = list(combinations(layers, k))
        values = [rows["ci/error_propagating" if k == 4 else "ci/layers_" + "".join(map(str, s))]["kl_nats"][stat]
                  for s in subsets]
        means.append(sum(values) / len(values))
        ax.scatter([k] * len(values), values, s=90, color="#9aa7b8", zorder=2)
        if k == 1:
            singles = singles_of[stat] = sorted(zip(values, (s[0] for s in subsets)))
    # Single-layer labels, spread apart where their values nearly coincide.
    top = max(max(v for v, _ in singles_of[stat]), max(means)) * 1.08
    gap, placed = 0.055 * top, []
    for v, layer in singles_of[stat]:
        y = max(v, placed[-1] + gap) if placed else v
        placed.append(y)
        ax.annotate(f"layer {layer}", (1, v), xytext=(1.08, y), textcoords="data",
                    va="center", fontsize=17, color="#4a5566")
    ax.plot(range(1, 5), means, color="#b2182b", lw=3.5, marker="o", ms=11, zorder=3)
    ax.annotate("average", (1, means[0]), xytext=(-14, 0), textcoords="offset points",
                ha="right", va="center", color="#b2182b", fontsize=18)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["1", "2", "3", "all 4"])
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(0, None)
    ax.set_xlabel("layers replaced by VPD's subcomponents")
    ax.set_title(title, pad=14)
axes[0].set_ylabel("KL to the model (nats per token)")
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor="white")
print(out)
