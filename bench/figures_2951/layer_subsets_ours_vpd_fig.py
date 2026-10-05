"""KL to the model against the number of layers replaced, our library explanation beside VPD (#2951).

Reads the battery's two runs on held-out rows 1024-1055 of vpd4l_clean4096 (examples/mpd_battery_2951,
`vpd` and `library`): per size k the mean over every subset of k layers replaced (CI masks for VPD),
in bits, drawn in nats.

    python bench/figures_2951/layer_subsets_ours_vpd_fig.py VPD.json OURS.json OUT.png
"""
import json
import math
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

vpd_path, ours_path, out = sys.argv[1:4]
vpd = json.load(open(vpd_path))["protocols"]["masks"]["ci"]
ours = json.load(open(ours_path))["protocols"]
L = 4


def curve(rows, stat):
    """Per size k, the mean over every subset of k layers of the KL statistic, in nats."""
    values = []
    for k in range(1, L + 1):
        keys = ["layers_" + "".join(map(str, s)) for s in combinations(range(L), k)]
        values.append(sum(rows[key]["kl_bits"][stat] for key in keys) / len(keys) * math.log(2))
    return values


plt.rcParams.update({"font.size": 20, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), facecolor="white")
for ax, (stat, title) in zip(axes, [("mean", "Mean over tokens"), ("q99", "99th percentile over tokens")]):
    ax.set_facecolor("white")
    for values, label, color in ((curve(vpd, stat), "VPD", "#b2182b"), (curve(ours, stat), "ours", "#2166ac")):
        ax.plot(range(1, L + 1), values, color=color, lw=3.5, marker="o", ms=11)
        ax.annotate(label, (L, values[-1]), xytext=(12, 0), textcoords="offset points", va="center", color=color, fontsize=20)
    ax.set_xticks(range(1, L + 1))
    ax.set_xticklabels(["1", "2", "3", "all 4"])
    ax.set_xlim(0.7, L + 0.8)
    ax.set_ylim(0, None)
    ax.set_xlabel("layers replaced")
    ax.set_title(title, pad=14)
axes[0].set_ylabel("KL to the model (nats per token)")
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor="white")
print(out)
