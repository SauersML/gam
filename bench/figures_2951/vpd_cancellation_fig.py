"""Removing more of VPD's "unimportant" subcomponents makes its error smaller (#2951).

From compare's held-out battery (rows 1024-1055 of vpd4l_clean4096): KL(M || E) per token in nats,
where E is the model with the subcomponents VPD marks unimportant for each token removed (causal-
importance masks) in the named layers, and the model's own weights everywhere else.

    python bench/figures_2951/vpd_cancellation_fig.py BATTERY.json OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

battery, out = sys.argv[1:3]
rows = json.load(open(battery))["held_out"]["rows"]
kl = lambda name: rows[f"ci/{name}"]["kl_nats"]["mean"]

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, GRAY, ORANGE = "#2a78d6", "#a9a9a6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 3, figsize=(18, 7.4), facecolor=SURFACE, sharey=True,
                         gridspec_kw={"wspace": 0.12})
for ax, l in zip(axes, [0, 1, 2]):
    ax.set_facecolor(SURFACE)
    bars = [(f"layer {l}\nonly", kl(f"layers_{l}"), BLUE),
            ("layer 3\nonly", kl("layers_3"), GRAY),
            (f"layers {l}\nand 3", kl(f"layers_{l}3"), ORANGE)]
    for i, (name, value, color) in enumerate(bars):
        ax.bar(i, value, 0.7, color=color, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=19, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bars], fontsize=17)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.8)
axes[0].set_ylabel("change in the model's next-token\npredictions (KL, nats per token)")
fig.suptitle("Removing VPD's \"unimportant\" pieces from layer 3 as well makes the change smaller",
             x=0.08, ha="left", fontsize=23, y=1.0)
fig.supxlabel("VPD's \"unimportant\" pieces removed from", fontsize=19, y=-0.04)
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
