"""Why removing more of VPD's subcomponents makes its error smaller (#2951).

From compare's held-out cancellation run (vpd_battery.py OUT.json cancellation 1024:1056): KL(M || E)
per token in nats, where E is VPD's published decomposition (goodfire/spd/runs/s-55ea3f9b) with, per
token, every subcomponent whose causal importance is 0 removed (VPD's own CI > 0 cutoff, the one
behind its count of 205 active per token) in the named layers, and the model's own weights elsewhere.
Let U be the output of layer 3's importance-0 subcomponents (what removing from layer 3 deletes),
and dU the change in U when removal at layer l changes layer 3's input. The hatched bar is layer l
only with dU subtracted: most of layer l's error is dU, produced in layer 3 by subcomponents VPD
scores as unimportant, which is why removing from layer 3 as well (deleting U, dU with it) lowers it.

    python bench/figures_2951/vpd_cancellation_fig.py CANCELLATION.json OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cancellation, out = sys.argv[1:3]
rows = json.load(open(cancellation))["cancellation"]["rounded"]
kl = lambda name: rows[f"kl_{name}"]["mean"]

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, GRAY, ORANGE = "#2a78d6", "#a9a9a6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.color": SURFACE, "hatch.linewidth": 2.5,
})
fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor=SURFACE, sharey=True,
                         gridspec_kw={"wspace": 0.1})
for ax, l in zip(axes, [0, 1, 2]):
    ax.set_facecolor(SURFACE)
    bars = [(f"layer {l}\nonly", kl(l), BLUE, None),
            (f"layer {l} only,\nminus the change\nit causes in the\noutput of layer 3's\nimportance-0\nsubcomponents",
             kl(f"{l}_without_I"), BLUE, "//"),
            ("layer 3\nonly", kl(3), GRAY, None),
            (f"layers {l}\nand 3", kl(f"{l}3"), ORANGE, None)]
    for i, (name, value, color, hatch) in enumerate(bars):
        ax.bar(i, value, 0.72, color=color, edgecolor=SURFACE, linewidth=2, hatch=hatch)
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=19, color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.8)
axes[0].set_ylabel("KL from the model's next-token predictions\n(nats per token; 0 = identical)")
fig.supxlabel("VPD's subcomponents with causal importance 0 removed from", fontsize=19, y=-0.15)
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
