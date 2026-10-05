"""Why removing more of VPD's subcomponents makes its error smaller (#2951).

From compare's held-out cancellation run (vpd_battery.py OUT.json cancellation 1024:1056): KL(M || E)
per token in nats, where E is VPD's published decomposition (goodfire/spd/runs/s-55ea3f9b) with, per
token, every subcomponent whose causal importance is 0 removed (VPD's own CI > 0 cutoff, the one
behind its count of 205 active per token) in the named layers, and the model's own weights elsewhere.
The last bar of each panel removes from layers l and 3, but holds what is removed from layer 3 at
its value on the model's own activations, so layer 3's removed subcomponents no longer respond to
the change that removal at layer l makes. That bar rising above the others shows the smaller error
of the third bar comes from those subcomponents' response.

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
            ("layer 3\nonly", kl(3), GRAY, None),
            (f"layers {l}\nand 3", kl(f"{l}3"), ORANGE, None),
            (f"layers {l} and 3,\nlayer 3's removal\ncomputed from\nthe model's own\nactivations",
             kl(f"{l}3_no_interaction"), ORANGE, "//")]
    for i, (name, value, color, hatch) in enumerate(bars):
        ax.bar(i, value, 0.72, color=color, edgecolor=SURFACE, linewidth=2, hatch=hatch)
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=19, color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.2)
axes[0].set_ylabel("KL from the model's next-token predictions\n(nats per token; 0 = identical)")
fig.supxlabel("VPD's subcomponents with causal importance 0 removed from", fontsize=19, y=-0.13)
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
