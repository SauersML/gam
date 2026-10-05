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
BLUE, GRAY, ORANGE = "#2a78d6", "#8c8c89", "#eb6834"
plt.rcParams.update({
    "mathtext.fontset": "custom", "mathtext.it": "Helvetica Neue:italic",
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.color": SURFACE, "hatch.linewidth": 2.5,
})
fig, ax = plt.subplots(figsize=(13, 9), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
series = [("layer $l$ only", lambda l: kl(l), BLUE, None),
          ("layer $l$ only, minus the change this causes in the\noutput of layer 3's importance-0 subcomponents",
           lambda l: kl(f"{l}_without_I"), BLUE, "//"),
          ("layers $l$ and 3", lambda l: kl(f"{l}3"), ORANGE, None)]
width = 0.26
for j, (name, value, color, hatch) in enumerate(series):
    xs = [l + (j - 1) * width for l in range(3)]
    vals = [value(l) for l in range(3)]
    ax.bar(xs, vals, width, color=color, edgecolor=SURFACE, linewidth=2, hatch=hatch, label=name)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=17, color=INK,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none"))
ax.axhline(kl(3), color=GRAY, lw=2.5, ls=(0, (6, 4)), zorder=0, label=f"layer 3 only ({kl(3):.2f})")
ax.set_xticks(range(3))
ax.set_xticklabels(["$l$ = 0", "$l$ = 1", "$l$ = 2"])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0, 0.75)
ax.set_xlabel("earlier layer $l$")
ax.set_ylabel("KL from the model's next-token predictions\n(nats per token; 0 = identical)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], title="VPD's subcomponents with causal importance 0 removed from:", title_fontsize=18,
          fontsize=17, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), alignment="left")
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
