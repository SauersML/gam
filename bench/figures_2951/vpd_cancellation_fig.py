"""Why removing more of VPD's subcomponents makes its error smaller (#2951).

From the battery's held-out cancellation test (examples/mpd_battery_2951 vpd, its `cancellation`, in
bits; drawn in nats): KL(M || E) per token, where E is VPD's published decomposition (goodfire/spd/runs/s-55ea3f9b) with, per
token, every subcomponent whose causal importance is 0 removed (VPD's own CI > 0 cutoff, the one
behind its count of 205 active per token) in the named layers, and the model's own weights elsewhere.
Bars are means over the earlier layer l = 0, 1, 2 (each of the three shows the same ordering). Let U be the output of
layer 3's importance-0 subcomponents (what removing from layer 3 deletes), and dU the change in U
when removal at layer l changes layer 3's input. The last bar is layer l only with dU subtracted:
most of layer l's error is dU, produced in layer 3 by subcomponents VPD scores as unimportant, which
is why removing from layer 3 as well (deleting U, dU with it) lowers it.

    python bench/figures_2951/vpd_cancellation_fig.py CANCELLATION.json OUT.png
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cancellation, out = sys.argv[1:3]
rows = json.load(open(cancellation))["cancellation"]["rounded"]
kl = lambda name: rows[f"kl_{name}"]["mean"] * math.log(2)

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, GRAY, ORANGE = "#2a78d6", "#8c8c89", "#eb6834"
plt.rcParams.update({
    "mathtext.fontset": "custom", "mathtext.it": "Helvetica Neue:italic",
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "hatch.color": SURFACE, "hatch.linewidth": 2.5,
})
fig, ax = plt.subplots(figsize=(14, 6.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bars = [("one earlier layer (0, 1 or 2)", [kl(l) for l in range(3)], BLUE, None),
        ("layer 3", [kl(3)], GRAY, None),
        ("one earlier layer and layer 3", [kl(f"{l}3") for l in range(3)], ORANGE, None),
        ("one earlier layer, then subtract the\nchange this causes in the output of\nlayer 3's importance-0 subcomponents",
         [kl(f"{l}_without_I") for l in range(3)], BLUE, "//")]
for i, (name, vals, color, hatch) in enumerate(bars):
    mean = sum(vals) / len(vals)
    ax.barh(i, mean, 0.66, color=color, edgecolor=SURFACE, linewidth=2, hatch=hatch)
    ax.annotate(f"{mean:.2f}", (mean, i), xytext=(10, 0), textcoords="offset points", va="center",
                fontsize=19, color=INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=18)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0, labelcolor=INK)
ax.set_xlim(0, 0.68)
ax.set_xlabel("KL from the model's next-token predictions (nats per token; 0 = identical)")
ax.set_ylabel("VPD's importance-0\nsubcomponents removed from", fontsize=18)
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
