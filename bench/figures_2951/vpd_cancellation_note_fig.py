"""Figure for the one-page note on VPD's layer cancellation (#2951): KL divergence from the model when an
earlier layer (0, 1 or 2) is replaced alone or with layer 3, before and after subtracting or adding
back layer 3's response (the change in the output of its zero-importance subcomponents), averaged over
the earlier layer. Rounded masks (CI > 0), held-out rows 1024-1056.

    python bench/figures_2951/vpd_cancellation_note_fig.py CANCELLATION.json OUT.pdf
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cancellation, out = sys.argv[1:3]
c = json.load(open(cancellation))["cancellation"]["rounded"]
mean_l = lambda key: sum(c[key.format(l)]["mean"] for l in range(3)) / 3

INK, MUTED, BLUE, ORANGE, GRAY = "#1f1f1e", "#6b6b68", "#2a78d6", "#eb6834", "#b4b4b1"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(5.4, 2.45))
bars = [("earlier\nlayer", mean_l("kl_{}"), GRAY),
        ("earlier layer,\nminus layer-3\nresponse", mean_l("kl_{}_without_I"), BLUE),
        ("earlier layer\nand layer 3", mean_l("kl_{}3"), GRAY),
        ("earlier layer\nand layer 3, plus\nlayer-3 response", mean_l("kl_{}3_no_interaction"), ORANGE)]
xs = [0, 1.05, 2.6, 3.65]
for x, (name, v, color) in zip(xs, bars):
    ax.bar(x, v, 0.8, color=color, edgecolor="white", linewidth=0.8)
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
for x0, x1, v in [(xs[0], xs[1], bars[0][1]), (xs[2], xs[3], bars[2][1])]:
    ax.plot([x0 - 0.4, x1 + 0.4], [v, v], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.set_xticks(xs)
ax.set_xticklabels([n for n, _, _ in bars], fontsize=11, linespacing=1.1)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.12)
ax.set_ylabel("KL divergence (nats/token)")
ax.set_xlabel("layers replaced", labelpad=6)
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(out)
