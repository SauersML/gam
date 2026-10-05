#!/usr/bin/env python3
"""Figure (#2951): per attention head of vpd4l, what an activation readout says against what a
measured crossed intervention says about copying a token seen earlier in the context.

x: two activation readouts: the head's attention, from the repeated token, to the token that followed
   its earlier occurrence (the standard readout of an induction head), and the head's direct logit
   for the copied token (its write through the final norm gain and the unembedding).
y: the share of the copy the head carries: x1 = random tokens with token A at position i followed by
   B and A again at the end, x0 = the same with B replaced; the input effect is log p(B | x1) -
   log p(B | x0); with the head's output map at zero it changes by gamma; share = -gamma / effect.

usage: MPD_MEM_GIB=1 venv python figure_represent_vs_use.py DATA.json OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1]))
heads = data["heads"]
y = [h["share"] for h in heads]
se = [h["share_se"] for h in heads]
readouts = [("attention_to_successor", "attention to the token after the\nearlier occurrence (readout)"),
            ("direct_logit", "the head's direct logit for the copied\ntoken (readout)")]

plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor="white", sharey=True)
for ax, (key, label) in zip(axes, readouts):
    x = [h[key] for h in heads]
    ax.set_facecolor("white")
    ax.errorbar(x, y, yerr=se, fmt="o", color="#2a78d6", ecolor="#9db8dc", markersize=9, elinewidth=1.5, capsize=0)
    ax.axhline(0, color="#b5b4ae", linewidth=1)
    # Direct labels: the heads that carry most of the copy, and the heads this readout ranks highest.
    carry = sorted(range(len(heads)), key=lambda i: -y[i])[:4]
    read = sorted(range(len(heads)), key=lambda i: -x[i])[:3]
    span = max(x) - min(x)
    crowded = sorted((i for i in dict.fromkeys(carry + read) if x[i] - min(x) < 0.1 * span), key=lambda i: -y[i])
    for i in dict.fromkeys(carry + read):
        h = heads[i]
        # Labels of points crowded near the left edge alternate sides so none overlaps.
        side = crowded.index(i) % 2 if i in crowded else 0
        ax.annotate(f"L{h['layer']} H{h['head']}", (x[i], y[i]), xytext=(-12 if side else 10, -6 if side else 2), textcoords="offset points",
                    fontsize=15, color="#0b0b0b", ha="right" if side else "left")
    ax.set_xlabel(label)
axes[0].set_ylabel("share of the copy the head carries\n(measured by crossed intervention)")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150, facecolor="white")
print(sys.argv[2])
