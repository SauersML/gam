"""Rank-k gated subcomponents on the mod-31 adder (#2951): the code per word of each explanation,
and the multi-direction blocks the code chose, by frequency. Data from
`mpd_blocks_2951 modadd` (crates/gam-mpd/examples).

usage: blocks_modadd_fig.py RESULT.json OUT.png
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

result = json.load(open(sys.argv[1]))
out = sys.argv[2]
INK, MUTED = "#0b0b0b", "#6b6a66"
KL_COLOR, RAN_COLOR = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial"], "font.size": 17, "axes.spines.top": False, "axes.spines.right": False})

points = {p["point"]: p for p in result["points"]}
order = [
    ("dense", "whole matrices,\nevery word"),
    ("all on", "every rank-one\nsubcomponent on"),
    ("all off", "nothing on"),
    ("rank one", "rank-one\nsubcomponents,\nselected"),
    ("blocks' columns as rank-one subcomponents", "the blocks' columns\nas rank-one\nsubcomponents"),
    ("blocks", "rank-k blocks,\nchosen by the code"),
]
order = [(k, label) for k, label in order if k in points]

fig = plt.figure(figsize=(18, 11), dpi=200)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.32, left=0.2, right=0.97, top=0.86, bottom=0.1)

ax = fig.add_subplot(gs[0])
y = np.arange(len(order))[::-1]
for yi, (key, label) in zip(y, order):
    p = points[key]
    ran = p["described_bits"] / p["rows"]
    kl = p["kl_bits"] / p["rows"]
    ax.barh(yi, ran, color=RAN_COLOR, height=0.62)
    ax.barh(yi, kl, left=ran, color=KL_COLOR, height=0.62)
    ax.text(ran + kl, yi, f"  {ran + kl:,.0f}   KL {p['kl']:.2f}, {p['active_rank_one_equivalents_per_word']:.1f} directions on",
            va="center", fontsize=14, color=INK)
ax.set_yticks(y, [label for _, label in order], fontsize=15)
ax.set_xlabel("bits per word")
ax.set_title("The code of each explanation", loc="left", fontsize=20, pad=14)
ax.legend(handles=[Patch(color=RAN_COLOR, label="the weights that ran"), Patch(color=KL_COLOR, label="n·KL / ln 2")],
          frameon=False, loc="lower right", fontsize=14)
xmax = max(points[k]["bits_per_word"] for k, _ in order)
ax.set_xlim(0, xmax * 1.9)

# The multi-direction blocks the code chose, by frequency.
blocks = [b for b in result["blocks"] if b["rank"] >= 2 and b["firing"] > 0]
ax2 = fig.add_subplot(gs[1])
if blocks:
    spectra, labels = [], []
    for b in blocks:
        s = b.get("output_spectrum") or b.get("input_spectrum")
        side = "writes" if b.get("output_spectrum") else "reads"
        if s is None:
            continue
        s = np.array(s[1:])
        spectra.append(s / max(s.sum(), 1e-300))
        site = b["site"].split(".")[-1]
        labels.append(f"{site}, rank {b['rank']}, {side}\n{b['bits']:.0f} bits vs {b['columns_as_rank_one_bits']:.0f} as columns, on {100 * b['firing']:.0f}%")
    if spectra:
        image = np.array(spectra)
        ax2.imshow(image, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax2.set_yticks(range(len(labels)), labels, fontsize=12)
        ax2.set_xticks(range(image.shape[1]), [str(f) for f in range(1, image.shape[1] + 1)], fontsize=12)
        ax2.set_xlabel("frequency (share of the block's energy)")
        for spine in ax2.spines.values():
            spine.set_visible(False)
    else:
        ax2.axis("off")
else:
    ax2.axis("off")
ax2.set_title("Blocks of rank 2 or more", loc="left", fontsize=20, pad=14)
fig.suptitle(f"Mod-31 addition: one gate per rank-k block (n = {result['observations']:g})", x=0.02, ha="left", fontsize=24)
fig.savefig(out, facecolor="white")
print(out)
