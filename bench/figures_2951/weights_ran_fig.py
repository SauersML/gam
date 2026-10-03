"""Weight numbers that ran per word on VPD 4L: the full model, VPD's choice of subcomponents, and ours
(#2951). Data from weights_ran_data.py."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

d = json.load(open(Path.home() / "mpd-data/figures/data/weights_ran.json"))
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#c3c2b7"
GRAY, ORANGE, BLACK = "#b9b8b1", "#eb6834", "#0b0b0b"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
seq = LinearSegmentedColormap.from_list("seq", ["#f3f2ee", "#9ec5f4", "#2a78d6", "#104281"])
NAMES = {"vpd": ("VPD's choice of subcomponents", ORANGE), "ours": ("our choice (Step A)", BLACK)}
dense = d["dense_per_word"]
rows = [("the full model", GRAY, np.array([dense]))]
rows += [(NAMES.get(k, (k, BLACK))[0], NAMES.get(k, (k, BLACK))[1], np.array(v["per_word"])) for k, v in d["sets"].items()]

fig = plt.figure(figsize=(17, 10), dpi=200)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.55, left=0.2, right=0.96, top=0.86, bottom=0.14)

# --- top: weight numbers per word, all 24 matrices
ax = fig.add_subplot(gs[0])
ax.set_facecolor(SURF)
for i, (label, color, x) in enumerate(rows):
    y = len(rows) - 1 - i
    m = x.mean()
    ax.barh(y, m / 1e6, height=0.46, color=color, zorder=3)
    if len(x) > 1:
        lo, hi = np.percentile(x, [5, 95])
        ax.plot([lo / 1e6, hi / 1e6], [y, y], color=INK, lw=1.4, zorder=4, solid_capstyle="butt")
        for v in (lo, hi):
            ax.plot([v / 1e6] * 2, [y - 0.1, y + 0.1], color=INK, lw=1.4, zorder=4)
        txt = f"{m / 1e6:.2f} million on average  ({m / dense:.1%} of the model; middle 90% of words: {lo / 1e6:.2f}–{hi / 1e6:.2f})"
        ax.annotate(txt, (hi / 1e6, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=13, color=INK)
    else:
        ax.annotate(f"{m / 1e6:.1f} million, every word", (m / 1e6, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=13, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=14.5, color=INK)
ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", colors=INK2, labelsize=12.5)
ax.set_xlim(0, dense / 1e6 * 1.32)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("weight numbers that ran on each word (millions)", color=INK, labelpad=8)

# --- bottom: share of each matrix's weights that ran per word, VPD (and ours, side by side)
kinds = [("q", "attention\nquery"), ("k", "attention\nkey"), ("v", "attention\nvalue"), ("o", "attention\noutput"),
         ("c_fc", "MLP\nin"), ("down_proj", "MLP\nout")]
names = list(d["sets"])
sub = gs[1].subgridspec(1, len(names), wspace=0.08)
for j, name in enumerate(names):
    bx = fig.add_subplot(sub[j])
    s = d["sets"][name]["per_site_weights"]
    M = np.array([[s[f"blocks.{L}.{k}"] / d["dense_per_site"][f"blocks.{L}.{k}"] for k, _ in kinds] for L in range(4)])
    bx.imshow(M, cmap=seq, vmin=0, vmax=0.075, aspect="auto")
    for L in range(4):
        for c in range(len(kinds)):
            v = M[L, c]
            bx.text(c, L, f"{v:.1%}", ha="center", va="center", fontsize=12.5,
                    color="white" if v > 0.04 else INK)
    bx.set_xticks(range(len(kinds)))
    bx.set_xticklabels([k for _, k in kinds], fontsize=12, color=INK)
    bx.set_yticks(range(4))
    bx.set_yticklabels([f"layer {L + 1}" for L in range(4)] if j == 0 else [], fontsize=13.5, color=INK)
    bx.tick_params(length=0)
    for side in bx.spines.values():
        side.set_visible(False)
    bx.set_title(f"{NAMES.get(name, (name, None))[0]}: share of each matrix's weights that ran, per word",
                 color=INK, fontsize=14.5, loc="left", pad=10)
rem = d["remainder"]
qk = [rem[f"blocks.{L}.{k}"] for L in range(4) for k in ("q", "k")]
fig.text(0.2, 0.012, f"VPD's query and key subcomponents don't add up to the whole matrix: what is left over is {min(qk):.0%}–{max(qk):.0%} "
         "the size of each query/key matrix,\nand that remainder runs in neither VPD's sets nor ours "
         "(for every other matrix the leftover is under 0.1%)", fontsize=12.5, color=INK2, ha="left", linespacing=1.4)

ratio = dense / rows[1][2].mean()
fig.suptitle(f"VPD runs about 1/{ratio:.0f} of the model's weights on each word", color=INK, fontsize=21,
             fontweight="bold", x=0.02, ha="left", y=0.975)
fig.text(0.02, 0.915, "VPD's 4-layer model on 65,536 words of web text; a subcomponent of an m × n matrix counts m + n numbers, "
         "the full model counts every weight of its 24 attention and MLP matrices", color=INK2, fontsize=14, ha="left")
out = Path.home() / "mpd-data/figures/weights_ran_per_word.png"
fig.savefig(out, facecolor=SURF)
print(out)
