"""One sentence, explained: every VPD subcomponent active on "The princess lost", by layer and token (#2951)."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

d = json.load(open(Path.home() / "mpd-data/figures/data/princess.json"))
INK, INK2, MUTED, SURF, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9", "#f3f2ee"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 14})
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#d3d2cc", ORANGE])
CLIP = 2.0

KIND = {"q_proj": "attention · query", "k_proj": "attention · key", "v_proj": "attention · value",
        "o_proj": "attention · output", "c_fc": "MLP · in", "down_proj": "MLP · out"}
names, T = d["names"], len(d["tokens"])
words = [w.replace("Ġ", "") for w in d["tokens"]]
cells = {(n, t): [] for n in names for t in range(T)}
for a in d["active"]:
    cells[(a["site"], a["t"])].append(a)
PER_ROW, DX, DY, COLW = 22, 0.24, 0.26, 6.4
x0 = {t: 0.6 + t * COLW for t in range(T)}

# rows bottom (layer 1) to top (layer 4); each site row is as tall as its fullest cell needs
y, rowy, layer_span = 0.0, {}, {}
for n in names:
    rows = max(1, max(math.ceil(len(cells[(n, t)]) / PER_ROW) for t in range(T)))
    h = rows * DY + 0.34
    rowy[n] = (y + 0.17, rows)
    L = int(n.split(".")[1])
    lo, hi = layer_span.get(L, (y, y))
    layer_span[L] = (min(lo, y), y + h)
    y += h
    if n.endswith("down_proj"):
        y += 0.35
top = y

fig, ax = plt.subplots(figsize=(16, 14.5), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
pos = {}
for L, (lo, hi) in layer_span.items():
    ax.add_patch(plt.Rectangle((0.2, lo - 0.06), T * COLW + 0.1, hi - lo + 0.12, color=BAND, lw=0, zorder=0))
    ax.text(-4.6, (lo + hi) / 2, f"layer {L + 1}", ha="left", va="center", fontsize=17, color=INK, fontweight="medium")
for n in names:
    yb, rows = rowy[n]
    ax.text(0.05, yb + (rows - 1) * DY / 2, KIND[n.split(".")[-1]], ha="right", va="center", fontsize=11.5, color=INK2)
    for t in range(T):
        cs = sorted(cells[(n, t)], key=lambda a: a["d_gap"])
        for i, a in enumerate(cs):
            px, py = x0[t] + 0.3 + (i % PER_ROW) * DX, yb + (i // PER_ROW) * DY
            pos[(n, t, a["c"])] = (px, py)
            s = -a["d_gap"]
            size = 16 + 150 * min(abs(s), CLIP) / CLIP
            ax.scatter(px, py, s=size, color=cmap(0.5 + 0.5 * np.clip(s, -CLIP, CLIP) / CLIP),
                       edgecolor=SURF, linewidth=1.0, zorder=3 + min(abs(s), CLIP))

# edges into the her->his subcomponent, measured: removing the source alone shrinks 281's activation
tx, ty = pos[("h.3.attn.o_proj", 2, 281)]
for e in d["edges281"]:
    drop = 1 - e["frac_left"]
    if drop < 0.5:
        continue
    sx, sy = pos[(e["site"], e["t"], e["c"])]
    ax.add_patch(FancyArrowPatch((sx, sy), (tx, ty), connectionstyle="arc3,rad=-0.18", arrowstyle="-",
                                 color=INK, lw=0.6 + 2.2 * drop, alpha=0.55, zorder=8, shrinkA=5, shrinkB=7))
ax.scatter(tx, ty, s=330, facecolor="none", edgecolor=INK, linewidth=2.0, zorder=12)

# the subcomponent's story, outside the grid on the right
p, q = d["p_her_his"], d["p_her_his_without_281"]
lx = x0[2] + COLW + 0.2
row_end = max(px for (n, t, c), (px, py) in pos.items() if n == "h.3.attn.o_proj" and t == 2 and py == ty)
ax.annotate("", xy=(row_end + 0.25, ty), xytext=(lx - 0.15, ty), arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0), zorder=2)
ax.text(lx, ty + 0.62, "3.attn.o : 281", fontsize=15, color=INK, fontweight="bold", va="center")
ax.text(lx, ty + 0.12, "the subcomponent that makes it “her”", fontsize=14, color=INK, va="center")
ax.text(lx, ty - 0.5, f"switch it off:\n“her”  {p[0]:.0%} → {q[0]:.0%}\n“his”  {p[1]:.0%} → {q[1]:.0%}",
        fontsize=13, color=INK2, va="top", linespacing=1.45)

# tokens in, prediction out
for t, w in enumerate(words):
    ax.text(x0[t] + 0.3 + (PER_ROW - 1) * DX / 2, -0.75, f"“{w}”", ha="center", va="center", fontsize=21, color=INK)
cx = x0[2] + 0.3 + (PER_ROW - 1) * DX / 2
ax.annotate("", xy=(cx, top + 0.8), xytext=(cx, top + 0.05), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
ax.text(cx, top + 1.15, f"next word: “her”  {p[0]:.0%}", ha="center", va="center", fontsize=17, color=INK, fontweight="medium")

# colour key
kx, ky, kw = x0[0] + 0.3, top + 1.05, 4.6
grad = np.linspace(-1, 1, 200)[None, :]
ax.imshow(grad, extent=(kx, kx + kw, ky - 0.11, ky + 0.11), cmap=cmap, vmin=-1, vmax=1, aspect="auto", zorder=2)
ax.text(kx, ky + 0.42, "dot colour: what switching that subcomponent off does", fontsize=12.5, color=INK, va="center")
ax.text(kx, ky - 0.42, "helps “his”", fontsize=11.5, color=INK2, va="center", ha="left")
ax.text(kx + kw / 2, ky - 0.42, "no effect", fontsize=11.5, color=INK2, va="center", ha="center")
ax.text(kx + kw, ky - 0.42, "helps “her”", fontsize=11.5, color=INK2, va="center", ha="right")
ey = ty - 3.3
ax.plot([lx, lx + 0.9], [ey, ey], color=INK, lw=2.0, alpha=0.55, solid_capstyle="round")
ax.text(lx + 1.1, ey, "subcomponents it needs", fontsize=13, color=INK, va="center")
ax.text(lx, ey - 0.45, "switching one off\nshrinks it by half or more", fontsize=12, color=INK2, va="top", linespacing=1.4)

ax.set_xlim(-4.8, lx + 7.4)
ax.set_ylim(-1.4, top + 1.9)
ax.axis("off")
fig.suptitle("One sentence, explained", color=INK, fontsize=24, fontweight="bold", x=0.035, ha="left", y=0.985)
fig.text(0.035, 0.945, f"each dot is a VPD subcomponent switched on at that word and layer ({len(d['active'])} in all), as VPD's 4-layer model reads three words",
         color=INK2, fontsize=15, ha="left")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
out = Path.home() / "mpd-data/figures/princess_explained.png"
fig.savefig(out, facecolor=SURF)
print(out)
