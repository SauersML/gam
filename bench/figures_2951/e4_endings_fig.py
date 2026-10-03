"""E4 (#2951): the change in an answer ending's log-probability against how many of its tokens finish a split word,
for the VPD edit and the equal-success LoRA, on HellaSwag and PIQA. Reads e4_endings.json (e4_benchmarks_data.py
endings).

usage: MPD_MEM_GIB=1 e4_endings_fig.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

d = json.load(open(Path.home() / "mpd-data/frontier/e4_side/bench/e4_endings.json"))
OUT = Path.home() / "mpd-data/figures/e4_endings.png"
INK, INK2, SURF, AXIS = "#0b0b0b", "#52514e", "#ffffff", "#c3c2b7"
ORANGE, BLUE = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 24, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(24, 9.5), dpi=150, gridspec_kw={"wspace": 0.18})
fig.patch.set_facecolor(SURF)
for ax, (task, title) in zip(axes, (("hellaswag", "HellaSwag"), ("piqa", "PIQA"))):
    for nm, col, dx in (("vpd_match_lora282_lam10", ORANGE, -0.08), ("lora282_lam10", BLUE, 0.08)):
        b = d[task]["models"][nm]["by_pieces"]
        x = np.arange(len(b)) + dx
        m = [e["mean"] for e in b]
        ax.plot(x, m, color=col, lw=2.8, zorder=3)
        ax.vlines(x, [e["ci"][0] for e in b], [e["ci"][1] for e in b], color=col, lw=2.6, zorder=3)
        ax.scatter(x, m, s=110, color=col, edgecolor=SURF, linewidth=1.8, zorder=4)
    labs = [str(e["lo"]) if e["lo"] == e["hi"] else (f"{e['lo']}+" if e["hi"] >= 99 else f"{e['lo']}–{e['hi']}")
            for e in d[task]["models"]["lora282_lam10"]["by_pieces"]]
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.axhline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_xlabel("tokens in the answer that finish a split word")
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("change in the answer's log-probability (nats)")
fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=2.8, ms=11, mec=SURF, label="VPD subcomponent edit"),
                    Line2D([], [], color=BLUE, marker="o", lw=2.8, ms=11, mec=SURF, label="LoRA, 282 examples")],
           loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.005, 0.9))
fig.suptitle("The VPD edit's damage to an answer grows with its split words; LoRA's does not", x=0.01, ha="left",
             y=0.995, fontsize=32, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.15)
fig.savefig(OUT, facecolor=SURF)
print(OUT)
