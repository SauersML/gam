"""E4 on standard benchmarks (#2951): the change each emoticon edit makes to HellaSwag, ARC-Easy, PIQA, LAMBADA
and the 67 BLiMP tasks, for VPD's subcomponent edit, the edit through the fitted decomposition's emoticon
subcomponent (bench/e4_decomp_edit.py), the paper's LoRA and the least-change weight edit, all at the same edit
success (98.5%): the change in each benchmark's correct-answer confidence. Reads bench/e4_methods_table.py's
methods/table.json.

usage: MPD_MEM_GIB=1 e4_benchmarks_three_fig.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

T = json.load(open(Path.home() / "mpd-data/frontier/e4_side/methods/table.json"))
OUT = Path.home() / "mpd-data/figures"
INK, INK2, SURF, AXIS = "#0b0b0b", "#52514e", "#ffffff", "#c3c2b7"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 24, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2})
METHODS = [("vpd", "VPD subcomponent edit", "#eb6834"),
           ("decomp_own", "fitted decomposition: edit through one subcomponent", "#a3360f"),
           ("lora", "LoRA, 282 examples", "#2a78d6"), ("compiled_span8", "minimum-disturbance weight edit", "#1baf7a")]
TASKS = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("piqa", "PIQA"), ("lambada", "LAMBADA"),
         ("blimp", "BLiMP (all 67 tasks)")]


def dots(ax, key):
    """One row per task, one dot with its 95% interval per method, rows top to bottom."""
    ys = np.arange(len(TASKS))[::-1]
    offsets = np.linspace(0.27, -0.27, len(METHODS))
    for y, (task, _) in zip(ys, TASKS):
        for (name, _, col), dy in zip(METHODS, offsets):
            m, lo, hi = T[name][key][task]
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.6, solid_capstyle="round", zorder=3)
            ax.scatter([m], [y + dy], s=90, color=col, edgecolor=SURF, linewidth=1.6, zorder=4)
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in TASKS], color=INK)
    ax.set_ylim(-0.6, len(TASKS) - 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(SURF)


fig, a1 = plt.subplots(figsize=(17, 12), dpi=150)
fig.patch.set_facecolor(SURF)
dots(a1, "bench_d_margin")
a1.set_xlabel("change in the correct answer's log-probability share (nats)", color=INK)
a1.xaxis.set_major_locator(MaxNLocator(5))
fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=2.6, ms=10, mec=SURF, label=l) for _, l, c in METHODS],
           loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(0.005, 0.93))
fig.suptitle("Benchmark change after each emoticon edit, at equal edit success", x=0.01, ha="left", y=0.995,
             fontsize=32, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.22, right=0.97, top=0.71, bottom=0.11)
fig.savefig(OUT / "e4_benchmarks_three.png", facecolor=SURF)
plt.close(fig)
print(OUT / "e4_benchmarks_three.png")
