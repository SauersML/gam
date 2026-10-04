"""E4 on standard benchmarks (#2951): the change each emoticon edit makes to HellaSwag, ARC-Easy, PIQA, LAMBADA
and the 67 BLiMP tasks, for VPD's subcomponent edit, the paper's LoRA and the least-change weight edit, all at the
same edit success (98.5%). Reads bench/e4_methods_table.py's methods/table.json.

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
METHODS = [("vpd", "VPD subcomponent edit", "#eb6834"), ("lora", "LoRA, 282 examples", "#2a78d6"),
           ("compiled_span8", "least change to ordinary text, exact on emoticons", "#1baf7a")]
TASKS = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("piqa", "PIQA"), ("lambada", "LAMBADA"),
         ("blimp", "BLiMP (all 67 tasks)")]


def dots(ax, key):
    """One row per task, one dot with its 95% interval per method, rows top to bottom."""
    ys = np.arange(len(TASKS))[::-1]
    offsets = np.linspace(0.22, -0.22, len(METHODS))
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


fig, (a1, a2) = plt.subplots(1, 2, figsize=(25, 10.5), dpi=150, gridspec_kw={"wspace": 0.08})
fig.patch.set_facecolor(SURF)
dots(a1, "bench_d_margin")
a1.set_xlabel("change in the correct answer's log-probability share (nats)", color=INK)
a1.xaxis.set_major_locator(MaxNLocator(5))
dots(a2, "bench_d_acc")
a2.set_yticklabels([])
a2.set_xlabel("change in accuracy", color=INK)
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:+.1f}%" if v else "0"))
a1.set_title("Confidence in the right answer", loc="left", fontweight="bold", color=INK, pad=12)
a2.set_title("Accuracy", loc="left", fontweight="bold", color=INK, pad=12)
fig.legend(handles=[Line2D([], [], color=c, marker="o", lw=2.6, ms=10, mec=SURF, label=l) for _, l, c in METHODS],
           loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.005, 0.9))
fig.suptitle("Benchmark change after each emoticon edit, at equal edit success", x=0.01, ha="left", y=0.995,
             fontsize=34, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.15, right=0.98, top=0.74, bottom=0.14)
fig.savefig(OUT / "e4_benchmarks_three.png", facecolor=SURF)
plt.close(fig)
print(OUT / "e4_benchmarks_three.png")
