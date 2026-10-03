"""E4 on standard benchmarks (#2951): the change each edit makes to HellaSwag, ARC-Easy, PIQA, LAMBADA and the 67
BLiMP tasks, LoRA against the VPD edit of exactly equal edit success. Reads bench/e4_benchmarks_data.py's summary.

usage: MPD_MEM_GIB=1 e4_benchmarks_fig.py [JSON] [HEADLINE_LORA]   (default lora282_lam10)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "mpd-data/frontier/e4_side/bench/e4_benchmarks.json"
d = json.load(open(src))
HEAD = sys.argv[2] if len(sys.argv) > 2 else "lora282_lam10"
VHEAD = "vpd_match_" + HEAD
OUT = Path.home() / "mpd-data/figures"
INK, INK2, SURF, AXIS = "#0b0b0b", "#52514e", "#ffffff", "#c3c2b7"
ORANGE, BLUE = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 18, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2})
meta, T = d["meta"], d["tasks"]
HL = f"LoRA, {meta[HEAD]['n_train']} examples"
HV = "VPD subcomponent edit"
MAIN = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-Easy"), ("piqa", "PIQA"), ("lambada", "LAMBADA"),
        ("blimp", "BLiMP (all 67 tasks)")]


def dots(ax, rows, key, xlabel, zero=True):
    """One row per task: VPD and LoRA means with 95% intervals; rows given top to bottom."""
    ys = np.arange(len(rows))[::-1]
    for y, (k, lab) in zip(ys, rows):
        for nm, col, dy in ((VHEAD, ORANGE, 0.14), (HEAD, BLUE, -0.14)):
            m, lo, hi = T[k][key][nm]
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=2.6, solid_capstyle="round", zorder=3)
            ax.scatter([m], [y + dy], s=90, color=col, edgecolor=SURF, linewidth=1.6, zorder=4)
    if zero:
        ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows], color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF)


legend = [Line2D([], [], color=ORANGE, marker="o", lw=2.6, ms=10, mec=SURF, label=HV),
          Line2D([], [], color=BLUE, marker="o", lw=2.6, ms=10, mec=SURF, label=HL)]

# ---------------------------------------------------------------- 1. the five benchmarks
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7.5), dpi=170, gridspec_kw={"wspace": 0.08})
fig.patch.set_facecolor(SURF)
dots(a1, MAIN, "d_margin", "change in the correct answer's log-probability share (nats)")
dots(a2, MAIN, "d_acc", "change in accuracy")
a2.set_yticklabels([])
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:+.1f}%" if v else "0"))
a1.set_title("Confidence in the right answer", loc="left", fontweight="bold", color=INK, pad=12)
a2.set_title("Accuracy", loc="left", fontweight="bold", color=INK, pad=12)
fig.legend(handles=legend, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("Benchmark change after each emoticon edit, at equal edit success", x=0.01, ha="left", y=0.995,
             fontsize=24, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.17, right=0.98, top=0.8, bottom=0.14)
fig.savefig(OUT / "e4_benchmarks.png", facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- 2. all 67 BLiMP tasks, ranked by the VPD edit's change
bl = [k for k in T if k.startswith("blimp:")]
bl.sort(key=lambda k: T[k]["d_margin"][VHEAD][0])
rows = [(k, k[6:].replace("_", " ")) for k in bl]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(22, 0.36 * len(rows) + 3), dpi=150, gridspec_kw={"wspace": 0.06})
fig.patch.set_facecolor(SURF)
dots(a1, rows, "d_margin", "change in the grammatical sentence's log-probability share (nats)")
a1.tick_params(axis="y", labelsize=14)
ys = np.arange(len(rows))[::-1]
for y, (k, _) in zip(ys, rows):
    r, lo, hi = T[k]["pairs"][HEAD]["abs_ratio"]
    col = BLUE if hi < 1 else ORANGE if lo > 1 else INK2
    a2.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle="round")
    a2.scatter([r], [y], s=90, color=col, edgecolor=SURF, linewidth=1.6, zorder=4)
a2.set_xscale("log")
a2.axvline(1, color=INK2, lw=1.2)
a2.set_yticks(ys)
a2.set_yticklabels([])
a2.set_ylim(-0.6, len(rows) - 0.4)
a2.set_xlabel("LoRA's average change ÷ VPD's (below 1: LoRA changes less)", color=INK)
a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
for s in ("top", "right"):
    a2.spines[s].set_visible(False)
a1.set_title("Change per task", loc="left", fontweight="bold", color=INK, pad=12)
a2.set_title("Size of change, LoRA ÷ VPD", loc="left", fontweight="bold", color=INK, pad=12)
H = 0.36 * len(rows) + 3
fig.legend(handles=legend, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 1 - 0.1 / H))
fig.suptitle("BLiMP grammar tasks after each emoticon edit, at equal edit success", x=0.01, ha="left",
             y=1 - 0.1 / H, fontsize=24, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.24, right=0.98, top=1 - 1.6 / H, bottom=1.0 / H)
fig.savefig(OUT / "e4_benchmarks_blimp.png", facecolor=SURF)
plt.close(fig)
print(OUT / "e4_benchmarks.png", OUT / "e4_benchmarks_blimp.png")
