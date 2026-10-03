"""E4 on real benchmarks: how much each LoRA edit moves benchmark answers, relative to the VPD edit tuned to the
same edit success (#2951). Reads ~/mpd-data/frontier/e4_side/bench/e4_benchmarks.json (e4_benchmarks_data.py).

Per item, the change in the margin log p(correct) − log Σ_choices p (LAMBADA: the true word's log p), absolute;
averaged over the benchmark's items; LoRA ÷ matched VPD, with the bootstrap 95% interval.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

R = json.load(open(Path.home() / "mpd-data/frontier/e4_side/bench/e4_benchmarks.json"))
OUT = Path.home() / "mpd-data/figures/e4_benchmarks_ratio.png"
INK, INK2, SURF, AXIS = "#0b0b0b", "#52514e", "#ffffff", "#c3c2b7"
BLUES = {"0.1": "#9ec5f0", "1": "#5c9ae3", "10": "#2a78d6", "100": "#174a8c"}
TASKS = [("blimp", "BLiMP grammar\n(67 tests)"), ("lambada", "LAMBADA\nlast word"), ("piqa", "PIQA"),
         ("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag")]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 20})

fig, ax = plt.subplots(figsize=(16, 9.5), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.axvline(1, color=AXIS, lw=2, zorder=1)
offsets = {"0.1": -0.27, "1": -0.09, "10": 0.09, "100": 0.27}
for row, (task, _) in enumerate(TASKS):
    pairs = R["tasks"][task]["pairs"]
    for lora, entry in pairs.items():
        ratio, lo, hi = entry["abs_ratio"]
        n_train, lam = lora.removeprefix("lora").split("_lam")
        y = row + offsets[lam] + (0.035 if n_train == "10" else -0.035)
        filled = n_train == "282"
        ax.plot([lo, hi], [y, y], color=BLUES[lam], lw=2.5, zorder=2, solid_capstyle="round")
        ax.scatter([ratio], [y], s=170 if filled else 150, zorder=3, marker="o" if filled else "D", linewidth=2.2,
                   facecolor=BLUES[lam] if filled else SURF, edgecolor=BLUES[lam])
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 3])
ax.set_xticklabels(["¼×", "½×", "same", "2×", "3×"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(0.2, 3.3)
ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels([label for _, label in TASKS])
ax.set_ylim(len(TASKS) - 0.5, -0.5)
ax.tick_params(colors=INK, labelsize=19, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("how much LoRA changes the model's benchmark answers, relative to VPD's edit at the same edit success",
              color=INK, fontsize=18, labelpad=14)
handles = [Line2D([], [], ls="", marker="o", ms=13, mfc=BLUES[lam], mec=BLUES[lam], label=f"stay-close weight {lam}") for lam in BLUES]
handles.append(Line2D([], [], ls="", marker="o", ms=13, mfc=INK2, mec=INK2, label="282 training examples"))
handles.append(Line2D([], [], ls="", marker="D", ms=12, mfc=SURF, mec=INK2, mew=2.2, label="10 training examples"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.18, 0.885), ncol=3, frameon=False, fontsize=16,
           labelcolor=INK, columnspacing=1.6, handletextpad=0.4)
fig.suptitle("At equal edit success, a LoRA held close to the model (weight ≥ 10) changes\nbenchmark answers 1.3–4× less than VPD's edit; loosely held LoRAs can change them more",
             color=INK, fontsize=24, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.subplots_adjust(left=0.18, right=0.98, top=0.74, bottom=0.13)
fig.savefig(OUT, facecolor=SURF)
print(OUT)
