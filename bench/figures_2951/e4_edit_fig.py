"""E4: the VPD paper's emoticon edit (one subcomponent) against LoRA, by the paper's own protocol (#2951).
Reads whatever points frontier's e4_replicate.py has written so far, so it can be re-run as LoRA points land.

usage: e4_edit_fig.py [REPLICATION.json]     (default ~/mpd-data/frontier/e4_replication_ci0.5.json)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "mpd-data/frontier/e4_replication_ci0.5.json"
d = json.load(open(src))
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#ffffff", "#e8e8e8", "#c3c2b7"
ORANGE, BLUE = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 22})
vpd = sorted(d["vpd"].values(), key=lambda e: e["p_fire"])
lora = [(e, False) for e in d["lora"].values()] + [(e, True) for e in d.get("lora_low", {}).values()]
# Loss is −ln P(correct next word), so a change of Δ nats scales that probability by e^±Δ: show it as a percent.
for e in list(vpd) + [e for e, _ in lora]:
    e["right_word_pct"] = 100 * np.expm1(e["declared_abs_damage_mean"])
panels = [("surr_kl", "Words near each emoticon", "KL from the original model (nats per word)"),
          ("global_kl", "40 ordinary documents", "KL from the original model (nats per word)"),
          ("right_word_pct", "Unrelated next-word predictions", "average % change in the probability\nthe model gives the correct next word")]


def vpd_at(key, p):
    """VPD's value at edit success p, log-interpolated along its strength sweep (None outside it)."""
    ps = np.array([e["p_fire"] for e in vpd])
    if not ps[0] <= p <= ps[-1]:
        return None
    return float(np.exp(np.interp(p, ps, np.log([e[key] for e in vpd]))))


fig, axes = plt.subplots(1, 3, figsize=(22, 9.5), dpi=200)
fig.patch.set_facecolor(SURF)
ratios = {}
for ax, (key, title, ylabel) in zip(axes, panels):
    ax.set_facecolor(SURF)
    ax.plot([e["p_fire"] for e in vpd], [e[key] for e in vpd], color=ORANGE, lw=3.2, zorder=3, solid_capstyle="round")
    ax.scatter([e["p_fire"] for e in vpd], [e[key] for e in vpd], s=110, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=4)
    for e, low in lora:
        ax.scatter(e["p_fire"], e[key], s=200, zorder=5, linewidth=2, marker="D" if low else "o",
                   facecolor=SURF if low else BLUE, edgecolor=BLUE)
        v = vpd_at(key, e["p_fire"])
        if v is not None and not low:
            ratios.setdefault(key, []).append(v / e[key])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.7, 1.005)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["70%", "80%", "90%", "100%"])
    ax.set_ylabel(ylabel, color=INK, labelpad=10, fontsize=20)
    ax.set_title(title, color=INK, fontsize=23, fontweight="bold", loc="left", pad=16)
    ax.tick_params(colors=INK2, labelsize=19)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
handles = [Line2D([], [], color=ORANGE, lw=3.2, marker="o", ms=11, mec=SURF, mew=2,
                  label="VPD: one subcomponent scaled up, increasing strength"),
           Line2D([], [], ls="", marker="o", ms=14, mfc=BLUE, mec=BLUE, label=f"LoRA, {d['n_train']} training examples")]
if any(low for _, low in lora):
    handles.append(Line2D([], [], ls="", marker="D", ms=12, mfc=SURF, mec=BLUE, mew=2, label="LoRA, 10 training examples"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.915), ncol=3, frameon=False, fontsize=20,
           labelcolor=INK, columnspacing=2.4)
near = ratios.get("surr_kl")
head = (f"At equal edit success, LoRA disturbs nearby text {min(near):.0f}–{max(near):.0f}× less than a VPD edit"
        if near else "VPD subcomponent edit vs LoRA")
fig.suptitle(head, color=INK, fontsize=28, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.supxlabel("edit success: probability of “o” after an emoticon colon", color=INK, fontsize=21, y=0.02)
fig.subplots_adjust(left=0.065, right=0.985, top=0.74, bottom=0.14, wspace=0.34)
out = Path.home() / "mpd-data/figures/e4_edit_vs_lora.png"
fig.savefig(out, facecolor=SURF)
print(out, {k: [round(x, 1) for x in v] for k, v in ratios.items()})
