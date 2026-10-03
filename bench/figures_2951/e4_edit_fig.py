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
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 15})
vpd = sorted(d["vpd"].values(), key=lambda e: e["p_fire"])
lora = [(e, False) for e in d["lora"].values()] + [(e, True) for e in d.get("lora_low", {}).values()]
panels = [("surr_kl", "Text next to the edit", "KL from the original model (nats per word)"),
          ("global_kl", "All other text", "KL from the original model (nats per word)"),
          ("declared_abs_damage_mean", "Six unrelated behaviours", "increase in loss (nats)")]


def vpd_at(key, p):
    """VPD's value at edit success p, log-interpolated along its strength sweep (None outside it)."""
    ps = np.array([e["p_fire"] for e in vpd])
    if not ps[0] <= p <= ps[-1]:
        return None
    return float(np.exp(np.interp(p, ps, np.log([e[key] for e in vpd]))))


fig, axes = plt.subplots(1, 3, figsize=(18, 7.4), dpi=200)
fig.patch.set_facecolor(SURF)
ratios = {}
for ax, (key, title, ylabel) in zip(axes, panels):
    ax.set_facecolor(SURF)
    ax.plot([e["p_fire"] for e in vpd], [e[key] for e in vpd], color=ORANGE, lw=2.4, zorder=3, solid_capstyle="round")
    ax.scatter([e["p_fire"] for e in vpd], [e[key] for e in vpd], s=60, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=4)
    for e, low in lora:
        ax.scatter(e["p_fire"], e[key], s=95, zorder=5, linewidth=2, marker="D" if low else "o",
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
    ax.set_xlabel("edit success (probability of “o” after an emoticon colon)", color=INK, labelpad=10, fontsize=13)
    ax.set_ylabel(ylabel, color=INK, labelpad=8, fontsize=13)
    ax.set_title(title, color=INK, fontsize=15.5, fontweight="bold", loc="left", pad=12)
    ax.tick_params(colors=INK2, labelsize=12)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    r = ratios.get(key)
    if r:
        if min(r) >= 1:
            note = f"LoRA {min(r):.0f}–{max(r):.0f}× lower"
        elif max(r) <= 1:
            note = f"LoRA {1 / max(r):.1f}–{1 / min(r):.1f}× higher"
        else:
            note = f"LoRA from {1 / min(r):.1f}× higher to {max(r):.1f}× lower"
        ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=13, color=INK, va="top")
handles = [Line2D([], [], color=ORANGE, lw=2.4, marker="o", ms=8, mec=SURF, mew=2,
                  label="VPD: one subcomponent scaled up, increasing strength"),
           Line2D([], [], ls="", marker="o", ms=9, mfc=BLUE, mec=BLUE, label=f"LoRA, {d['n_train']} training examples")]
if any(low for _, low in lora):
    handles.append(Line2D([], [], ls="", marker="D", ms=8, mfc=SURF, mec=BLUE, mew=2, label="LoRA, 10 training examples"))
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.875), ncol=3, frameon=False, fontsize=13,
           labelcolor=INK, columnspacing=2.4)
near = ratios.get("surr_kl")
head = (f"At equal edit success, LoRA disturbs nearby text {min(near):.0f}–{max(near):.0f}× less than a VPD edit"
        if near else "VPD subcomponent edit vs LoRA")
fig.suptitle(head, color=INK, fontsize=19, fontweight="bold", x=0.02, ha="left", y=0.975)
fig.text(0.02, 0.905, "The VPD paper's emoticon edit on its 4-layer model, with the paper's protocol. "
         "Lower is better in every panel.", color=INK2, fontsize=14, ha="left")
fig.subplots_adjust(left=0.06, right=0.99, top=0.7, bottom=0.12, wspace=0.28)
out = Path.home() / "mpd-data/figures/e4_edit_vs_lora.png"
fig.savefig(out, facecolor=SURF)
print(out, {k: [round(x, 1) for x in v] for k, v in ratios.items()})
