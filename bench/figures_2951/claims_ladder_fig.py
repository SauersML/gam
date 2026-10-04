"""How much error each faithfulness claim allows on VPD's 4-layer model (#2951): the same explanations (VPD's
sets, and our search's sets on VPD's subcomponents) scored under claims of increasing strength, from all 24
weight matrices replaced at once to any combination of individual subcomponents per token. Reads the honesty
harness's results (sites_exact_32.json, sites_attack_32.json, word_battery_32.json in ~/mpd-data/frontier/claims).

usage: MPD_MEM_GIB=1 claims_ladder_fig.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

D = Path.home() / "mpd-data/frontier/claims"
OUT = Path.home() / "mpd-data/figures"
INK, INK2, SURF, AXIS = "#0b0b0b", "#52514e", "#ffffff", "#c3c2b7"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 24, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2})
exact = json.load(open(D / "sites_exact_32.json"))["layer_switches"]
attack = json.load(open(D / "sites_attack_32.json"))["sites"]
word = json.load(open(D / "word_battery_32.json"))["box"]
FAMILIES = [("vpd_rounded", "VPD's subcomponent sets", "#eb6834"),
            ("ours_corner", "our search's sets (on VPD's subcomponents)", "#2a78d6")]
CLAIMS = ["all 24 matrices\nreplaced", "worst mix of\nwhole layers", "worst mix of\nthe 24 matrices",
          "worst mix of single\nsubcomponents,\nper token"]


def values(name):
    return [exact[name]["all_replaced"]["mean"], exact[name]["passage_worst_subset"]["mean"],
            attack[name]["passage_worst_subset"]["mean"], word["word_adversary"][f"{name}/delta_adversarial"]["box"]["mean"]]


fig, ax = plt.subplots(figsize=(17, 10), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
xs = np.arange(len(CLAIMS))
for (name, label, col), dx in zip(FAMILIES, (-0.08, 0.08)):
    v = values(name)
    ax.plot(xs + dx, v, color=col, lw=2.6, zorder=3)
    ax.scatter(xs + dx, v, s=150, color=col, edgecolor=SURF, linewidth=2, zorder=4, label=label)
off = word["fixed_kl"]["all_off"]["mean"]
ax.axhline(off, color=INK2, lw=1.6, ls=(0, (5, 4)), zorder=1)
ax.text(-0.35, off * 1.18, "every subcomponent switched off", color=INK2, ha="left", fontsize=20)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(0.2, 300)
ax.set_xticks(xs)
ax.set_xticklabels(CLAIMS, color=INK, fontsize=21)
ax.set_xlim(-0.4, len(CLAIMS) - 0.6)
ax.set_ylabel("KL per token (nats)", color=INK)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=21)
fig.suptitle("Error the same explanations allow under stronger faithfulness claims", x=0.01, ha="left", y=0.98,
             fontsize=30, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2)
fig.savefig(OUT / "claims_ladder.png", facecolor=SURF)
plt.close(fig)
print(OUT / "claims_ladder.png")
