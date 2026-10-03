"""The site-switch claim, family by family (#2951): per passage each of the 24 sites runs its
explanation or its native map, and the claim's error is the worst KL over those choices.

Reads `vpd_stepA_honesty.py` results (`claim`, or `sites` runs under the keys `layer_switches`,
`sites_layer_L`, `sites_attack`; several files merge) and draws, per family, the mean over
passages of each passage's KL (per word) with every site replaced, and the worst over: the 16 layer
switches (exact), each layer's 64 site subsets with the other layers native (exact), and the 24
sites (the attack's worst found, a lower bound). A dot marks the worst passage.

usage: vpd_site_claim_fig.py OUT.png RESULTS.json... [--families vpd_rounded,ours_corner,...]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("results", type=Path, nargs="+")
parser.add_argument("--families", default=None, help="comma-separated, in drawing order (default: every family found)")
args = parser.parse_args()

data = {}
for path in args.results:
    for key, value in json.load(open(path)).items():
        if isinstance(value, dict):
            data.setdefault(key, {}).update(value)

LABELS = {
    "vpd_rounded": "VPD's subcomponents and choices (g > 0)",
    "vpd_ci": "VPD's choices, importance as mask",
    "ours_corner": "our search on VPD's subcomponents, corner claim",
}
COLOURS = {"vpd_rounded": "#eb6834", "vpd_ci": "#f2a07d", "ours_corner": "#000000"}
layers = sorted(int(k.rsplit("_", 1)[1]) for k in data if k.startswith("sites_layer_"))
claims = [("sites_attack", "all_replaced", "all 24 sites\nreplaced")] + \
    [("layer_switches", "passage_worst_subset", "worst of the\n16 layer subsets")] + \
    [(f"sites_layer_{l}", "passage_worst_subset", f"worst of layer {l}'s\n64 site subsets") for l in layers] + \
    [("sites_attack", "passage_worst_subset", "worst found over\nall 24 sites")]
claims = [c for c in claims if c[0] in data]
present = sorted({f for key, _, _ in claims for f in data[key]})
families = args.families.split(",") if args.families else present
families = [f for f in families if all(f in data[key] for key, _, _ in claims)]

plt.rcParams.update({"font.size": 20, "axes.titlesize": 30, "axes.labelsize": 22, "legend.fontsize": 18,
                     "xtick.labelsize": 17, "ytick.labelsize": 20})
fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
ax.set_facecolor("white")
width = 0.8 / max(len(families), 1)
for f_index, family in enumerate(families):
    xs = np.arange(len(claims)) + (f_index - (len(families) - 1) / 2) * width
    means = [data[key][family][stat]["mean"] for key, stat, _ in claims]
    worst = [data[key][family][stat]["max"] for key, stat, _ in claims]
    colour = COLOURS.get(family, "#2a78d6")
    ax.bar(xs, means, width * 0.92, color=colour, label=LABELS.get(family, family), zorder=2)
    ax.plot(xs, worst, linestyle="none", marker="o", markersize=9, color=colour, markeredgecolor="white", zorder=3)
    for x, m in zip(xs, means):
        ax.text(x, m, f"{m:.2f}", ha="center", va="bottom", fontsize=14)
ax.plot([], [], linestyle="none", marker="o", markersize=9, color="#555555", label="dot: the worst passage")
ax.set_xticks(np.arange(len(claims)))
ax.set_xticklabels([label for _, _, label in claims])
ax.set_ylabel("KL per word, mean over passages")
ax.set_title("Replacing sites by their explanations, in any combination", loc="left", fontweight="bold")
ax.grid(False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.savefig(args.out, dpi=150, facecolor="white", bbox_inches="tight")
print(f"wrote {args.out}: " + "; ".join(
    f"{f}: " + ", ".join(f"{key}/{stat} {data[key][f][stat]['mean']:.3f}" for key, stat, _ in claims) for f in families))
