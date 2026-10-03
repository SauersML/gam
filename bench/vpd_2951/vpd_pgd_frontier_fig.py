"""KL under VPD's own adversarial claim against the weights each explanation runs per word (#2951).

Reads the `box` results of `vpd_stepA_honesty.py` (one JSON, every family scored on the same
passages with `--parts fixed,pgd --delta only --seeds N`): per family the worst KL any PGD restart
finds after STEPS sign steps (`pgd_restarts[family/delta_adversarial].max`, filled; the median restart open) against
`weights_per_word` (each subcomponent on is d_in + d_out - 1 reals). Families are grouped by name:
`vpd_gt_*` and `vpd_rounded` (VPD's sets at thresholds on its importance, one curve), `vpd_ci`
(VPD's importance box itself), `ours_corner` (our search on VPD's subcomponents under the corner claim),
`full_*` (our full method under the box claim, one point per n), `base_{neurons,svd,random}_*` (those libraries through the same selection),
`e2e_*` (from-scratch libraries), `all_off`.

usage: vpd_pgd_frontier_fig.py OUT.png RESULTS.json... [--steps 80] [--title TEXT]
(several results files are merged; a family scored in more than one keeps the last).
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
parser.add_argument("--steps", default="80")
parser.add_argument("--title", default="KL under VPD's adversary, by weights run per word")
args = parser.parse_args()

box = {"pgd_restarts": {}, "weights_per_word": {}, "l0": {}}
for path in args.results:
    part = json.load(open(path))["box"]
    for k in box:
        box[k].update(part.get(k, {}))
restarts = box["pgd_restarts"]
points = {}
for tag, r in restarts.items():
    name, delta = tag.split("/")
    if delta != "delta_adversarial" or name not in box.get("weights_per_word", {}):
        continue
    runs = sorted(seed[args.steps] for seed in r["seeds"])
    points[name] = (box["weights_per_word"][name] / 1e6, r["max"][args.steps], box["l0"][name], float(np.median(runs)))

# (label, colour, marker, names in drawing order, joined by a line)
groups = [
    ("VPD's choices at importance thresholds g > 0 to 0.9", "#eb6834", "o",
     sorted([n for n in points if n.startswith("vpd_gt_") or n == "vpd_rounded"], key=lambda n: points[n][0]), True),
    ("VPD's importance box [g, 1]", "#eb6834", "D", [n for n in points if n == "vpd_ci"], False),
    ("our search on VPD's subcomponents, corner claim", "#000000", "s", [n for n in points if n == "ours_corner"], False),
    ("our full method, box claim", "#000000", "o",
     sorted([n for n in points if n.startswith("full_")], key=lambda n: points[n][0]), True),
    ("model's own neurons", "#1baf7a", "^", sorted([n for n in points if n.startswith("base_neurons")], key=lambda n: points[n][0]), True),
    ("per-matrix SVD", "#4a3aa7", "v", sorted([n for n in points if n.startswith("base_svd")], key=lambda n: points[n][0]), True),
    ("random basis", "#eda100", "P", sorted([n for n in points if n.startswith("base_random")], key=lambda n: points[n][0]), True),
    ("from-scratch libraries (e2e)", "#2a78d6", "o",
     sorted([n for n in points if n.startswith("e2e_")], key=lambda n: points[n][0]), True),
    ("everything off", "#777777", "X", [n for n in points if n == "all_off"], False),
]

plt.rcParams.update({"font.size": 20, "axes.titlesize": 30, "axes.labelsize": 22, "legend.fontsize": 16,
                     "xtick.labelsize": 20, "ytick.labelsize": 20})
fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
ax.set_facecolor("white")
for label, colour, marker, names, joined in groups:
    if not names:
        continue
    xs, ys, med = [points[n][0] for n in names], [points[n][1] for n in names], [points[n][3] for n in names]
    ax.plot(xs, ys, color=colour, marker=marker, markersize=11, linewidth=2 if joined and len(names) > 1 else 0,
            markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
    # The median restart: open markers on a thin dashed line.
    ax.plot(xs, med, color=colour, marker=marker, markersize=9, linewidth=1 if joined and len(names) > 1 else 0,
            linestyle="--", markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.5, zorder=2)
ax.set_yscale("log")
ax.set_xlabel("weight numbers run per word (millions)")
ax.set_ylabel(f"KL per word under VPD's adversary ({args.steps} steps)")
ax.set_title(args.title, loc="left", fontweight="bold")
ax.grid(False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
# What filled and open markers mean, as two more legend entries.
restart_count = len(next(iter(restarts.values()))["seeds"])
ax.plot([], [], color="#555555", marker="o", linewidth=0, markersize=11, label=f"filled: worst of {restart_count} random restarts")
ax.plot([], [], color="#555555", marker="o", linewidth=0, markersize=9, markerfacecolor="white", label="open: median restart")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout()
fig.savefig(args.out, dpi=150, facecolor="white", bbox_inches="tight")
print(f"wrote {args.out}: " + ", ".join(f"{n} ({x:.2f}M, max {y:.2f}, median {m:.2f})" for n, (x, y, _, m) in sorted(points.items())))
