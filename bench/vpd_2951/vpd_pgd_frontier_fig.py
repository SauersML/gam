"""KL under VPD's own adversarial claim against the weights each explanation runs per word (#2951).

Reads the `box` results of `vpd_stepA_honesty.py` (one JSON, every family scored on the same
passages with `--parts fixed,pgd --delta only --seeds N`): per family the worst KL any PGD restart
finds after STEPS sign steps (`pgd_restarts[family/delta_adversarial].max`) against
`weights_per_word` (each subcomponent on is d_in + d_out - 1 reals). Families are grouped by name:
`vpd_gt_*` and `vpd_rounded` (VPD's sets at thresholds on its importance, one curve), `vpd_ci`
(VPD's importance box itself), `ours_corner` (the corner-searched sets), `ours_*` (the box-searched
sweep, one curve), `base_{neurons,svd,random}_*` (those libraries through the same selection),
`all_off`.

usage: vpd_pgd_frontier_fig.py RESULTS.json OUT.png [--steps 80] [--title TEXT]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--steps", default="80")
parser.add_argument("--title", default="Error under VPD's adversary against weights run per word")
args = parser.parse_args()

box = json.load(open(args.results))["box"]
restarts = box.get("pgd_restarts", {})
points = {}
for tag, r in restarts.items():
    name, delta = tag.split("/")
    if delta != "delta_adversarial" or name not in box.get("weights_per_word", {}):
        continue
    points[name] = (box["weights_per_word"][name] / 1e6, r["max"][args.steps], box["l0"][name])

# (label, colour, marker, names in drawing order, joined by a line)
groups = [
    ("VPD, thresholds on its importance (g > 0 ... 0.9)", "#eb6834", "o",
     sorted([n for n in points if n.startswith("vpd_gt_") or n == "vpd_rounded"], key=lambda n: points[n][0]), True),
    ("VPD, its importance box [g, 1]", "#eb6834", "D", [n for n in points if n == "vpd_ci"], False),
    ("ours, searched at the corner (Step A)", "#000000", "s", [n for n in points if n == "ours_corner"], False),
    ("ours, searched under the box claim", "#000000", "o",
     sorted([n for n in points if n.startswith("ours_") and n != "ours_corner"], key=lambda n: points[n][0]), True),
    ("model's own neurons", "#1baf7a", "^", sorted([n for n in points if n.startswith("base_neurons")], key=lambda n: points[n][0]), True),
    ("per-matrix SVD", "#4a3aa7", "v", sorted([n for n in points if n.startswith("base_svd")], key=lambda n: points[n][0]), True),
    ("random basis", "#eda100", "P", sorted([n for n in points if n.startswith("base_random")], key=lambda n: points[n][0]), True),
    ("everything off", "#777777", "X", [n for n in points if n == "all_off"], False),
]

plt.rcParams.update({"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 19, "legend.fontsize": 14})
fig, ax = plt.subplots(figsize=(13, 8.5), facecolor="white")
ax.set_facecolor("white")
for label, colour, marker, names, joined in groups:
    if not names:
        continue
    xs, ys = [points[n][0] for n in names], [points[n][1] for n in names]
    ax.plot(xs, ys, color=colour, marker=marker, markersize=11, linewidth=2 if joined and len(names) > 1 else 0,
            markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
ax.set_yscale("log")
ax.set_xlabel("weight numbers run per word (millions)")
ax.set_ylabel(f"KL per word under VPD's adversary ({args.steps} steps)")
ax.set_title(args.title, loc="left")
ax.grid(False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(args.out, dpi=150, facecolor="white")
print(f"wrote {args.out}: " + ", ".join(f"{n} ({x:.2f}M, {y:.2f})" for n, (x, y, _) in sorted(points.items())))
