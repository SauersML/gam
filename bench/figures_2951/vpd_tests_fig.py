"""Two held-out tests of VPD's 4-layer decomposition (#2951), from compare's results.

Left: resample, from another text, the input directions that VPD's active subcomponents do not read
(the causal-abstraction complement test), one attention or MLP block at a time and at all eight blocks
at once; VPD predicts no change. KL(M || E) per token in nats, held-out rows 1024-1055, mean over 64
source texts with the worst source marked. Right: faithfulness of circuits on the four subject-verb
agreement tasks (Marks et al. 2025), mean over tasks, nodes ranked by RelP attribution and the rest
mean-ablated, for the model's own MLP neurons, VPD's MLP subcomponents (attention left whole, matched
coverage) and VPD's subcomponents at all 24 sites.

    python bench/figures_2951/vpd_tests_fig.py BATTERY.json CIRCUITS.json OUT.png

BATTERY.json is examples/mpd_battery_2951's `vpd` output (its interchange with 64 sources),
CIRCUITS.json its `circuits` output; their KLs are in bits, drawn here in nats.
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

battery, circuits, out = sys.argv[1:4]
NATS = math.log(2)
run = json.load(open(battery))
fam = run["interchange"]["patches"]
allb = {"mean": fam["complement_every_block"]["all_sources"]["mean"] * NATS}
own = run["protocols"]["masks"]["ci"]["layers_0123"]["kl_bits"]["mean"] * NATS
tasks = {t: v for t, v in json.load(open(circuits))["circuits"].items() if t != "mean"}

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, (left, right) = plt.subplots(1, 2, figsize=(19, 8.2), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.42})

# Left: complement patches.
names = [f"layer {b // 2} {'attention' if b % 2 == 0 else 'MLP'}" for b in range(8)]
means = [np.mean(fam[f"complement_block_{b}"]["per_source_mean_bits"]) * NATS for b in range(8)]
worst = [fam[f"complement_block_{b}"]["shared_source"]["worst_of_64"] * NATS for b in range(8)]
labels = names + ["all 8 blocks at once"]
means.append(allb["mean"])
worst.append(None)
y = np.arange(len(labels))[::-1].astype(float)
y[-1] -= 0.6
left.barh(y, means, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
for yi, w in zip(y, worst):
    if w is not None:
        left.plot([w], [yi], marker="o", ms=10, color=INK, markeredgecolor=SURFACE, markeredgewidth=2)
left.axvline(own, color=ORANGE, lw=3)
left.set_yticks(y)
left.set_yticklabels(labels)
left.tick_params(axis="y", length=0)
left.set_xlabel("KL to the model (nats per token)")
left.set_xlim(0, 10.2)
left.annotate("VPD's own error", (own, y[0] + 0.75), xytext=(8, 0), textcoords="offset points",
              color=ORANGE, fontsize=17, va="center", annotation_clip=False)
left.annotate("worst of 64 texts", (worst[1], y[1]), xytext=(12, 0), textcoords="offset points",
              color=INK, fontsize=16, va="center")
left.annotate(f"{allb['mean']:.1f}", (allb["mean"], y[-1]), xytext=(-10, 0), textcoords="offset points",
              color=SURFACE, fontsize=18, fontweight="bold", va="center", ha="right")
left.set_title("Replacing what VPD says is unused", loc="left", pad=16, fontsize=22)

# Right: circuit faithfulness against size.
def mean_curve(kind):
    ks = tasks["simple"][kind]["k"]
    vals = np.mean([tasks[t][kind]["faithfulness"] for t in tasks], axis=0)
    return np.array(ks), vals

for kind, color, name in [("neurons", BLUE, "the model's own neurons"),
                          ("vpd_mlp", ORANGE, "VPD's MLP subcomponents"),
                          ("vpd", AQUA, "VPD, all subcomponents")]:
    ks, vals = mean_curve(kind)
    keep = ks <= 4096
    right.plot(ks[keep], vals[keep], color=color, lw=3, marker="o", ms=8, markeredgecolor=SURFACE,
               markeredgewidth=2)
    right.annotate(name, (ks[keep][-1], vals[keep][-1]), xytext=(10, 0), textcoords="offset points",
                   color=INK, fontsize=17, va="center")
right.set_xscale("log", base=2)
right.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
right.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
right.set_xlim(0.8, 4096 * 1.2)
right.set_ylim(0, 1.25)
right.axhline(1.0, color="#d9d9d6", lw=1.5, zorder=0)
right.set_xlabel("circuit size (nodes kept)")
right.set_ylabel("share of the behaviour recovered")
right.set_title("Circuits for subject–verb agreement", loc="left", pad=16, fontsize=22)

fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
