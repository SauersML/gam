"""Circuit faithfulness and completeness against circuit size, nodes ranked by measured patching
effects (#2951, explanation_battery::NodeBasis::patch_effects), for the model's own MLP neurons,
VPD's MLP subcomponents (matched coverage) and VPD's subcomponents at all 24 sites.

    python3 vpd_circuits_fig.py CIRCUITS.json OUT.png

CIRCUITS.json is mpd_battery_2951's `circuits` output on the subject-verb agreement pairs of Marks
et al. 2025 (four tasks; held-out pairs). With m the logit difference of the correct verb form over
the wrong one, C the k top-ranked nodes kept and every other node mean-ablated, and the empty set
ablating all nodes, faithfulness is (m(C) - m(empty)) / (m(M) - m(empty)); completeness is the same
quantity with C itself ablated and the rest kept. Each curve is the mean over the four tasks.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

circuits, out = sys.argv[1:3]
tasks = {t: v for t, v in json.load(open(circuits))["circuits"].items() if t != "mean"}

INK, MUTED, SURFACE = "#1f1f1e", "#6b6b68", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 19, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})
fig, panels = plt.subplots(1, 2, figsize=(19, 7.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.45})
BASES = [("neurons", BLUE, "model's MLP neurons"), ("vpd_mlp", ORANGE, "VPD, MLP subcomponents"), ("vpd", AQUA, "VPD, all subcomponents")]
for ax, measure, title in zip(panels, ["faithfulness", "completeness"], ["Faithfulness: the circuit alone", "Completeness: the circuit removed"]):
    for kind, color, name in BASES:
        if any(kind not in tasks[t] for t in tasks):
            continue
        ks = np.array(next(iter(tasks.values()))[kind]["k"])
        vals = np.mean([tasks[t][kind][measure] for t in tasks], axis=0)
        ax.plot(ks, vals, color=color, lw=3, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color="#d9d9d6", lw=1.5, zorder=0)
    ax.axhline(0.0, color="#d9d9d6", lw=1.5, zorder=0)
    ax.set_xlabel("circuit size k (nodes)")
    ax.set_ylabel(f"{measure}")
    ax.set_title(title, loc="left", pad=14, fontsize=21)
handles, names = panels[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=18)
fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
print(out)
