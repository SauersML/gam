"""Figure for the one-page note on VPD's layer cancellation (#2951): (a) KL from the model for all 15
sets of replaced layers, by size, split by whether the set includes layer 3; (b) subtracting or adding
back layer 3's response (the change in the output of its zero-importance subcomponents),
averaged over the earlier layer l = 0, 1, 2. Rounded masks (CI > 0), held-out rows 1024-1056.

    python bench/figures_2951/vpd_cancellation_note_fig.py BATTERY.json CANCELLATION.json OUT.pdf
"""
import itertools
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

battery, cancellation, out = sys.argv[1:4]
rows = json.load(open(battery))["held_out"]["rows"]
c = json.load(open(cancellation))["cancellation"]["rounded"]
kl = lambda s: rows["rounded/error_propagating" if len(s) == 4 else "rounded/layers_" + "".join(map(str, s))]["kl_nats"]["mean"]
mean_l = lambda key: sum(c[key.format(l)]["mean"] for l in range(3)) / 3

INK, MUTED, BLUE, ORANGE, GRAY = "#1f1f1e", "#6b6b68", "#2a78d6", "#eb6834", "#b4b4b1"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42,
})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})

# (a) every set of replaced layers
for k in range(1, 5):
    sets = list(itertools.combinations(range(4), k))
    for has3, color, dx in [(False, BLUE, -0.09), (True, ORANGE, 0.09)]:
        ys = [kl(s) for s in sets if (3 in s) == has3]
        a.scatter([k + dx] * len(ys), ys, s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3,
                  label=("includes layer 3" if has3 else "excludes layer 3") if k == 2 else None)
a.set_xticks(range(1, 5))
a.set_xlim(0.5, 4.5)
a.set_ylim(0, 0.75)
a.set_xlabel("number of layers replaced")
a.set_ylabel("KL divergence (nats/token)")
a.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.1)
a.text(-0.2, 1.02, "a", transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom")

# (b) interventions
bars = [("earlier\nlayer", mean_l("kl_{}"), GRAY),
        ("earlier\nlayer,\nminus\nresponse", mean_l("kl_{}_without_I"), BLUE),
        ("earlier\nlayer and\nlayer 3", mean_l("kl_{}3"), GRAY),
        ("earlier\nlayer and\nlayer 3,\nplus\nresponse", mean_l("kl_{}3_no_interaction"), ORANGE)]
xs = [0, 1.0, 2.5, 3.5]
for x, (name, v, color) in zip(xs, bars):
    b.bar(x, v, 0.75, color=color, edgecolor="white", linewidth=0.8)
    b.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
for x0, x1, v in [(xs[0], xs[1], bars[0][1]), (xs[2], xs[3], bars[2][1])]:
    b.plot([x0 - 0.375, x1 + 0.375], [v, v], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=0)
b.set_xticks(xs)
b.set_xticklabels([n for n, _, _ in bars], fontsize=8.5, linespacing=1.05)
b.tick_params(axis="x", length=0)
b.set_ylim(0, 1.12)
b.set_ylabel("KL divergence (nats/token)")
b.set_xlabel("layers replaced", labelpad=4)
b.text(-0.17, 1.02, "b", transform=b.transAxes, fontsize=11, fontweight="bold", va="bottom")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print(out)
