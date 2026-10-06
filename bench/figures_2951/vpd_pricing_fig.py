"""VPD's code length per epoch of its bits-back pricing (#2951, explanation_battery::vpd_pricing).

    python3 vpd_pricing_fig.py PRICE.json OUT.png

PRICE.json is mpd_battery_2951's `price` output. Per epoch it draws, in bits per training token
(F / N, N the training tokens), F = KL(q || p) + sum_G (1/2) log2 |G| + N * data, split into the
description (KL(q || p) plus the groups' variance bits) and the data term (KL(M || VPD at a
posterior sample) per token), with VPD's subcomponent means held fixed.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

price, out = sys.argv[1:3]
run = json.load(open(price))["pricing"]
tokens = run.get("training_tokens") or 2**20
epochs = run["epochs"]
x = [e["epoch"] for e in epochs]
description = [(e["divergence_bits"] + e["variance_bits"]) / tokens for e in epochs]
data = [e["data_bits_per_token"] for e in epochs]
total = [e["objective_bits"] / tokens for e in epochs]

INK, MUTED = "#1f1f1e", "#5f5e58"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 17, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="white")
ax.set_facecolor("white")
ax.plot(x, total, color=INK, lw=2.6)
ax.plot(x, description, color=BLUE, lw=2.2)
ax.plot(x, data, color=ORANGE, lw=2.2)
ax.set_yscale("log")
ax.set_xlabel("pricing epoch")
ax.set_ylabel(f"bits per training token (N = {tokens:,.0f})")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
last = x[-1]
ax.set_xlim(x[0], last + 0.22 * (last - x[0] + 1))
pad = 0.012 * (last - x[0] + 1)
ax.annotate(f"F  {total[-1]:.2f}", (last, total[-1]), xytext=(last + pad, total[-1] * 1.15), color=INK, va="bottom", fontsize=16)
ax.annotate(f"description  {description[-1]:.2f}", (last, description[-1]), xytext=(last + pad, description[-1]), color=BLUE, va="center", fontsize=16)
ax.annotate(f"data  {data[-1]:.2f}", (last, data[-1]), xytext=(last + pad, data[-1]), color=ORANGE, va="center", fontsize=16)
fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
print(out)
