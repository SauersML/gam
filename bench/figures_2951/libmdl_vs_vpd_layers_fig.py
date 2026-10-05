"""KL to the model against how many leading layers are replaced: our library explanation after each
training epoch (labelled with the bits that describe it, KL(q||p)), and VPD's decomposition (#2951).

Both on held-out rows 1024-1055 of vpd4l_clean4096. Ours: library_mdl's held-out `clean` KL at cut l
(P's first l layers, M's after), parsed from the fit log's epoch records. VPD: compare's battery rows
ci/layers_0, ci/layers_01, ci/layers_012 and ci/error_propagating (causal-importance masks).

    python bench/figures_2951/libmdl_vs_vpd_layers_fig.py FIT_LOG BATTERY.json OUT.png
"""
import json
import math
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log, battery, out = sys.argv[1:4]
epochs = []
for line in open(log):
    if not line.startswith("library fit epoch"):
        continue
    clean = re.search(r"clean: \[([^\]]*)\]", line)
    values = [float(v) for v in re.findall(r"Some\(([0-9.e+-]+)\)", clean.group(1))]
    bits = float(re.search(r"description_bits: ([0-9.e+]+)", line).group(1))
    epochs.append((int(re.search(r"epoch: (\d+)", line).group(1)), values, bits))
rows = json.load(open(battery))["held_out"]["rows"]
vpd = [rows[k]["kl_nats"]["mean"] / math.log(2)
       for k in ["ci/layers_0", "ci/layers_01", "ci/layers_012", "ci/error_propagating"]]

plt.rcParams.update({"font.size": 20, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12.5, 7.5), facecolor="white")
ax.set_facecolor("white")
x = range(1, 5)
cmap = plt.get_cmap("Blues")
for i, (epoch, values, bits) in enumerate(epochs):
    shade = cmap(0.35 + 0.6 * i / max(1, len(epochs) - 1))
    ax.plot(x, values, color=shade, lw=3, marker="o", ms=8)
    ax.annotate(f"ours, epoch {epoch + 1}: {bits / 1e6:.0f}M bits", (4, values[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=16, color=shade)
ax.plot(x, vpd, color="#b2182b", lw=3.5, marker="s", ms=10)
ax.annotate("VPD", (4, vpd[-1]), xytext=(10, 0), textcoords="offset points", va="center",
            fontsize=18, color="#b2182b")
ax.set_xticks(list(x))
ax.set_xticklabels(["first 1", "first 2", "first 3", "all 4"])
ax.set_xlim(0.7, 5.6)
ax.set_ylim(0, None)
ax.set_xlabel("layers replaced (the rest are the model's own)")
ax.set_ylabel("KL to the model (bits per token)")
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor="white")
print(out)
