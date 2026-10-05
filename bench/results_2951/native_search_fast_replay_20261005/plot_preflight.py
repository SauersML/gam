"""Plot the paired structural-preflight timing, not model training throughput."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "PAIRED_PREFLIGHT.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(9, 4.9), facecolor="#fafafa")
ax.set_facecolor("#fafafa")
times = [data["uncached_seconds"], data["cached_seconds"]]
bars = ax.barh([1, 0], times, height=0.48, color=["#969da8", "#147d92"])
ax.set_yticks([1, 0], ["Re-encode unchanged weights", "Reuse exact encoded weights"])
ax.set_xlim(0, 470)
ax.set_xlabel("Wall time (seconds; includes setup)")
for bar, seconds in zip(bars, times):
    ax.text(seconds + 8, bar.get_y() + bar.get_height() / 2,
            f"{seconds:.1f} s", va="center", fontweight="bold")
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
fig.suptitle(f"{data['speedup']:.2f}× faster structural preflight", x=0.06,
             ha="left", fontsize=21, fontweight="bold")
fig.text(0.06, 0.87, "128 proposals • identical proposal records • full artifact verification",
         fontsize=11, color="#3d4650")
fig.text(0.06, 0.055,
         "Same Mac and binary; one sequential run per arm. Concurrent activity was not controlled.\n"
         "No model fitting or behavioral evaluation: this is an engineering result, not mechanism discovery.",
         fontsize=9, color="#505861")
fig.subplots_adjust(left=0.37, right=0.96, top=0.77, bottom=0.24)
for suffix in ("png", "pdf"):
    fig.savefig(root / f"preflight_speed.{suffix}", dpi=180, facecolor=fig.get_facecolor())
