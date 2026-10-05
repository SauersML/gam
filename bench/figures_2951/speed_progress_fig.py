"""Training-step throughput of the library fit on one RTX 4090 over 5 October 2026 (#2951).

(a) vpd4l, 8 base sequences per step, 8,192 scored tokens per step: scored tokens per second at each
stage (speed's measurements: host posterior 1.063 s/step, device posterior 0.233 s, decoder engine
0.070 s with patch directions excluded). (b) Qwen3-0.6B,
28 layers, 4 x 512 tokens, decoder engine alone: share of the card's dense bf16 peak (165 TFLOPS).

    python bench/figures_2951/speed_progress_fig.py OUT.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
INK, MUTED, BLUE, GRAY = "#1f1f1e", "#6b6b68", "#2a78d6", "#b4b4b1"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 17,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})

stages = [("posterior\non the CPU", 8192 / 1.063), ("posterior\non the GPU", 8192 / 0.233), ("decoder engine\n(patch directions\nexcluded)", 8192 / 0.070)]
xs = range(len(stages))
a.bar(xs, [v / 1e3 for _, v in stages], 0.66, color=[GRAY, GRAY, BLUE], edgecolor="white", linewidth=2)
for x, (_, v) in zip(xs, stages):
    a.text(x, v / 1e3 + 2, f"{v / 1e3:.0f}k", ha="center", va="bottom")
a.set_xticks(list(xs))
a.set_xticklabels([n for n, _ in stages], fontsize=14)
a.tick_params(axis="x", length=0)
a.set_ylabel("scored tokens per second")
a.set_ylim(0, 135)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
a.text(-0.12, 1.02, "a", transform=a.transAxes, fontsize=20, fontweight="bold")

passes = [("forward", 0.47), ("reverse", 0.38), ("both", 0.41)]
b.bar(range(3), [p * 100 for _, p in passes], 0.6, color=BLUE, edgecolor="white", linewidth=2)
for x, (_, p) in enumerate(passes):
    b.text(x, p * 100 + 1, f"{p * 100:.0f}%", ha="center", va="bottom")
b.set_xticks(range(3))
b.set_xticklabels([n for n, _ in passes])
b.tick_params(axis="x", length=0)
b.set_ylabel("share of the GPU's bf16 peak")
b.set_ylim(0, 60)
b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
b.text(-0.2, 1.02, "b", transform=b.transAxes, fontsize=20, fontweight="bold")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(out)
