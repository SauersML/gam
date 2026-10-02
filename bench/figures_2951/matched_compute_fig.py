"""Step A at matched compute: how much shorter our selection makes VPD's explanation per GFLOP per word (#2951).

Per word, VPD chooses its subcomponents with one pass of its causal-importance network (≈0.54B
parameters, ≈1 GFLOP). Our selection starts from that choice and searches: each round runs the
masked model forward, one mask-gradient reverse pass (≈2 forwards), and the proposal's forward,
so ≈4 masked forwards. A masked forward's FLOPs per word are counted from the run's own sites
(`2·C·(d_in + d_out)` per site for `z = Vᵀx` and `U z`, plus the unembedding).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path.home() / "mpd-data/pieces/vpd4l/stepA_warm_1024.log"
OUT = Path.home() / "mpd-data/figures/matched_compute.png"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
OURS, VPD = "#2a78d6", "#eb6834"
D_MODEL, VOCAB = 768, 50277
GAMMA_GFLOP = 2 * 0.54  # one pass of VPD's causal-importance network, ≈2 FLOPs per parameter

text = LOG.read_text()
sites = re.findall(r"(\d+)×(\d+), (\d+) given pieces", text)
forward_flop = sum(2 * int(c) * (int(a) + int(b)) for a, b, c in sites) + 2 * D_MODEL * VOCAB
round_gflop = 4 * forward_flop / 1e9

# Per sequence: the current code at the start of every round, then the final selected code.
sequences, current = [], []
for line in text.splitlines():
    m = re.search(r"selection round (\d+) .*code ([\d.]+) ->", line)
    if m:
        current.append(float(m.group(2)))
        continue
    m = re.search(r"eval sequence \d+: start L0 [\d.]+ KL [\d.]+ code ([\d.]+); selected L0 [\d.]+ KL [\d.]+ code ([\d.]+)", line)
    if m:
        start, final = float(m.group(1)), float(m.group(2))
        sequences.append([start] + current[1:] + [final] if current else [start, final])
        current = []

rounds = max(len(s) for s in sequences)
gain = np.array([[(s[0] - s[min(i, len(s) - 1)]) / s[0] * 100 for i in range(rounds)] for s in sequences])
x = np.arange(rounds) * round_gflop

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 14})
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=12)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for row in gain:
    ax.plot(x, row, color=OURS, lw=0.8, alpha=0.25, zorder=2)
ax.plot(x, gain.mean(axis=0), color=OURS, lw=2.6, zorder=3, label=f"our search, average over {len(sequences)} passages")
ax.axvline(GAMMA_GFLOP, color=VPD, lw=2, ls="--", zorder=2)
ax.annotate("cost of VPD's own choice\n(one pass of its 0.54B-parameter network)", (GAMMA_GFLOP, ax.get_ylim()[1] * 0.92),
            xytext=(8, 0), textcoords="offset points", color=INK, fontsize=12, va="top")
ax.axhline(0, color=INK2, lw=1)
ax.set_xlabel("extra compute per word spent improving VPD's choice (GFLOP, ≈)", color=INK, labelpad=10)
ax.set_ylabel("how much shorter the explanation is than VPD's (%)", color=INK, labelpad=10)
ax.legend(frameon=False, loc="lower right", fontsize=13)
ax.set_title("At VPD's own cost per word our search gains nothing yet; its 4% needs ~15–20× more compute",
             color=INK, fontsize=14.5, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(OUT, facecolor=SURF)
print(f"masked forward {forward_flop / 1e9:.3f} GFLOP/word, round {round_gflop:.2f} GFLOP/word, "
      f"gain at Γ cost {np.interp(GAMMA_GFLOP, x, gain.mean(axis=0)):.2f}%, final {gain.mean(axis=0)[-1]:.2f}%")
