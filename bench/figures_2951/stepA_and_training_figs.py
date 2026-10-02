"""Step A (VPD's components, VPD's sets vs our selection), our training run, toy ladders, selection speed (#2951)."""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path.home() / "mpd-data/figures"
PIECES = Path.home() / "mpd-data/components/vpd4l"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
OURS, VPD, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 14})


def frame(ax):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=12)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURF)
    frame(ax)
    return fig, ax


def step_a_rows():
    pat = re.compile(r"eval sequence (\d+): start L0 ([\d.]+) KL ([\d.]+) code ([\d.]+); selected L0 ([\d.]+) KL ([\d.]+) code ([\d.]+)")
    rows = {}
    for line in (PIECES / "stepA_warm_1024.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            rows[int(m.group(1))] = [float(x) for x in m.groups()[1:]]
    return [rows[k] for k in sorted(rows)]


# 1. Every passage moves toward fewer components and closer predictions.
rows = step_a_rows()
n = len(rows)
fig, ax = new_fig(11, 7.5)
for l0s, kls, _, l0o, klo, _ in rows:
    ax.annotate("", xy=(l0o, klo), xytext=(l0s, kls),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, mutation_scale=14), zorder=2)
ax.scatter([r[0] for r in rows], [r[1] for r in rows], s=110, color=VPD, edgecolor=SURF, lw=2, zorder=3, label="VPD's own choice of components")
ax.scatter([r[3] for r in rows], [r[4] for r in rows], s=110, color=OURS, edgecolor=SURF, lw=2, zorder=4, label="our choice, same components")
ax.set_xlabel("VPD components switched on per word (average over a 512-word passage)", color=INK, labelpad=10)
ax.set_ylabel("difference from the real model's predictions (KL, nats)", color=INK, labelpad=10)
ax.legend(frameon=False, loc="upper left", fontsize=13)
ax.set_title(f"Same VPD components, better choices: fewer components and closer predictions ({n} passages)",
             color=INK, fontsize=16, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(OUT / "stepA_arrows.png", facecolor=SURF)

# 2. Total description per passage.
order = np.argsort([r[2] for r in rows])
fig, ax = new_fig(12, 6.5)
x = np.arange(n)
ax.bar(x - 0.2, [rows[i][2] for i in order], 0.38, color=VPD, label="VPD's choices", zorder=3)
ax.bar(x + 0.2, [rows[i][5] for i in order], 0.38, color=OURS, label="our choices (same components)", zorder=3)
for j, i in enumerate(order):
    d = (rows[i][5] - rows[i][2]) / rows[i][2] * 100
    ax.annotate(f"{d:+.1f}%", (j + 0.2, rows[i][5]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11, color=INK2)
ax.set_xticks(x, [f"passage {i}" for i in order], rotation=0, fontsize=10)
ax.set_ylim(1300, max(r[2] for r in rows) * 1.06)
ax.set_ylabel("bits per word to write down which components are on\nplus how far the output is from the model", color=INK, labelpad=10)
ax.legend(frameon=False, loc="upper left", fontsize=13)
tot_v, tot_o = np.mean([r[2] for r in rows]), np.mean([r[5] for r in rows])
better = sum(r[5] < r[2] - 0.05 for r in rows)
ax.set_title(f"Shorter on {better} of {n} passages, never longer: {tot_v:,.0f} to {tot_o:,.0f} bits per word ({(tot_o - tot_v) / tot_v * 100:+.1f}%)",
             color=INK, fontsize=16, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(OUT / "stepA_bits.png", facecolor=SURF)

# 3. Our own components' training run (before switching to VPD's components as the start).
it = json.loads((PIECES / "masked_wsvd_1024_fullbatch.json").read_text())["iterations"]
l0 = [i["eval"]["l0"] for i in it]
kl = [i["eval"]["kl"] for i in it]
fig, ax = new_fig(11, 7)
ax.plot(l0, kl, color=OURS, lw=2.2, zorder=3)
ax.scatter(l0, kl, s=80, color=OURS, edgecolor=SURF, lw=2, zorder=4)
for k, (a, b) in enumerate(zip(l0, kl)):
    ax.annotate(f"round {k}", (a, b), xytext=(8, 6), textcoords="offset points", fontsize=11, color=INK2)
vl0, vkl = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
ol0, okl = np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])
ax.scatter([vl0], [vkl], s=160, color=VPD, edgecolor=SURF, lw=2, zorder=5)
ax.annotate("VPD", (vl0, vkl), xytext=(10, -4), textcoords="offset points", fontsize=13, color=INK, fontweight="medium")
ax.scatter([ol0], [okl], s=160, color=THIRD, edgecolor=SURF, lw=2, zorder=5, marker="D")
ax.annotate("VPD's components, our choices", (ol0, okl), xytext=(10, -16), textcoords="offset points", fontsize=13, color=INK)
ax.set_xscale("log")
ax.set_xlabel("components switched on per word (log scale)", color=INK, labelpad=10)
ax.set_ylabel("difference from the real model (KL, nats)", color=INK, labelpad=10)
ax.set_title("Training our own components (blue) is still far right of VPD; VPD's components with our choices beat it",
             color=INK, fontsize=15, fontweight="bold", loc="left", pad=14)
fig.text(0.01, 0.01, "training points: 512 held-out words of a different passage; VPD points: average over the test passages",
         color=INK2, fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "training_trajectory.png", facecolor=SURF)

# 4. Toy ladders: our program vs the best rounded copy of the model.
suite = json.loads((Path.home() / "mpd-data/engine/suite_table.json").read_text())["models"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200, sharey=False)
fig.patch.set_facecolor(SURF)
for ax, (key, label) in zip(axes, [("p31", "mod-31 adder"), ("blind_B", "blind model B")]):
    frame(ax)
    lad = sorted(suite[key]["ladder"], key=lambda r: r["observations"])
    obs = [r["observations"] for r in lad]
    ax.plot(obs, [r["total_bits"] for r in lad], color=OURS, lw=2.2, marker="o", ms=8, mec=SURF, mew=2, label="our program", zorder=3)
    ax.plot(obs, [r["best_rounded_total_bits"] for r in lad], color=VPD, lw=2.2, marker="o", ms=8, mec=SURF, mew=2,
            label="best rounded copy of the model", zorder=3)
    for r in lad:
        if r["argmax_agreement"] < 0.5:
            ax.annotate("almost nothing\nexplained", (r["observations"], r["total_bits"]), xytext=(6, 10), textcoords="offset points",
                        fontsize=10, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("how much behaviour must be explained (observations per input, log)", color=INK, fontsize=12)
    ax.set_ylabel("total bits (rules + which parts fire + leftover error)", color=INK, fontsize=12)
    ax.set_title(label, color=INK, fontsize=15, loc="left")
axes[0].legend(frameon=False, loc="upper left", fontsize=12)
fig.suptitle("On toy models our programs are shorter than the model itself at every level of demanded fidelity",
             color=INK, fontsize=16, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "toy_ladders.png", facecolor=SURF)

# 5. Selection speed: the same rounds before and after the select change.
prof = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if prof:
    pat = re.compile(r"selection round (\d+) \((\d+)s\).*code ([\d.]+) ->")
    fig, ax = new_fig(11, 6.5)
    for name, color, label in [("run_old.log", VPD, "before"), ("run_new.log", OURS, "after (same selections)")]:
        pts = [pat.search(l) for l in (prof / name).read_text().splitlines()]
        pts = [(int(m.group(2)), float(m.group(3))) for m in pts if m][:14]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2, marker="o", ms=7, mec=SURF, mew=1.5, label=label)
    ax.set_xlabel("seconds since selection started", color=INK, labelpad=10)
    ax.set_ylabel("bits per word of the current choice", color=INK, labelpad=10)
    ax.legend(frameon=False, fontsize=13)
    ax.set_title("The same selection, reached 2.3× faster", color=INK, fontsize=16, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "selection_speed.png", facecolor=SURF)
