"""The engine's blind factors of a mod-31 adder: inputs and answer sit on the same three circles (#2951)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

d = json.load(open("/Users/user/mpd-data/engine/p31_s0/factors.json"))
p = d["p"]


def dominant(v):
    spectrum = np.abs(np.fft.rfft(np.asarray(v) - np.mean(v)))
    spectrum[0] = 0
    return int(np.argmax(spectrum))


def planes(vectors):
    """Pair the two strongest vectors at each frequency; keep the three strongest planes."""
    by_k = {}
    for weight, v in vectors:
        by_k.setdefault(dominant(v), []).append((weight, np.asarray(v, float)))
    pairs = [(k, sorted(vs, key=lambda t: -t[0])[:2]) for k, vs in by_k.items() if len(vs) >= 2]
    return {k: (vs[0][1], vs[1][1]) for k, vs in sorted(pairs, key=lambda kv: -sum(w for w, _ in kv[1]))[:3]}


reads = planes([(f["coefficient_energy"], f["over_a"]) for f in d["read_factors"]])
writes = planes([(w["energy_share"], w["over_class"]) for w in d["writes_over_classes"]])
ks = [k for k in reads if k in writes]
print("read planes", list(reads), "write planes", list(writes), "shared", ks)


def whiten(x, y):
    x, y = x - x.mean(), y - y.mean()
    cov = np.cov(np.vstack([x, y]))
    # 2x2 symmetric eigendecomposition in closed form (no LAPACK/Accelerate call).
    a_, b_, c_ = cov[0, 0], cov[0, 1], cov[1, 1]
    mid, rad = (a_ + c_) / 2, np.hypot((a_ - c_) / 2, b_)
    w = np.array([mid - rad, mid + rad])
    theta = 0.5 * np.arctan2(2 * b_, a_ - c_)
    vecs = np.array([[-np.sin(theta), np.cos(theta)], [np.cos(theta), np.sin(theta)]])
    return (vecs.T @ np.vstack([x, y])) / np.sqrt(w)[:, None]


SURF, INK, INK2, RING = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 15})
fig, axes = plt.subplots(2, len(ks), figsize=(5.6 * len(ks) + 1.6, 11.8), dpi=200, squeeze=False)
fig.patch.set_facecolor(SURF)
colors = cm.viridis(np.linspace(0.05, 0.95, p))
t = np.linspace(0, 2 * np.pi, 400)
for row, (source, label) in enumerate([(reads, "number going in, a"), (writes, "answer coming out, (a + b) mod 31")]):
    for col, k in enumerate(ks):
        ax = axes[row, col]
        z = whiten(*source[k])
        ax.set_facecolor(SURF)
        ax.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=RING, lw=10, zorder=1)
        ax.scatter(z[0], z[1], s=330, c=colors, edgecolor=SURF, linewidth=2, zorder=2)
        for a in range(p):
            ax.annotate(str(a), (z[0, a], z[1, a]), ha="center", va="center", fontsize=9.5,
                        color="white", fontweight="bold", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"frequency {k}", color=INK, fontsize=17, pad=4)
    axes[row, 0].annotate(label, xy=(-0.1, 0.5), xycoords="axes fraction", rotation=90,
                          ha="center", va="center", color=INK, fontsize=16)
fig.suptitle("Without being told, the engine finds the adder's clocks: numbers go in on three circles,\n"
             "and the answer comes out on the same three", color=INK, fontsize=20, fontweight="bold",
             x=0.02, ha="left", y=0.985)
fig.text(0.02, 0.015, "each dot is a number 0–30, placed by two directions the engine discovered from the weights",
         color=INK2, fontsize=13, ha="left")
fig.tight_layout(rect=(0.03, 0.03, 1, 0.92))
fig.savefig("/Users/user/mpd-data/figures/p31_circles.png", facecolor=SURF)
