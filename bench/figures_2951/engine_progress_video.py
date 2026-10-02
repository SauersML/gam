"""Our engine's per-token point on VPD 4L moving toward VPD's curve across iterations (#2951)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

F = Path.home() / "mpd-data/frontier"
P = Path.home() / "mpd-data/pieces/vpd4l"
vpd = json.load(open(F / "pertoken_vpd4l_vpd.json"))
bases = json.load(open(F / "pertoken_vpd4l_bases.json"))
its = json.load(open(P / "summary.json"))["fullbatch_wsvd_n1024"]
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
ORANGE, AQUA = "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS"], "font.size": 16})


def curve(ps):
    ps = sorted((p for p in ps if p["kl"] > 1e-6 and p["bits"] > 0), key=lambda p: p["l0"])
    return np.array([p["bits"] for p in ps]), np.array([p["kl"] for p in ps])


vx, vy = curve(vpd["rounded"])
wx, wy = curve(bases["wsvd"]["points"])


def vpd_bits_at(kl):
    """VPD's bits per token where its curve reaches this KL (log-log interpolation)."""
    o = np.argsort(vy)
    return float(np.exp(np.interp(np.log(kl), np.log(vy[o]), np.log(vx[o]))))


ex = np.array([p["bits"] for p in its])
ey = np.array([p["kl"] for p in its])
FPS, STEP, HOLD0, HOLD1 = 30, 36, 30, 90
frames = HOLD0 + STEP * (len(its) - 1) + HOLD1

fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
fig.patch.set_facecolor(SURF)
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)
ax.set_facecolor(SURF)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(900, 16000)
ax.set_ylim(0.3, 1.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
ax.xaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 8000, 16000]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["1,000", "2,000", "4,000", "8,000", "16,000"])
ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1.0, 1.5]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["0.3", "0.5", "1.0", "1.5"])
ax.tick_params(colors=INK2, labelsize=13)
ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("bits per token to say which subcomponents are on", color=INK, labelpad=10)
ax.set_ylabel("difference from the model's outputs (KL)", color=INK, labelpad=10)

ax.plot(vx, vy, color=ORANGE, lw=2.4, zorder=3, solid_capstyle="round")
ax.scatter(vx, vy, s=40, color=ORANGE, edgecolor=SURF, linewidth=1.6, zorder=4)
ax.text(vx[0] * 0.95, vy[0], "VPD", color=INK, fontsize=16, ha="right", va="center", fontweight="medium")
m = (wx > 800) & (wx < 20000)
ax.plot(wx[m], wy[m], color=AQUA, lw=2.4, zorder=3, solid_capstyle="round")
ax.scatter(wx[m], wy[m], s=40, color=AQUA, edgecolor=SURF, linewidth=1.6, zorder=4)
ax.text(8849 * 1.06, 1.41, "best fixed basis", color=INK, fontsize=15, ha="left", va="center")

trail, = ax.plot([], [], color=INK, lw=1.2, alpha=0.35, zorder=5)
past = ax.scatter([], [], s=36, color=INK, alpha=0.3, edgecolor=SURF, linewidth=1.2, zorder=5)
dot = ax.scatter([], [], s=150, color=INK, edgecolor=SURF, linewidth=2.5, zorder=7)
gap, = ax.plot([], [], color=INK2, lw=1.1, zorder=2)
lab = ax.text(0, 0, "", color=INK, fontsize=15, ha="left", va="center", zorder=8, fontweight="medium")
ratio = ax.text(0, 0, "", color=INK2, fontsize=13.5, ha="center", va="bottom", zorder=8)
fig.suptitle("Our engine closing in on VPD", color=INK, fontsize=22, fontweight="bold", x=0.035, ha="left", y=0.965)
counter = fig.text(0.035, 0.885, "", color=INK2, fontsize=15, ha="left")


def ease(u):
    return u * u * (3 - 2 * u)


def draw(f):
    k = min(max(f - HOLD0, 0), STEP * (len(its) - 1))
    i, u = divmod(k, STEP)
    if i >= len(its) - 1:
        i, u = len(its) - 1, 0
    t = ease(u / STEP)
    j = min(i + 1, len(its) - 1)
    x = float(np.exp((1 - t) * np.log(ex[i]) + t * np.log(ex[j])))
    y = float(np.exp((1 - t) * np.log(ey[i]) + t * np.log(ey[j])))
    trail.set_data(list(ex[:i + 1]) + [x], list(ey[:i + 1]) + [y])
    past.set_offsets(np.c_[ex[:i + 1], ey[:i + 1]] if i > 0 or t > 0 else np.empty((0, 2)))
    dot.set_offsets([[x, y]])
    lab.set_position((x * 1.07, y))
    lab.set_text("our engine")
    vb = vpd_bits_at(y)
    gap.set_data([vb * 1.04, x / 1.06], [y, y])
    ratio.set_position((np.sqrt(vb * x), y * 1.03))
    ratio.set_text(f"{x / vb:.1f}× VPD's bits")
    shown = i + (1 if t > 0.5 else 0)
    counter.set_text(f"iteration {shown} of {len(its) - 1}   ·   per-token code on VPD's 4-layer model")


writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
out = Path.home() / "mpd-data/figures/engine_progress_vpd4l.mp4"
with writer.saving(fig, str(out), dpi=150):
    for f in range(frames):
        draw(f)
        writer.grab_frame(facecolor=SURF)
draw(frames - 1)
fig.savefig(out.with_suffix(".png"), facecolor=SURF)
print(out)
