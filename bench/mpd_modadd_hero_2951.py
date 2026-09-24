"""#2951 hero image: "Addition, on a circle". Analysis only (SPEC 8 exception); every mark is the trained
network's own output or weights, nothing idealized or fitted.

Left disc, "turning": operand a walks once around its circle along the angle path (every embedding plane
turned by w_k t, t from 0 to p). Angle = output token c (2 pi c / p), radius = t, brightness = the network's
probability. Right disc, "fading": the same walk along straight segments between consecutive embedding rows.
Bottom: the embedding rows projected onto the five key planes, hue = token value.
"""
from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd_modadd_2951 import build_model
from mpd_modadd_paths_2951 import angle_table, planes_of

INK = "#07060b"
PAPER = "#ece7f5"
MUTED = "#8c86a3"


def walk(model, table, mean, planes, p, a, b, samples, kind):
    tokens = torch.tensor([[a, b, p]])
    ts = np.linspace(0, p, samples)
    out = []
    cyc = table.double()
    with torch.inference_mode():
        for t in ts:
            if kind == "angle":
                edited = angle_table(table, mean, planes, p, t)
            else:
                n, f = int(math.floor(t)) % p, t - math.floor(t)
                edited = cyc.clone()
                idx = torch.arange(p)
                edited[:p] = (1 - f) * cyc[(idx + n) % p] + f * cyc[(idx + n + 1) % p]
            out.append(torch.softmax(model(tokens, embed_at={0: edited}).double(), -1)[0].numpy())
    return ts, np.array(out)


def disc(ax, ts, probs, p, a, b, cmap, norm, title, subtitle):
    theta = 2 * np.pi * (np.arange(p + 1) - 0.5) / p
    radius = 0.22 + 0.78 * np.append(ts, ts[-1] + (ts[1] - ts[0])) / ts.max()
    T, R = np.meshgrid(theta, radius)
    mesh = ax.pcolormesh(T, R, np.log10(np.clip(probs, 1e-300, 1.0)), cmap=cmap, norm=norm, shading="flat",
                         rasterized=True)
    ax.set_ylim(0, 1.02)
    ax.set_facecolor(INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.text(0.5, -0.03, title, transform=ax.transAxes, ha="center", va="top", color=PAPER, fontsize=17,
            fontweight="light")
    ax.text(0.5, -0.085, subtitle, transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=9.5)
    return mesh


def clocks(fig, table, mean, planes, keys, p, box):
    hue = plt.get_cmap("twilight_shifted")
    left, bottom, width, height = box
    n = len(keys)
    for i, k in enumerate(keys):
        ax = fig.add_axes([left + i * width / n + 0.012, bottom, width / n - 0.024, height])
        ax.set_facecolor(INK)
        U = planes[k - 1]  # d x 2
        coords = torch.linalg.lstsq(U, (table[:p].double() - mean[None]).T).solution.T.numpy()  # p x 2
        coords = coords / np.median(np.linalg.norm(coords, axis=1))
        # colour = the token's position on the first key circle, (keys[0] * a mod p) / p: that circle reads as
        # a smooth wheel, and every other circle shows how its frequency re-deals the same colours
        colors = hue(((keys[0] * np.arange(p)) % p) / p)
        ax.scatter(coords[:, 0], coords[:, 1], s=12, c=colors, edgecolors="none", zorder=3)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, -1.62, f"k = {k}", ha="center", va="top", color=PAPER, fontsize=10, fontweight="light")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--early-step", type=int, default=2000)
    parser.add_argument("--bare", action="store_true", help="the two discs alone, no text but the step under each")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    p = run["config"]["p"]
    model = build_model(run["config"])
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    model = model.double().eval()
    table = model.W_E.detach().cpu().double()
    mean, planes = planes_of(table, p)
    power = planes.pow(2).sum((1, 2))
    keys = (torch.argsort(power, descending=True)[:5] + 1).tolist()
    a, b = 17, 40
    ts, turning = walk(model, table, mean, planes, p, a, b, args.samples, "angle")
    early = build_model(run["config"])
    early.load_state_dict(run["checkpoints"][args.early_step])
    early = early.double().eval()
    early_table = early.W_E.detach().cpu().double()
    early_mean, early_planes = planes_of(early_table, p)
    _, memorized = walk(early, early_table, early_mean, early_planes, p, a, b, args.samples, "angle")
    logs = np.log10(np.clip(np.concatenate([turning, memorized]), 1e-300, 1.0))
    norm = matplotlib.colors.Normalize(vmin=float(np.quantile(logs, 0.01)), vmax=0.0)
    cmap = plt.get_cmap("magma")
    if args.bare:
        fig = plt.figure(figsize=(16, 8.4), facecolor=INK)
        for left, probs, label in ((0.02, turning, "Grokked"), (0.51, memorized, "Memorized")):
            ax = fig.add_axes([left, 0.02, 0.47, 0.84], projection="polar")
            theta = 2 * np.pi * (np.arange(p + 1) - 0.5) / p
            radius = 0.22 + 0.78 * np.append(ts, ts[-1] + (ts[1] - ts[0])) / ts.max()
            T, R = np.meshgrid(theta, radius)
            ax.pcolormesh(T, R, np.log10(np.clip(probs, 1e-300, 1.0)), cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.set_ylim(0, 1.02)
            ax.set_facecolor(INK)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["polar"].set_visible(False)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.text(0.5, 1.035, label, transform=ax.transAxes, ha="center", va="bottom", color="white",
                    fontsize=40, fontweight="light")
        fig.savefig(args.out, dpi=220, facecolor=INK)
        print(f"[hero] wrote {args.out}", flush=True)
        return
    fig = plt.figure(figsize=(16, 11), facecolor=INK)
    ax1 = fig.add_axes([0.035, 0.32, 0.44, 0.555], projection="polar")
    ax2 = fig.add_axes([0.525, 0.32, 0.44, 0.555], projection="polar")
    mesh = disc(ax1, ts, turning, p, a, b, cmap, norm, f"after grokking  (step {max(run['checkpoints']):,})",
                "the answer rides the rotation: one unbroken spiral, its competitors in lockstep")
    disc(ax2, ts, memorized, p, a, b, cmap, norm, f"memorizing only  (step {args.early_step:,})",
         "the same continuous rotation of its own embedding circles: nothing follows it")
    cax = fig.add_axes([0.487, 0.45, 0.008, 0.34])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("log10 probability of each output token", color=MUTED, fontsize=8.5)
    bar.ax.tick_params(colors=MUTED, labelsize=7.5)
    bar.outline.set_visible(False)
    clocks(fig, table, mean, planes, keys, p, (0.14, 0.03, 0.72, 0.15))
    fig.text(0.5, 0.975, "Addition, on a circle", ha="center", va="top", color=PAPER, fontsize=30, fontweight="light")
    fig.text(0.5, 0.928, f"a one-layer transformer trained on (a + b) mod {p}, probed through its own parameters.  "
             f"Input {a} + {b}: operand a is turned continuously along its embedding circles, once around.",
             ha="center", va="top", color=MUTED, fontsize=10.5)
    fig.text(0.5, 0.905, "angle = output token c     radius = how far a has been turned (centre: not at all, rim: one full turn)"
             "     colour = log probability", ha="center", va="top", color=MUTED, fontsize=10.5)
    fig.text(0.5, 0.2, "the grokked embedding on its five key planes: colour = a token's place on the first circle; "
             "each frequency re-deals the same 113 tokens", ha="center", va="top", color=MUTED, fontsize=10)
    fig.savefig(args.out, dpi=220, facecolor=INK)
    print(f"[hero] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
