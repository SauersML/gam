"""#2951 movie: modular addition from initialization through memorization to grokking. Analysis only (SPEC 8
exception); every mark is the trained network's own output or weights at that checkpoint.

* left, the answer field: for every pair (a, b) of the 113 x 113 table, log10 of the probability the network puts on
  (a + b) mod p. 30% of the cells are the training set; the rest were never seen;
* right top, the embedding on its plane of strongest frequency k* (the frequency with the most power at the end of
  training): each token a is a dot, hue = a. Axes rescale per frame, so the panel shows shape, not size;
* right middle, the embedding's power at each frequency k = 1..(p-1)/2, as a share of the total;
* right bottom, train and test loss against step, with the current step marked.
"""
from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd_modadd_2951 import all_pairs, build_model

INK = "#07060b"
PAPER = "#ece7f5"
MUTED = "#8c86a3"
TRAIN = "#f0a35e"
TEST = "#7ab8f5"


def measure(run):
    config = run["config"]
    p = config["p"]
    tokens = all_pairs(p)
    labels = run["labels"]
    model = build_model(config).double().eval()
    steps = sorted(run["checkpoints"])
    k = torch.arange(1, (p - 1) // 2 + 1, dtype=torch.float64)
    a = torch.arange(p, dtype=torch.float64)
    basis = torch.exp(-2j * math.pi * k[:, None] * a[None, :] / p)  # K x p
    frames = []
    for step in steps:
        model.load_state_dict({n: t.double() for n, t in run["checkpoints"][step].items()})
        with torch.inference_mode():
            logp = torch.log_softmax(model(tokens).double(), -1)
        field = logp.gather(1, labels[:, None])[:, 0].reshape(p, p).numpy() / math.log(10)
        table = model.W_E.detach()[:p].double()
        centred = table - table.mean(0)
        F = basis @ centred.to(torch.complex128)  # K x d
        power = (F.abs() ** 2).sum(-1)
        frames.append({"step": step, "field": field, "F": F.numpy(), "centred": centred.numpy(),
                       "power": (power / power.sum()).numpy()})
    kstar = int(np.argmax(frames[-1]["power"]))
    for fr in frames:
        u, v = fr["F"][kstar].real, fr["F"][kstar].imag
        u = u / np.linalg.norm(u)
        v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
        fr["xy"] = np.stack([fr["centred"] @ u, fr["centred"] @ v], 1)
        del fr["F"], fr["centred"]
    return frames, kstar + 1, p


def render(job):
    i, fr, kstar, p, curves, mask, out_dir, n_frames = job
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=INK)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.78], left=0.03, right=0.97, top=0.93, bottom=0.07,
                            wspace=0.1, hspace=0.38)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(fr["field"], cmap="magma", vmin=-8, vmax=0, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("b", color=MUTED, fontsize=15)
    ax.set_ylabel("a", color=MUTED, fontsize=15, rotation=0, labelpad=14)
    ax.set_title("log₁₀ p( (a + b) mod 113 )", color=PAPER, fontsize=17, pad=12)
    acc = fr["field"].reshape(-1)
    ax.text(0.0, -0.045, f"train cells {np.mean(acc[mask] > math.log10(0.5)):.0%} right   "
            f"never-seen cells {np.mean(acc[~mask] > math.log10(0.5)):.0%} right",
            transform=ax.transAxes, color=MUTED, fontsize=13, va="top")

    cax = fig.add_subplot(grid[0, 1])
    xy = fr["xy"]
    lim = np.quantile(np.linalg.norm(xy, axis=1), 0.99) * 1.15 + 1e-12
    cax.scatter(xy[:, 0], xy[:, 1], c=np.arange(p), cmap="twilight_shifted", s=22, linewidths=0)
    cax.set_xlim(-lim, lim)
    cax.set_ylim(-lim, lim)
    cax.set_aspect("equal")
    cax.set_facecolor(INK)
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.set_title(f"token embeddings, frequency-{kstar} plane", color=PAPER, fontsize=14)

    sax = fig.add_subplot(grid[1, 1])
    ks = np.arange(1, len(fr["power"]) + 1)
    sax.bar(ks, fr["power"], color=PAPER, width=0.8)
    sax.set_ylim(0, 0.35)
    sax.set_xlim(0, len(ks) + 1)
    sax.set_facecolor(INK)
    sax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        sax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        sax.spines[s].set_color(MUTED)
    sax.set_title("embedding power by frequency k", color=PAPER, fontsize=14)

    lax = fig.add_subplot(grid[2, 1])
    st = np.maximum(np.array(curves["step"]), 1)
    lax.plot(st, curves["train_loss"], color=TRAIN, lw=1.6, label="train")
    lax.plot(st, curves["test_loss"], color=TEST, lw=1.6, label="never seen")
    lax.axvline(max(fr["step"], 1), color=PAPER, lw=1.0)
    lax.set_xscale("log")
    lax.set_yscale("log")
    lax.set_facecolor(INK)
    lax.tick_params(colors=MUTED, labelsize=10)
    for s in ("top", "right"):
        lax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        lax.spines[s].set_color(MUTED)
    lax.set_xlabel("step", color=MUTED, fontsize=12)
    lax.set_title("loss", color=PAPER, fontsize=14)
    lax.legend(frameon=False, labelcolor=PAPER, fontsize=11, loc="lower left")
    fig.suptitle(f"step {fr['step']:,}", color=PAPER, fontsize=22, x=0.03, ha="left")
    fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"), facecolor=INK)
    plt.close(fig)
    return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    frames, kstar, p = measure(run)
    mask = np.zeros(p * p, dtype=bool)
    mask[run["train_idx"].numpy()] = True
    os.makedirs(args.frames_dir, exist_ok=True)
    jobs = [(i, fr, kstar, p, run["curves"], mask, args.frames_dir, len(frames)) for i, fr in enumerate(frames)]
    with Pool(args.workers) as pool:
        for i in pool.imap_unordered(render, jobs):
            if i % 50 == 0:
                print(f"[movie] frame {i}/{len(frames)}", flush=True)
    print(f"[movie] {len(frames)} frames, k*={kstar}", flush=True)


if __name__ == "__main__":
    main()
