"""#2951 movie: the "Addition, on a circle" disc at every checkpoint of one training run, from initialization through
memorization to grokking. Analysis only (SPEC 8 exception); every frame is that checkpoint's own output.

Per frame, input a + b with a = 17, b = 40: operand a is turned continuously along the checkpoint's OWN embedding
planes (``mpd_modadd_hero_2951.walk``, angle path), once around. Angle = output token c, radius = how far a has
been turned (centre: not at all, rim: one full turn), colour = log10 probability, on one colour scale for all frames.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd_modadd_2951 import build_model
from mpd_modadd_hero_2951 import INK, walk
from mpd_modadd_paths_2951 import planes_of

A, B = 17, 40
VMIN = -12.0


def frame(job):
    i, step, state, config, samples, out_dir = job
    torch.set_num_threads(1)
    p = config["p"]
    model = build_model(config)
    model.load_state_dict(state)
    model = model.double().eval()
    table = model.W_E.detach().cpu().double()
    mean, planes = planes_of(table, p)
    ts, probs = walk(model, table, mean, planes, p, A, B, samples, "angle")
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=INK)
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.88], projection="polar")
    theta = 2 * np.pi * (np.arange(p + 1) - 0.5) / p
    radius = 0.22 + 0.78 * np.append(ts, ts[-1] + (ts[1] - ts[0])) / ts.max()
    T, R = np.meshgrid(theta, radius)
    ax.pcolormesh(T, R, np.log10(np.clip(probs, 1e-300, 1.0)), cmap="magma", vmin=VMIN, vmax=0.0, shading="flat",
                  rasterized=True)
    ax.set_ylim(0, 1.02)
    ax.set_facecolor(INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    fig.text(0.5, 0.965, f"step {step:,}", ha="center", va="top", color="white", fontsize=30, fontweight="light")
    fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"), facecolor=INK)
    plt.close(fig)
    return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    steps = sorted(run["checkpoints"])
    os.makedirs(args.frames_dir, exist_ok=True)
    jobs = [(i, s, run["checkpoints"][s], run["config"], args.samples, args.frames_dir) for i, s in enumerate(steps)]
    with Pool(args.workers) as pool:
        for i in pool.imap_unordered(frame, jobs):
            if i % 50 == 0:
                print(f"[disc] frame {i}/{len(jobs)}", flush=True)
    print(f"[disc] {len(jobs)} frames", flush=True)


if __name__ == "__main__":
    main()
