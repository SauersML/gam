"""Figures for the #2951 modadd receipts: training curves, the P2 angle-versus-mask paths, and the
exact-versus-adversary ablation ladder. Analysis only (SPEC 8 exception); reads the JSON receipts and
the harvested run, writes PNGs."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def curves(run_paths, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, path in run_paths.items():
        run = torch.load(path, map_location="cpu", weights_only=True)
        c = run["curves"]
        for ax, key in zip(axes, ("acc", "loss")):
            ax.plot(c["step"], c[f"train_{key}"], label=f"{label} train", lw=1.2)
            ax.plot(c["step"], c[f"test_{key}"], label=f"{label} test", lw=1.2, ls="--")
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("cross-entropy (nats)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.suptitle("#2951 modadd benchmark, p=113: addition groks, random labels memorize")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def paths(report, out):
    freqs = report["output_key_frequencies"][:4]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, site in enumerate(("pos0", "global")):
        for path, color in (("angle", "C0"), ("mask", "C3")):
            rows = report["paths"][f"{site}/{path}"]
            t = [r["t"] for r in rows]
            axes[row, 0].plot(t, [r["max_prob_mean"] for r in rows], color=color, label=f"{path} path")
            axes[row, 0].plot(t, [r["argmax_hits_nearest_integer"] for r in rows], color=color, ls=":", label=f"{path}: argmax = nearest shift")
            for j, f in enumerate(freqs):
                axes[row, 1].plot(t, [r["amplitude_ratio_median"][j] for r in rows], color=color, alpha=1 - 0.2 * j,
                                  label=f"{path} k={f}" if j < 2 else None)
                axes[row, 1].plot(t, [r["expected_amplitude"][j] for r in rows], color=color, ls="--", alpha=0.4)
                axes[row, 2].plot(t, [r["phase_shift_median"][j] for r in rows], color=color, alpha=1 - 0.2 * j,
                                  label=f"{path} k={f}" if j < 2 else None)
                axes[row, 2].plot(t, [r["expected_phase"][j] for r in rows], color=color, ls="--", alpha=0.4)
        axes[row, 0].set_ylabel(f"{site}\nmean max prob")
        axes[row, 1].set_ylabel("output harmonic amplitude ratio")
        axes[row, 2].set_ylabel("output harmonic phase shift (rad)")
        for ax in axes[row]:
            ax.set_xlabel("t (fraction of a one-position shift)")
            ax.legend(fontsize=7)
    fig.suptitle("P2 on the grokked transformer: turning every embedding plane by w_k t (angle) vs the straight "
                 "segment to the shifted table (mask). Dashed: no-parameter predictions")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def adversary(report, out):
    ladder = report["ladder"]
    F = [r["F"] for r in ladder]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    ax.plot(F, [r["exact_shared"] for r in ladder], "k-o", label="exact sup, batch-shared mask (vertex enumeration)")
    ax.plot(F, [r["exact_per_input"]["q50"] for r in ladder], "k--s", label="exact sup per input, median")
    ax.plot(F, [r["exact_per_input"]["max"] for r in ladder], "k:^", label="exact sup per input, max")
    for n, c in (("20", "C1"), ("80", "C2"), ("320", "C3")):
        ax.plot(F, [r["pgd_shared"][n]["q50"] for r in ladder], color=c, marker="o", label=f"PGD {n} steps (median of restarts)")
    ax.plot(F, [r["fw_shared"]["q50"] for r in ladder], color="C0", marker="d", label="Frank-Wolfe, vertex LMO")
    ax.plot(F, [r["uniform_mean"] for r in ladder], color="grey", marker="x", label="uniform-mask mean (a law moment)")
    ax.set_yscale("log")
    ax.set_xlabel("F = number of free (non-key) unembedding planes")
    ax.set_ylabel("KL(target || ablated) nats")
    ax.set_title(f"Worst ablation of non-key W_U planes, keys {report['keys']} kept")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def structure(report, out):
    steps = sorted(report["checkpoints"], key=int)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for step in steps:
        hist = report["checkpoints"][step]["redundancy_embed"]["smallest_sufficient_size_hist"]
        total = sum(hist.values())
        sizes = sorted(int(k) for k in hist)
        axes[0].plot(sizes, [hist[str(s)] / total for s in sizes], marker="o",
                     label=f"step {step} (acc {report['checkpoints'][step]['test_acc']:.2f})")
    axes[0].set_xlabel("size of the smallest sufficient set of W_E planes (top 8)")
    axes[0].set_ylabel("fraction of test pairs")
    axes[0].set_title("Memorization needs every plane; the grokked code tolerates deletions")
    axes[0].legend(fontsize=7)
    last = report["checkpoints"][steps[-1]]
    deciles = last["margin_law"]["deciles"]
    mid = [(d["margin_lo"] + d["margin_hi"]) / 2 for d in deciles]
    axes[1].semilogy(mid, [d["sup_median"] for d in deciles], "o-", label="median per decile")
    axes[1].semilogy(mid, [d["sup_max"] for d in deciles], "^--", label="max per decile")
    a, b = last["margin_law"]["fit_log_sup_eq_a_plus_b_margin"]
    xs = [min(mid), max(mid)]
    axes[1].semilogy(xs, [2.718281828 ** (a + b * x) for x in xs], "k:", label=f"fit log sup = {a:.2f} {b:+.2f} margin")
    axes[1].set_xlabel("logit margin of the test pair")
    axes[1].set_ylabel("exact worst deletion of 16 non-key W_U planes (nats)")
    axes[1].set_title(f"Margin law, Spearman {last['margin_law']['spearman_logsup_vs_margin']:.2f}")
    axes[1].legend(fontsize=7)
    freqs = last["interactions"]["output_frequencies"]
    planes = list(last["interactions"]["turn_pi_over_2"].keys())
    grid = [[abs(v) for v in last["interactions"]["turn_pi_over_2"][k]["angle"]] for k in planes]
    im = axes[2].imshow(grid, cmap="viridis", vmin=0, vmax=3.1416 / 2)
    axes[2].set_xticks(range(len(freqs)), [str(f) for f in freqs])
    axes[2].set_yticks(range(len(planes)), planes)
    axes[2].set_xlabel("output harmonic")
    axes[2].set_ylabel("embedding plane turned by pi/2 (operand a)")
    axes[2].set_title("|phase shift| of each output harmonic")
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    curves({lab: os.path.join(args.harvest, f"{lab}_s0.pt") for lab in ("addition", "random")},
           os.path.join(args.out_dir, "curves.png"))
    for name, fn in (("paths_addition", paths), ("adversary_addition", adversary), ("structure_addition", structure)):
        path = os.path.join(args.results, f"{name}.json")
        if os.path.exists(path):
            with open(path) as handle:
                fn(json.load(handle), os.path.join(args.out_dir, f"{name}.png"))
    print("[plots] done", flush=True)


if __name__ == "__main__":
    main()
