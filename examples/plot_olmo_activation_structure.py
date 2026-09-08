#!/usr/bin/env python3
"""Plot the structure of real OLMo-3-32B residual-stream activations.

Input is the banked OLMo activation tensor used by the SAE research battery:

    activations.npy  float32, shape [prompts, token_positions, d_model]

For the current OLMo-3-32B bank this is [635, 64, 5120]. The plot samples
token-position activation vectors, computes PCA on that sample, and shows:

  * cumulative PCA explained variance,
  * PC1/PC2 colored by token position,
  * PC1/PC2 colored by activation L2 norm,
  * activation L2 norm by token position.

The plotted "activation vector length" is ||x||_2, not the dimension count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, help="Path to OLMo activations.npy")
    ap.add_argument("--out", default="olmo_structure_clean.png", help="Output PNG path")
    ap.add_argument("--sample", type=int, default=4000, help="Token activations to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pcs-shown", type=int, default=80)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 11,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#0e1116",
            "figure.facecolor": "#0e1116",
            "savefig.facecolor": "#0e1116",
            "text.color": "#e6e6e6",
            "axes.labelcolor": "#cfd3da",
            "xtick.color": "#aab0bb",
            "ytick.color": "#aab0bb",
            "axes.titlecolor": "#f2f4f8",
            "axes.edgecolor": "#3a4252",
        }
    )

    acts = np.load(args.activations)
    if acts.ndim != 3:
        raise SystemExit(f"expected [prompts, token_positions, d_model], got {acts.shape}")
    prompts, token_positions, d_model = acts.shape

    x = acts.reshape(-1, d_model).astype(np.float64)
    pos = np.tile(np.arange(token_positions), prompts)
    length = np.linalg.norm(x, axis=1)

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.sample, x.shape[0])
    idx = rng.choice(x.shape[0], n_sample, replace=False)

    sub = x[idx]
    centered = sub - sub.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    ev = s**2 / np.sum(s**2)
    cum = np.cumsum(ev)
    z = centered @ vt.T

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    shown = min(args.pcs_shown, len(cum))
    ax[0, 0].plot(np.arange(1, shown + 1), cum[:shown], color="#ffd166", lw=2.4)
    ax[0, 0].set_xlabel("PCA components")
    ax[0, 0].set_ylabel("explained variance")
    ax[0, 0].set_ylim(0, 1)

    by_pos = ax[0, 1].scatter(
        z[:, 0],
        z[:, 1],
        c=pos[idx],
        cmap="viridis",
        s=11,
        alpha=0.85,
        edgecolors="none",
    )
    fig.colorbar(by_pos, ax=ax[0, 1], label="token position")
    ax[0, 1].set_xlabel("PC1")
    ax[0, 1].set_ylabel("PC2")

    by_len = ax[1, 0].scatter(
        z[:, 0],
        z[:, 1],
        c=np.log10(np.maximum(length[idx], 1e-12)),
        cmap="afmhot",
        s=11,
        alpha=0.9,
        edgecolors="none",
    )
    fig.colorbar(by_len, ax=ax[1, 0], label="log activation norm")
    ax[1, 0].set_xlabel("PC1")
    ax[1, 0].set_ylabel("PC2")

    mean_len = np.array([length[pos == t].mean() for t in range(token_positions)])
    sd_len = np.array([length[pos == t].std() for t in range(token_positions)])
    xs = np.arange(token_positions)
    ax[1, 1].fill_between(xs, mean_len - sd_len, mean_len + sd_len, color="#06d6a0", alpha=0.16)
    ax[1, 1].plot(xs, mean_len, color="#06d6a0", lw=2.4)
    ax[1, 1].set_xlabel("token position")
    ax[1, 1].set_ylabel("activation norm")

    fig.suptitle(
        f"OLMo-3-32B layer-25 residual stream, {x.shape[0]:,} real token activations",
        color="#f2f4f8",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"source shape: prompts={prompts}, token_positions={token_positions}, d_model={d_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
