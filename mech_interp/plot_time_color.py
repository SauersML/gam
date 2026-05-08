"""Color UMAP / PCA of SAE decoder directions by per-feature continuous attributes:
  - mean firing time (wall-clock seconds into the profile)
  - mean dwell-bucket of the upcoming event when this feature fires
  - mean stack-depth at which this feature fires

If the SAE residual stream has internal representations of *time*, *duration*, or
*depth*, features encoding the same value should cluster on these projections.

Plots:
  22_umap_by_meantime.png       — UMAP of W_dec, colored by mean firing time (s)
  23_umap_by_dwellpref.png      — UMAP of W_dec, colored by preferred dwell bucket
  24_umap_by_depthpref.png      — UMAP of W_dec, colored by preferred stack depth
  25_pca_by_meantime.png        — PCA of W_dec, same coloring
  26_meantime_distribution.png  — histogram of per-feature mean firing time
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from umap import UMAP

from data import DWELL_BASE, N_DWELL
from model import SAEConfig, TopKSAE


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    sym = json.loads((out_dir / "vocab_sym.json").read_text())
    id_to_label = {int(k): v for k, v in sym.items()}

    sae_blob = torch.load(out_dir / "sae.pt", map_location="cpu", weights_only=False)
    sae_cfg = SAEConfig(**sae_blob["cfg"])
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_blob["state_dict"], strict=False)
    sae.eval()

    blob = np.load(out_dir / "activations.npz")
    acts = torch.from_numpy(blob["acts"])
    ctx_toks = blob["ctx_toks"]
    win_idx = blob["win_idx"]
    pos_idx = blob["pos_idx"]
    seq_len = int(blob["seq_len"])

    print("computing TopK activations...")
    z_chunks = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, acts.shape[0], bs):
            sparse, _ = sae.encode_topk(acts[i : i + bs])
            z_chunks.append(sparse.cpu().numpy())
    z = np.concatenate(z_chunks, axis=0)
    max_act = z.max(axis=0)
    alive = max_act > 1e-6
    alive_idx = np.where(alive)[0]
    n_alive = int(alive.sum())
    print(f"alive: {n_alive}/{len(max_act)}")

    # --- per-position metadata: wall-clock time, dwell bucket, stack depth ---
    meta = np.load(out_dir / "events_meta.npz")
    tok_start = meta["token_start"]
    start_time_ms = meta["start_time_ms"]
    abs_tok = win_idx.astype(np.int64) * seq_len + pos_idx.astype(np.int64)
    event_idx = np.clip(np.searchsorted(tok_start, abs_tok, side="right") - 1, 0, len(tok_start) - 1)
    pos_time_s = start_time_ms[event_idx] / 1000.0  # s

    # Dwell bucket of the *next-emitted* dwell token at each position:
    # use the dwell-bucket of the event this position belongs to.
    # The dwell token is the last in the event; encode dwell index via meta dwell_ms.
    dwell_ms_all = meta["dwell_ms"]
    log_dwell = np.log(dwell_ms_all + 1e-6)
    pos_log_dwell = log_dwell[event_idx]

    # Stack depth: for each position, depth = pos_in_stream - last_BOS. Computed from stream.
    stream = np.load(out_dir / "stream.npz")
    tokens_full = stream["tokens"]
    BOS_ID = 1
    bos_at_idx = np.where(tokens_full == BOS_ID)[0]
    j = np.clip(np.searchsorted(bos_at_idx, abs_tok, side="right") - 1, 0, len(bos_at_idx) - 1)
    pos_depth = abs_tok - bos_at_idx[j]

    # Per-feature activation-weighted means.
    print("computing per-feature attribute means...")
    z_alive = z[:, alive_idx]
    weight_sum = z_alive.sum(axis=0) + 1e-9  # (n_alive,)

    feat_meantime = (z_alive.T @ pos_time_s) / weight_sum  # (n_alive,)
    feat_logdwell = (z_alive.T @ pos_log_dwell) / weight_sum
    feat_depth = (z_alive.T @ pos_depth.astype(np.float64)) / weight_sum

    # Decoder dirs UMAP
    W_dec = sae.W_dec.detach().cpu().numpy().T
    Wn = W_dec[alive] / (np.linalg.norm(W_dec[alive], axis=1, keepdims=True) + 1e-9)
    print("running UMAP on W_dec rows...")
    proj = UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42).fit_transform(Wn)
    print("running PCA on W_dec rows...")
    pca = PCA(n_components=2).fit(Wn)
    proj_p = pca.transform(Wn)

    def scatter_continuous(proj, color_vals, title, path, cmap, cbar_label):
        fig, ax = plt.subplots(figsize=(11, 8))
        sc = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=color_vals, cmap=cmap,
            s=10 + 30 * (max_act[alive_idx] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    scatter_continuous(
        proj, feat_meantime,
        f"UMAP of W_dec  (n={n_alive})\ncolor = activation-weighted mean firing time (s into profile)",
        plots_dir / "22_umap_by_meantime.png", "viridis", "mean firing time (s)",
    )
    print("  [ok] 22_umap_by_meantime.png")

    scatter_continuous(
        proj, feat_logdwell,
        f"UMAP of W_dec  (n={n_alive})\ncolor = activation-weighted mean log(dwell ms)",
        plots_dir / "23_umap_by_dwellpref.png", "magma", "mean log(dwell ms)",
    )
    print("  [ok] 23_umap_by_dwellpref.png")

    scatter_continuous(
        proj, feat_depth,
        f"UMAP of W_dec  (n={n_alive})\ncolor = activation-weighted mean stack depth",
        plots_dir / "24_umap_by_depthpref.png", "plasma", "mean stack depth",
    )
    print("  [ok] 24_umap_by_depthpref.png")

    # PCA versions side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, vals, title, cmap, cbar in zip(
        axes,
        [feat_meantime, feat_logdwell, feat_depth],
        ["time (s)", "log(dwell ms)", "stack depth"],
        ["viridis", "magma", "plasma"],
        ["s", "log(ms)", "depth"],
    ):
        sc = ax.scatter(
            proj_p[:, 0], proj_p[:, 1],
            c=vals, cmap=cmap,
            s=10 + 25 * (max_act[alive_idx] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=cbar)
        ax.set_title(f"PCA W_dec — color = {title}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    fig.suptitle(f"PCA projection (n={n_alive} alive features) colored by 3 continuous attributes")
    fig.tight_layout()
    fig.savefig(plots_dir / "25_pca_by_attributes.png", dpi=130)
    plt.close(fig)
    print("  [ok] 25_pca_by_attributes.png")

    # Histogram of mean firing time
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].hist(feat_meantime, bins=40, color="#1f77b4")
    axes[0].set_xlabel("mean firing time (s)")
    axes[0].set_ylabel("# features")
    axes[0].set_title("Per-feature mean firing time")

    axes[1].hist(feat_logdwell, bins=40, color="#d62728")
    axes[1].set_xlabel("mean log(dwell ms)")
    axes[1].set_title("Per-feature mean dwell preference")

    axes[2].hist(feat_depth, bins=40, color="#2ca02c")
    axes[2].set_xlabel("mean stack depth")
    axes[2].set_title("Per-feature mean stack depth")

    fig.tight_layout()
    fig.savefig(plots_dir / "26_attribute_distributions.png", dpi=130)
    plt.close(fig)
    print("  [ok] 26_attribute_distributions.png")

    # Quantitative test: linear-probe accuracy of W_dec → predicted attribute.
    # If features encode time, a linear regression W_dec → meantime should explain a lot.
    print("\nLinear-probe R² (regress W_dec → attribute, 80/20 train/test):")
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_alive)
    split = int(0.8 * n_alive)
    tr, te = perm[:split], perm[split:]
    for name, y in [("time", feat_meantime), ("log_dwell", feat_logdwell), ("depth", feat_depth)]:
        model = Ridge(alpha=1.0).fit(Wn[tr], y[tr])
        r2 = model.score(Wn[te], y[te])
        print(f"  W_dec → {name:10s}  R²={r2:.3f}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
