"""UMAP directly on the feature × feature co-activation matrix.

Each alive feature is represented by its co-firing vector (length = n_alive).
UMAP unfolds that. Spectral clustering gives community labels. Result: features
that fire on the same inputs cluster geometrically, regardless of decoder direction.

Outputs:
  18_coactivation_umap.png — UMAP on co-fire vectors, colored by spectral cluster
  19_coactivation_umap_by_token.png — same UMAP, colored by dominant context token
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from umap import UMAP

from data import DWELL_BASE, N_DWELL
from model import SAEConfig, TopKSAE


def short_label(s: str, n: int = 30) -> str:
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    s = s.replace("$", r"\$")
    if len(s) <= n:
        return s
    parts = s.split("::")
    if len(parts) >= 3:
        return "..::" + "::".join(parts[-2:])[: n - 4]
    return s[: n - 1] + "…"


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

    # Batched TopK encode
    print("computing TopK activations...")
    z_chunks = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, acts.shape[0], bs):
            sparse, _ = sae.encode_topk(acts[i : i + bs])
            z_chunks.append(sparse.cpu().numpy())
    z = np.concatenate(z_chunks, axis=0)
    fired = (z > 0)
    max_act = z.max(axis=0)
    alive = max_act > 1e-6
    n_alive = int(alive.sum())
    alive_idx = np.where(alive)[0]
    print(f"alive: {n_alive}/{len(max_act)}")

    # Co-firing similarity among alive features
    F = fired[:, alive].astype(np.float32)  # (N, A)
    counts = F.sum(axis=0) + 1e-6
    co = F.T @ F  # joint counts
    sim = co / np.sqrt(np.outer(counts, counts))
    np.fill_diagonal(sim, 0.0)

    sim_pos = np.maximum(sim, 0.0)
    Wn = sim_pos / (np.linalg.norm(sim_pos, axis=1, keepdims=True) + 1e-9)

    # Cluster via plain KMeans on row vectors (robust, fast).
    from sklearn.cluster import KMeans
    n_clusters = 14
    print(f"kmeans clustering ({n_clusters} clusters)...")
    labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=42).fit_predict(Wn)

    print("running UMAP on co-fire vectors...")
    proj = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(Wn)

    # Per-cluster signature: dominant context token (mode)
    n_vocab = int(ctx_toks.max()) + 1
    ctx64 = ctx_toks.astype(np.int64)
    N = z.shape[0]
    S = csr_matrix(
        (np.ones(N, dtype=np.float32), (ctx64, np.arange(N))),
        shape=(n_vocab, N),
    )
    z_alive = z[:, alive_idx].astype(np.float32, copy=False)
    tok_act = np.asarray(S @ z_alive)  # (n_vocab, n_alive)
    feat_top_token = tok_act.argmax(axis=0)
    cluster_sig = {}
    for c in range(n_clusters):
        sel = labels == c
        if sel.sum() == 0:
            continue
        toks = feat_top_token[sel]
        most_common = np.bincount(toks).argmax()
        cluster_sig[c] = id_to_label.get(int(most_common), f"<id:{int(most_common)}>")

    # ---------- 18: by spectral cluster ----------
    fig, ax = plt.subplots(figsize=(13, 9))
    palette = plt.cm.tab20.colors
    for c in range(n_clusters):
        sel = labels == c
        if sel.sum() == 0:
            continue
        ax.scatter(
            proj[sel, 0], proj[sel, 1],
            c=[palette[c % len(palette)]],
            label=f"C{c} {short_label(cluster_sig.get(c, '?'), 32)} (n={sel.sum()})",
            s=10 + 30 * (max_act[alive_idx[sel]] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
    ax.set_title(
        f"UMAP of co-activation rows  (n={n_alive} alive features)\n"
        "geometry = features that fire on the same inputs sit close;  color = spectral cluster"
    )
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "18_coactivation_umap.png", dpi=130)
    plt.close(fig)
    print("  [ok] 18_coactivation_umap.png")

    # ---------- 19: same UMAP, colored by dominant context token (top 12 + other) ----------
    tok_counts = np.bincount(feat_top_token, minlength=n_vocab)
    top_toks_global = np.argsort(-tok_counts)[:12]
    fig, ax = plt.subplots(figsize=(13, 9))
    for i, t in enumerate(top_toks_global):
        sel = feat_top_token == t
        if sel.sum() == 0:
            continue
        label = short_label(id_to_label.get(int(t), f"<{int(t)}>"), 30)
        ax.scatter(
            proj[sel, 0], proj[sel, 1],
            c=[palette[i % len(palette)]], label=f"{label} ({sel.sum()})",
            s=10 + 30 * (max_act[alive_idx[sel]] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
    other = ~np.isin(feat_top_token, top_toks_global)
    ax.scatter(proj[other, 0], proj[other, 1], c="#cccccc",
               label=f"other ({other.sum()})", s=5, alpha=0.4, edgecolors="none")
    ax.set_title(
        f"UMAP of co-activation rows  (n={n_alive})\ncolor = feature's dominant context token (top 12 + other)"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "19_coactivation_umap_by_token.png", dpi=130)
    plt.close(fig)
    print("  [ok] 19_coactivation_umap_by_token.png")

    # ---------- 20: PCA on co-fire rows, colored by spectral cluster ----------
    print("running PCA on co-fire vectors...")
    pca = PCA(n_components=2).fit(Wn)
    proj_p = pca.transform(Wn)
    fig, ax = plt.subplots(figsize=(13, 9))
    for c in range(n_clusters):
        sel = labels == c
        if sel.sum() == 0:
            continue
        ax.scatter(
            proj_p[sel, 0], proj_p[sel, 1],
            c=[palette[c % len(palette)]],
            label=f"C{c} {short_label(cluster_sig.get(c, '?'), 32)} (n={sel.sum()})",
            s=10 + 30 * (max_act[alive_idx[sel]] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
    ax.set_title(
        f"PCA of co-activation rows  (n={n_alive} alive features)\n"
        f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}%  •  PC2 {pca.explained_variance_ratio_[1]*100:.1f}%  •  color = spectral cluster"
    )
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(plots_dir / "20_coactivation_pca.png", dpi=130)
    plt.close(fig)
    print("  [ok] 20_coactivation_pca.png")

    # ---------- 21: PCA same, colored by dominant token ----------
    fig, ax = plt.subplots(figsize=(13, 9))
    for i, t in enumerate(top_toks_global):
        sel = feat_top_token == t
        if sel.sum() == 0:
            continue
        label = short_label(id_to_label.get(int(t), f"<{int(t)}>"), 30)
        ax.scatter(
            proj_p[sel, 0], proj_p[sel, 1],
            c=[palette[i % len(palette)]], label=f"{label} ({sel.sum()})",
            s=10 + 30 * (max_act[alive_idx[sel]] / max_act[alive_idx].max()),
            alpha=0.85, edgecolors="none",
        )
    other = ~np.isin(feat_top_token, top_toks_global)
    ax.scatter(proj_p[other, 0], proj_p[other, 1], c="#cccccc",
               label=f"other ({other.sum()})", s=5, alpha=0.4, edgecolors="none")
    ax.set_title(f"PCA of co-activation rows  (n={n_alive})\ncolor = feature's dominant context token")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(plots_dir / "21_coactivation_pca_by_token.png", dpi=130)
    plt.close(fig)
    print("  [ok] 21_coactivation_pca_by_token.png")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
