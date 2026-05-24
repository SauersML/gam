"""Discovery plots — find interesting structure in the trained SAE features.

  15_coactivation_communities.png — co-activation graph spectral-clustered, UMAP colored by community
  16_dwell_signature.png         — feature × dwell-bucket mean-activation heatmap (reveals time-aware features)
  17_depth_signature.png         — feature × stack-position mean-activation heatmap (reveals depth-specific features)
  18_feature_directory.png       — top-N features as cards with logit-lens label + dwell + depth signature
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
from sklearn.cluster import SpectralClustering
from umap import UMAP

from data import DWELL_BASE, FRAME_BASE, N_DWELL
from model import LMConfig, SAEConfig, StackLM, TopKSAE


def short_label(s: str, n: int = 36) -> str:
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    s = s.replace("$", r"\$")
    if len(s) <= n:
        return s
    parts = s.split("::")
    if len(parts) >= 3:
        return "..::" + "::".join(parts[-2:])[: n - 4]
    return s[: n - 1] + "…"


def load_all(out_dir: Path):
    sae_blob = torch.load(out_dir / "sae.pt", map_location="cpu", weights_only=False)
    sae_cfg = SAEConfig(**sae_blob["cfg"])
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_blob["state_dict"], strict=False)
    sae.eval()

    blob = np.load(out_dir / "activations.npz")
    acts = torch.from_numpy(blob["acts"])
    ctx_toks = blob["ctx_toks"]
    pos_idx = blob["pos_idx"]
    win_idx = blob["win_idx"]
    seq_len = int(blob["seq_len"])

    sym = json.loads((out_dir / "vocab_sym.json").read_text())
    id_to_label = {int(k): v for k, v in sym.items()}
    return sae, sae_cfg, acts, ctx_toks, win_idx, pos_idx, seq_len, id_to_label


def fire_matrix(sae, acts):
    with torch.no_grad():
        sparse, _ = sae.encode_topk(acts)
        z = sparse.cpu().numpy()
    return z


# ---------- 15: co-activation graph + community detection on UMAP ----------

def plot_coactivation_communities(out_dir: Path, plots_dir: Path):
    sae, sae_cfg, acts, ctx_toks, win_idx, pos_idx, seq_len, id_to_label = load_all(out_dir)
    z = fire_matrix(sae, acts)
    fired = (z > 0)
    max_act = z.max(axis=0)
    alive = max_act > 1e-6
    print(f"alive features: {alive.sum()}")

    # Co-activation matrix among alive features
    F = fired[:, alive].astype(np.float32)  # (N, A)
    counts = F.sum(axis=0)  # per-feature firing counts
    # cosine-style co-occurrence (normalized by independent firing rates)
    co = (F.T @ F)  # (A, A) — counts of joint firing
    # Convert to a similarity (probabilistic association): co[i,j] / sqrt(counts[i] * counts[j])
    denom = np.sqrt(np.outer(counts, counts) + 1e-9)
    sim = co / denom
    np.fill_diagonal(sim, 0.0)

    # Spectral clustering needs an affinity matrix. Drop very low values to denoise.
    aff = np.maximum(sim, 0.0)
    # Drop the low-affinity bottom 95% to make graph sparse
    keep_thresh = np.quantile(aff[aff > 0], 0.85) if (aff > 0).any() else 0.0
    aff = np.where(aff > keep_thresh, aff, 0.0)
    print(f"co-activation similarity > {keep_thresh:.3f}, density={float((aff>0).sum())/aff.size:.3f}")

    n_clusters = 12
    try:
        sc = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed",
            assign_labels="kmeans", random_state=42,
        ).fit(aff)
        labels = sc.labels_
    except Exception as e:
        print(f"spectral failed ({e}); falling back to kmeans on rows")
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=42).fit_predict(aff)

    # UMAP on decoder directions, colored by cluster
    W_dec = sae.W_dec.detach().cpu().numpy().T
    W_alive = W_dec[alive]
    Wn = W_alive / (np.linalg.norm(W_alive, axis=1, keepdims=True) + 1e-9)
    proj = UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42).fit_transform(Wn)

    # Identify each cluster's "signature" by its dominant context token
    feat_top_token = np.zeros(z.shape[1], dtype=np.int64)
    n_vocab = int(ctx_toks.max()) + 1
    for f in np.where(alive)[0]:
        col = z[:, f]
        if col.max() < 1e-6:
            continue
        sums = np.bincount(ctx_toks.astype(np.int64), weights=col, minlength=n_vocab)
        feat_top_token[f] = int(np.argmax(sums))
    alive_top_tok = feat_top_token[alive]

    cluster_signatures = {}
    for c in range(n_clusters):
        sel = labels == c
        if sel.sum() == 0:
            continue
        toks_c = alive_top_tok[sel]
        most_common = np.bincount(toks_c).argmax()
        cluster_signatures[c] = id_to_label.get(int(most_common), f"<id:{int(most_common)}>")

    fig, ax = plt.subplots(figsize=(13, 9))
    palette = plt.cm.tab20.colors
    for c in range(n_clusters):
        sel = labels == c
        if sel.sum() == 0:
            continue
        sig = short_label(cluster_signatures.get(c, "?"), 30)
        ax.scatter(
            proj[sel, 0], proj[sel, 1],
            c=[palette[c % len(palette)]], label=f"C{c} {sig} (n={sel.sum()})",
            s=10 + 30 * (max_act[alive][sel] / max_act[alive].max()),
            alpha=0.85, edgecolors="none",
        )
    ax.set_title(f"Co-activation communities  (n={alive.sum()} alive features, {n_clusters} clusters from spectral on co-fire similarity)")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "15_coactivation_communities.png", dpi=130)
    plt.close(fig)
    print("  [ok] 15_coactivation_communities.png")
    return alive, labels, z, ctx_toks


# ---------- 16: dwell signature ----------

def plot_dwell_signature(out_dir: Path, plots_dir: Path):
    sae, _, acts, ctx_toks, win_idx, pos_idx, seq_len, id_to_label = load_all(out_dir)
    z = fire_matrix(sae, acts)
    max_act = z.max(axis=0)
    frac_fire = (z > 0).mean(axis=0)
    alive = max_act > 1e-6

    # For each feature, mean activation conditioned on the context-token being a dwell-bucket k.
    sig = np.zeros((int(alive.sum()), N_DWELL))
    alive_idx = np.where(alive)[0]
    for k in range(N_DWELL):
        mask = ctx_toks == (DWELL_BASE + k)
        if mask.sum() == 0:
            continue
        sig[:, k] = z[mask][:, alive_idx].mean(axis=0)

    # Filter to features that actually have dwell-conditional structure
    # (some variance across buckets)
    var = sig.std(axis=1) / (sig.mean(axis=1) + 1e-6)
    good = np.argsort(-var)[: min(64, len(var))]
    mat = sig[good]
    rmax = mat.max(axis=1, keepdims=True) + 1e-9
    mat_n = mat / rmax

    # Sort within selected by argmax bucket so similar profiles cluster
    order = np.argsort(np.argmax(mat_n, axis=1))
    mat_n = mat_n[order]
    sel_feat_ids = alive_idx[good][order]

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(mat_n, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(N_DWELL))
    ax.set_xticklabels([f"<DWELL_{k}>" for k in range(N_DWELL)], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(sel_feat_ids)))
    ax.set_yticklabels([f"f{int(f)}" for f in sel_feat_ids], fontsize=6)
    ax.set_xlabel("dwell bucket (low → high log-time)")
    ax.set_ylabel("features (sorted by dwell preference)")
    ax.set_title(f"Dwell signature — top-{len(sel_feat_ids)} features by dwell-conditional variance\n"
                 "(row-norm; bright = feature fires preferentially when the upcoming dwell token is bucket k)")
    fig.colorbar(im, ax=ax, label="row-norm mean activation")
    fig.tight_layout()
    fig.savefig(plots_dir / "16_dwell_signature.png", dpi=130)
    plt.close(fig)
    print("  [ok] 16_dwell_signature.png")


# ---------- 17: stack-depth signature ----------

def plot_depth_signature(out_dir: Path, plots_dir: Path):
    sae, _, acts, ctx_toks, win_idx, pos_idx, seq_len, id_to_label = load_all(out_dir)
    z = fire_matrix(sae, acts)
    max_act = z.max(axis=0)
    alive = max_act > 1e-6
    alive_idx = np.where(alive)[0]

    # Compute "depth within stack" for each position: how many tokens since the previous <BOS>.
    # Load original stream.
    stream = np.load(out_dir / "stream.npz")
    tokens_full = stream["tokens"]
    BOS_ID = 1  # by data.py convention

    # win starts (from harvest config)
    n_full = (tokens_full.size - seq_len - 1) // seq_len + 1
    n_wins = int(win_idx.max() + 1) if (win_idx.max() < n_full) else n_full

    # For each position in our harvested set, find depth = pos - last_BOS_pos in absolute stream.
    # This is the position WITHIN the call chain of the current event.
    # Since each window is a slice of contiguous tokens, we walk per harvested point.
    abs_pos = win_idx.astype(np.int64) * seq_len + pos_idx.astype(np.int64)
    # For efficiency, precompute last-BOS index up to each token index.
    # Cumulative max trick: last_bos[i] = max j<=i where tokens[j]==BOS.
    bos_at = (tokens_full == BOS_ID)
    bos_idx_array = np.where(bos_at)[0]
    # For each abs_pos, find last bos via searchsorted
    j = np.searchsorted(bos_idx_array, abs_pos, side="right") - 1
    j = np.clip(j, 0, len(bos_idx_array) - 1)
    last_bos = bos_idx_array[j]
    depth = abs_pos - last_bos  # 0 means at the <BOS>, 1 = first frame, ...

    max_depth = 16  # cap; deeper positions get folded
    sig = np.zeros((int(alive.sum()), max_depth))
    counts = np.zeros(max_depth)
    for d in range(max_depth):
        mask = depth == d
        if mask.sum() == 0:
            continue
        sig[:, d] = z[mask][:, alive_idx].mean(axis=0)
        counts[d] = mask.sum()

    # Filter to features with strong depth-conditional variance
    var = sig.std(axis=1) / (sig.mean(axis=1) + 1e-6)
    good = np.argsort(-var)[: min(64, len(var))]
    mat = sig[good]
    rmax = mat.max(axis=1, keepdims=True) + 1e-9
    mat_n = mat / rmax
    order = np.argsort(np.argmax(mat_n, axis=1))
    mat_n = mat_n[order]
    sel_feat_ids = alive_idx[good][order]

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(mat_n, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(max_depth))
    ax.set_xticklabels([str(d) for d in range(max_depth)], fontsize=8)
    ax.set_yticks(np.arange(len(sel_feat_ids)))
    ax.set_yticklabels([f"f{int(f)}" for f in sel_feat_ids], fontsize=6)
    ax.set_xlabel("position-in-stack (0 = <BOS>, 1+ = call-chain depth)")
    ax.set_ylabel("features (sorted by depth preference)")
    ax.set_title(f"Stack-depth signature — top-{len(sel_feat_ids)} features by depth-conditional variance\n"
                 "(row-norm; bright = feature fires at this depth in the call chain)")
    fig.colorbar(im, ax=ax, label="row-norm mean activation")
    fig.tight_layout()
    fig.savefig(plots_dir / "17_depth_signature.png", dpi=130)
    plt.close(fig)
    print("  [ok] 17_depth_signature.png")


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_coactivation_communities(out_dir, plots_dir)
    plot_dwell_signature(out_dir, plots_dir)
    plot_depth_signature(out_dir, plots_dir)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
