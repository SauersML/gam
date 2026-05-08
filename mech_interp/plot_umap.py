"""UMAP of SAE decoder directions, colored by the feature's dominant context.

For each feature, we compute the token-activation profile (mean activation
per token), then color by the most-activating token category:
  - frame token (Rust function in gam/criterion/std)
  - <BOS> / <EOS> structural
  - <DWELL_*> time-bucket
  - <stub:...> dyld trampoline

Plots:
  09_umap_by_category.png — colored by token category (categorical)
  10_umap_by_topframe.png — colored by which top-N frame they fire on
  11_umap_by_dwell.png    — colored by dwell-bucket affinity (continuous)
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
from umap import UMAP

from data import DWELL_BASE, FRAME_BASE, N_DWELL
from model import SAEConfig, TopKSAE


def short_label(s: str, n: int = 32) -> str:
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    if len(s) <= n:
        return s
    parts = s.split("::")
    if len(parts) >= 3:
        return "..::" + "::".join(parts[-2:])[: n - 4]
    return s[: n - 1] + "…"


def categorize(label: str) -> str:
    if label.startswith("<DWELL_"):
        return "dwell"
    if label in ("<BOS>", "<EOS>", "<SEP>", "<PAD>", "<UNK>"):
        return "structural"
    if label.startswith("<stub:"):
        return "stub"
    if "criterion::" in label:
        return "criterion"
    if "rayon" in label:
        return "rayon"
    if label.startswith(("std::", "core::", "alloc::")):
        return "std"
    if "cubic_cell_kernel" in label or "evaluate_cell_moments" in label or "evaluate_non_affine" in label:
        return "kernel"
    if "gam::" in label or "cell_moment_dedup" in label:
        return "gam"
    return "other"


CATEGORY_COLORS = {
    "kernel": "#d62728",   # red — the hot inner kernel
    "gam":    "#ff7f0e",   # orange — gam outer
    "criterion": "#2ca02c",
    "rayon": "#9467bd",
    "std":    "#1f77b4",
    "stub":   "#8c564b",
    "dwell":  "#e377c2",
    "structural": "#7f7f7f",
    "other":  "#bcbd22",
}


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load symbolicated id → label
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

    # Per-feature TopK activations (only features that win TopK count as "fired")
    with torch.no_grad():
        sparse, _ = sae.encode_topk(acts)
        z = sparse.cpu().numpy()  # (N, dict_size), zeros except at TopK positions
        # Also keep the pre-topk for activation magnitude.

    max_act = z.max(axis=0)
    frac_fire = (z > 0).mean(axis=0)
    alive = max_act > 1e-6
    print(f"alive (ever fired in TopK): {alive.sum()}/{len(max_act)}")

    # For each feature, find its dominant token
    # mean activation per token id
    n_vocab = max(ctx_toks) + 1
    print(f"computing per-feature dominant tokens (n_vocab={n_vocab})...")
    # vectorized: for each feature, sum z[:, f] grouped by ctx_toks
    feat_top_token = np.zeros(z.shape[1], dtype=np.int64)
    feat_dwell_aff = np.zeros(z.shape[1], dtype=np.float32)  # dwell weighting
    for f in np.where(alive)[0]:
        col = z[:, f]
        if col.max() < 1e-6:
            continue
        # tokens weighted by activation
        # use bincount with weights
        sums = np.bincount(ctx_toks.astype(np.int64), weights=col, minlength=n_vocab)
        feat_top_token[f] = int(np.argmax(sums))
        # dwell affinity: fraction of total activation on dwell tokens
        dwell_mask = (np.arange(n_vocab) >= DWELL_BASE) & (np.arange(n_vocab) < DWELL_BASE + N_DWELL)
        feat_dwell_aff[f] = sums[dwell_mask].sum() / max(sums.sum(), 1e-9)

    # Decoder directions, alive only
    W_dec = sae.W_dec.detach().cpu().numpy().T  # (dict_size, d_model)
    W_alive = W_dec[alive]
    feats_alive = np.where(alive)[0]
    top_tok_alive = feat_top_token[alive]
    dwell_aff_alive = feat_dwell_aff[alive]
    max_act_alive = max_act[alive]
    frac_fire_alive = frac_fire[alive]

    # L2-normalize for cosine UMAP
    Wn = W_alive / (np.linalg.norm(W_alive, axis=1, keepdims=True) + 1e-9)

    print(f"running UMAP on {Wn.shape[0]} alive features...")
    reducer = UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=42)
    proj = reducer.fit_transform(Wn)

    # ---------- 09: by category ----------
    cats = np.array([categorize(id_to_label.get(int(t), "?")) for t in top_tok_alive])
    fig, ax = plt.subplots(figsize=(11, 9))
    for cat, color in CATEGORY_COLORS.items():
        sel = cats == cat
        if sel.sum() == 0:
            continue
        ax.scatter(
            proj[sel, 0], proj[sel, 1],
            c=color, label=f"{cat} ({sel.sum()})",
            s=8 + 30 * (max_act_alive[sel] / max_act_alive.max()),
            alpha=0.75, edgecolors="none",
        )
    ax.set_title(f"UMAP of SAE decoder directions  (n={Wn.shape[0]} alive features)\ncolor = category of dominant context token, size ∝ peak activation")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "09_umap_by_category.png", dpi=130)
    plt.close(fig)
    print("  [ok] 09_umap_by_category.png")

    # ---------- 10: by top frame name (top 12) ----------
    # Get top-12 most-targeted tokens (across alive features), label remainder "other"
    tok_counts = np.bincount(top_tok_alive, minlength=n_vocab)
    top_toks_global = np.argsort(-tok_counts)[:12]

    fig, ax = plt.subplots(figsize=(13, 9))
    palette = plt.cm.tab20.colors
    for i, t in enumerate(top_toks_global):
        sel = top_tok_alive == t
        if sel.sum() == 0:
            continue
        label = short_label(id_to_label.get(int(t), f"<{int(t)}>"), 30)
        ax.scatter(
            proj[sel, 0], proj[sel, 1],
            c=[palette[i % len(palette)]], label=f"{label} ({sel.sum()})",
            s=10 + 30 * (max_act_alive[sel] / max_act_alive.max()),
            alpha=0.85, edgecolors="none",
        )
    other_mask = ~np.isin(top_tok_alive, top_toks_global)
    ax.scatter(
        proj[other_mask, 0], proj[other_mask, 1],
        c="#cccccc", label=f"other ({other_mask.sum()})",
        s=5, alpha=0.4, edgecolors="none",
    )
    ax.set_title(f"UMAP colored by feature's dominant context token (top 12 + other)")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "10_umap_by_topframe.png", dpi=130)
    plt.close(fig)
    print("  [ok] 10_umap_by_topframe.png")

    # ---------- 11: by dwell affinity (continuous) ----------
    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=dwell_aff_alive,
        s=10 + 30 * (max_act_alive / max_act_alive.max()),
        cmap="magma",
        alpha=0.85, edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="fraction of feature activation on <DWELL_*> tokens")
    ax.set_title("UMAP — bright = time-aware features (fire on dwell tokens)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "11_umap_by_dwell.png", dpi=130)
    plt.close(fig)
    print("  [ok] 11_umap_by_dwell.png")

    # ---------- 12: by firing density (continuous) ----------
    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=np.log10(frac_fire_alive + 1e-8),
        s=10 + 30 * (max_act_alive / max_act_alive.max()),
        cmap="viridis",
        alpha=0.85, edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="log10(firing fraction)")
    ax.set_title("UMAP — bright = densely-firing features, dark = sparse/specific")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(plots_dir / "12_umap_by_density.png", dpi=130)
    plt.close(fig)
    print("  [ok] 12_umap_by_density.png")

    print(f"\nUMAP plots written to {plots_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
