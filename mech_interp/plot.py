"""Generate diagnostic + interpretability plots from the trained pipeline.

Outputs PNGs into data/plots/.
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
from sklearn.decomposition import PCA

from model import LMConfig, SAEConfig, StackLM, TopKSAE


def parse_train_log(path: Path):
    """Parse train.log into (lm_steps, lm_loss), (sae_steps, sae_recon, sae_dead, sae_expl)."""
    lm_re = re.compile(r"\[lm\] step (\d+)/\d+\s+loss\s+([\d.]+)\s+ppl\s+([\d.]+)")
    sae_re = re.compile(
        r"\[sae\] step (\d+)/\d+\s+recon\s+([\d.]+)\s+expl\s+([\d.]+)\s+dead\s+(\d+)/(\d+)"
    )
    lm = []
    sae = []
    for line in path.read_text().splitlines():
        m = lm_re.search(line)
        if m:
            lm.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = sae_re.search(line)
        if m:
            sae.append(
                (int(m.group(1)), float(m.group(2)), float(m.group(3)),
                 int(m.group(4)), int(m.group(5)))
            )
    return lm, sae


def short_label(s: str, n: int = 36) -> str:
    s = s.replace("_$u7b$$u7b$closure$u7d$$u7d$", "{closure}")
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    if len(s) <= n:
        return s
    # Keep last 2 path components
    parts = s.split("::")
    if len(parts) >= 3:
        return "..::" + "::".join(parts[-2:])
    return s[: n - 1] + "…"


def plot_training(out_dir: Path, log_path: Path):
    lm, sae = parse_train_log(log_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # LM loss + ppl
    ls = np.array(lm)
    ax = axes[0, 0]
    ax.plot(ls[:, 0], ls[:, 1], lw=2, color="#1f77b4")
    ax.set_xlabel("step")
    ax.set_ylabel("CE loss", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(ls[:, 0], ls[:, 2], lw=2, color="#d62728", linestyle="--")
    ax2.set_ylabel("perplexity", color="#d62728")
    ax2.set_yscale("log")
    ax.set_title("LM training: loss + perplexity")

    # SAE recon
    ss = np.array(sae)
    ax = axes[0, 1]
    ax.plot(ss[:, 0], ss[:, 1], lw=2, color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("recon MSE (log)", color="#2ca02c")
    ax.set_title("SAE: reconstruction loss")
    ax2 = ax.twinx()
    ax2.plot(ss[:, 0], ss[:, 2], lw=2, color="#9467bd", linestyle="--")
    ax2.set_ylabel("variance explained", color="#9467bd")

    # Dead features
    ax = axes[1, 0]
    ax.plot(ss[:, 0], ss[:, 3], lw=2, color="#7f7f7f")
    ax.fill_between(ss[:, 0], 0, ss[:, 3], alpha=0.2, color="#7f7f7f")
    ax.axhline(ss[0, 4], color="black", lw=0.5, linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("dead features")
    ax.set_title(f"SAE: dead features (of {ss[0, 4]:.0f})")

    # LM ppl alone log scale
    ax = axes[1, 1]
    ax.plot(ls[:, 0], ls[:, 2], lw=2, color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity (log)")
    ax.set_title("LM perplexity (1.0 = perfect, vocab=1443 = random)")
    ax.axhline(1.0, color="green", lw=0.7, linestyle="--", label="perfect")
    ax.axhline(1443, color="red", lw=0.7, linestyle="--", label="random")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "01_training_curves.png", dpi=120)
    plt.close(fig)


def plot_vocab_zipf(out_dir: Path, tokens: np.ndarray, vocab: dict[str, int]):
    inv = {v: k for k, v in vocab.items()}
    counts = np.bincount(tokens.astype(np.int64), minlength=len(vocab))
    sort_idx = np.argsort(-counts)
    sorted_counts = counts[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    rank = np.arange(1, len(sorted_counts) + 1)
    ax.loglog(rank, sorted_counts, lw=1.5, color="#1f77b4")
    ax.set_xlabel("rank (log)")
    ax.set_ylabel("frequency (log)")
    ax.set_title(f"Token frequency Zipf — vocab={len(vocab)}, total tokens={tokens.size:,}")

    # Annotate top 8
    for i in range(8):
        tid = int(sort_idx[i])
        name = short_label(inv.get(tid, f"<{tid}>"), 28)
        ax.annotate(name, xy=(i + 1, sorted_counts[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "02_vocab_zipf.png", dpi=120)
    plt.close(fig)


def plot_stack_dwell(out_dir: Path, summary: dict):
    """Stack depth + dwell time histograms from raw samples."""
    # Re-derive from stream (we don't have raw events anymore, but the dwell tokens are there).
    # Just plot summary stats and dwell-bucket population.
    pass  # combined into plot_dwell_buckets


def plot_dwell_buckets(out_dir: Path, tokens: np.ndarray, n_dwell: int = 16):
    from data import DWELL_BASE
    dwell_ids = tokens[(tokens >= DWELL_BASE) & (tokens < DWELL_BASE + n_dwell)] - DWELL_BASE
    counts = np.bincount(dwell_ids.astype(np.int64), minlength=n_dwell)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(n_dwell), counts, color="#ff7f0e")
    ax.set_xlabel("dwell bucket (low → high log-time)")
    ax.set_ylabel("# events")
    ax.set_title(f"Dwell-time bucket distribution (quantile-based, total events={dwell_ids.size:,})")
    ax.set_xticks(np.arange(n_dwell))
    fig.tight_layout()
    fig.savefig(out_dir / "03_dwell_buckets.png", dpi=120)
    plt.close(fig)


def plot_feature_distribution(out_dir: Path, sae: TopKSAE, acts: torch.Tensor):
    with torch.no_grad():
        z = (acts - sae.b_dec) @ sae.W_enc.T + sae.b_enc
        z = torch.relu(z).cpu().numpy()
    max_act = z.max(axis=0)
    frac_fire = (z > 0).mean(axis=0)
    alive = max_act > 1e-6
    n_alive = alive.sum()
    n_total = len(max_act)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Firing rate distribution (alive only)
    ax = axes[0]
    ax.hist(np.log10(frac_fire[alive] + 1e-8), bins=40, color="#1f77b4")
    ax.set_xlabel("log10(firing fraction)")
    ax.set_ylabel("# features")
    ax.set_title(f"Feature firing rate ({n_alive}/{n_total} alive)")

    # Max activation distribution
    ax = axes[1]
    ax.hist(max_act[alive], bins=40, color="#2ca02c")
    ax.set_xlabel("max activation")
    ax.set_ylabel("# features")
    ax.set_title("Feature peak activation")

    # density vs strength scatter
    ax = axes[2]
    ax.scatter(frac_fire[alive], max_act[alive], s=4, alpha=0.5, color="#9467bd")
    ax.set_xscale("log")
    ax.set_xlabel("firing fraction (log)")
    ax.set_ylabel("max activation")
    ax.set_title("density vs peak strength")

    fig.tight_layout()
    fig.savefig(out_dir / "04_feature_distribution.png", dpi=120)
    plt.close(fig)
    return z, max_act, frac_fire


def plot_feature_token_heatmap(
    out_dir: Path, z: np.ndarray, ctx_toks: np.ndarray, vocab: dict[str, int],
    n_top_features: int = 32, n_top_tokens: int = 24,
):
    inv = {v: k for k, v in vocab.items()}
    max_act = z.max(axis=0)
    frac_fire = (z > 0).mean(axis=0)
    safe = np.maximum(frac_fire, 1.0 / z.shape[0])
    score = max_act * np.log(1.0 / safe)
    score[max_act < 1e-6] = -np.inf
    top_feats = np.argsort(-score)[:n_top_features]

    # Per top-feature, find the tokens at which it most often fires
    # Build a matrix: feature x token of mean activation
    # Restrict to top tokens by overall frequency among firing positions.
    fire_mask = z[:, top_feats] > 0
    # tokens occurring in any firing position
    fired_toks_all = ctx_toks[fire_mask.any(axis=1)]
    if fired_toks_all.size == 0:
        return
    counts = np.bincount(fired_toks_all.astype(np.int64), minlength=len(vocab))
    top_toks = np.argsort(-counts)[:n_top_tokens]

    mat = np.zeros((len(top_feats), len(top_toks)))
    for i, tok in enumerate(top_toks):
        mask = ctx_toks == tok
        if mask.sum() == 0:
            continue
        mat[:, i] = z[mask][:, top_feats].mean(axis=0)

    # row-normalize for visibility
    rmax = mat.max(axis=1, keepdims=True) + 1e-9
    mat_n = mat / rmax

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(mat_n, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(top_toks)))
    ax.set_xticklabels([short_label(inv.get(int(t), f"<{int(t)}>"), 22) for t in top_toks],
                       rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(top_feats)))
    ax.set_yticklabels([f"f{int(f)}" for f in top_feats], fontsize=7)
    ax.set_xlabel("context token")
    ax.set_ylabel("feature")
    ax.set_title(f"Top-{len(top_feats)} features × top-{len(top_toks)} tokens (row-norm mean activation)")
    fig.colorbar(im, ax=ax, label="row-normalized mean activation")
    fig.tight_layout()
    fig.savefig(out_dir / "05_feature_token_heatmap.png", dpi=120)
    plt.close(fig)


def plot_feature_pca(out_dir: Path, sae: TopKSAE, max_act: np.ndarray, frac_fire: np.ndarray):
    W_dec = sae.W_dec.detach().cpu().numpy().T  # (dict_size, d_model)
    alive = max_act > 1e-6
    if alive.sum() < 3:
        return
    pca = PCA(n_components=2).fit(W_dec[alive])
    proj = pca.transform(W_dec[alive])

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=np.log10(frac_fire[alive] + 1e-8),
        s=8 + 30 * (max_act[alive] / max_act[alive].max()),
        alpha=0.7, cmap="plasma",
    )
    fig.colorbar(sc, ax=ax, label="log10(firing fraction)")
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"SAE decoder direction PCA  (n_alive={alive.sum()})\nsize ∝ peak activation")
    fig.tight_layout()
    fig.savefig(out_dir / "06_feature_pca.png", dpi=120)
    plt.close(fig)


def plot_feature_trace(
    out_dir: Path, z: np.ndarray, ctx_toks: np.ndarray, win_idx: np.ndarray,
    pos_idx: np.ndarray, vocab: dict[str, int], n_features: int = 24,
):
    """For one window, plot feature activations as a heatmap (feature × position)."""
    inv = {v: k for k, v in vocab.items()}
    # Pick a window in the middle of the dataset (more representative)
    target_win = int(np.median(win_idx))
    sel = win_idx == target_win
    if sel.sum() < 16:
        return
    # Order positions
    order = np.argsort(pos_idx[sel])
    z_w = z[sel][order]
    toks_w = ctx_toks[sel][order]
    pos_w = pos_idx[sel][order]

    # Pick top-N features by activity in this window
    feat_strength = z_w.max(axis=0)
    top = np.argsort(-feat_strength)[:n_features]
    mat = z_w[:, top].T  # (n_features, n_positions)

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xlabel("position in window")
    ax.set_ylabel("top features (by activity in window)")
    ax.set_title(f"Feature trace for window {target_win}  ({mat.shape[1]} positions, {mat.shape[0]} features)")

    # Annotate position axis with token labels every N
    step = max(1, len(pos_w) // 30)
    ticks = np.arange(0, len(pos_w), step)
    labels = [short_label(inv.get(int(toks_w[i]), f"<{int(toks_w[i])}>"), 14) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"f{int(f)}" for f in top], fontsize=7)
    fig.colorbar(im, ax=ax, label="activation")
    fig.tight_layout()
    fig.savefig(out_dir / "07_feature_trace.png", dpi=120)
    plt.close(fig)


def plot_position_profile(out_dir: Path, z: np.ndarray, pos_idx: np.ndarray, max_act: np.ndarray):
    """For each top feature, where in the input window does it tend to fire?"""
    alive = max_act > 1e-6
    if alive.sum() == 0:
        return
    top = np.argsort(-max_act)[:48]
    seq_len = int(pos_idx.max() + 1)
    counts = np.zeros((len(top), seq_len))
    for i, f in enumerate(top):
        sel = z[:, f] > 0
        if sel.sum() == 0:
            continue
        c = np.bincount(pos_idx[sel].astype(np.int64), minlength=seq_len)
        counts[i] = c / max(1, c.max())  # row-normalize for visibility
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(counts, aspect="auto", cmap="cividis")
    ax.set_xlabel("position in window")
    ax.set_ylabel("top features (by peak activation)")
    ax.set_title("Where in the window each top feature fires (row-normalized)")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"f{int(f)}" for f in top], fontsize=6)
    fig.colorbar(im, ax=ax, label="firing density (row-norm)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_position_profile.png", dpi=120)
    plt.close(fig)


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load original name→id vocab (every id has a unique name here).
    vocab = json.loads((out_dir / "vocab.json").read_text())
    # Optional: id→symbolicated_label override map for prettier display.
    sym_path = out_dir / "vocab_sym.json"
    if sym_path.exists():
        id_to_label = {int(k): v for k, v in json.loads(sym_path.read_text()).items()}
        # Build a name→id vocab where the names are the symbolicated labels;
        # if multiple ids map to the same symbol, keep the highest-ranked id.
        # For inverse-lookup purposes we keep the original vocab AND apply the map at display time.
        # We pass `vocab` through to plotting fns but override the inv map.
        # Simplest: replace vocab with a synthetic name→id derived from id_to_label.
        # This keeps "name" lookup the same; inverse becomes id→symbol.
        vocab = {}
        for tid, label in id_to_label.items():
            # Make names unique: append __id when there's a collision.
            base = label
            name = base
            i = 2
            while name in vocab:
                name = f"{base}__{i}"
                i += 1
            vocab[name] = tid

    # 1. Training curves
    log_path = out_dir.parent / "train.log"
    if log_path.exists():
        plot_training(plots_dir, log_path)
        print(f"  [ok] 01_training_curves.png")
    else:
        print(f"  [skip] no train.log at {log_path}")

    # 2. Vocab zipf
    stream = np.load(out_dir / "stream.npz")
    tokens = stream["tokens"]
    plot_vocab_zipf(plots_dir, tokens, vocab)
    print(f"  [ok] 02_vocab_zipf.png")

    # 3. Dwell buckets
    plot_dwell_buckets(plots_dir, tokens)
    print(f"  [ok] 03_dwell_buckets.png")

    # 4-8: load LM + SAE + activations
    lm_blob = torch.load(out_dir / "lm.pt", map_location="cpu", weights_only=False)
    lm_cfg = LMConfig(**lm_blob["cfg"])
    lm = StackLM(lm_cfg)
    lm.load_state_dict(lm_blob["state_dict"])
    lm.eval()

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

    # 4
    z, max_act, frac_fire = plot_feature_distribution(plots_dir, sae, acts)
    print(f"  [ok] 04_feature_distribution.png")

    # 5
    plot_feature_token_heatmap(plots_dir, z, ctx_toks, vocab)
    print(f"  [ok] 05_feature_token_heatmap.png")

    # 6
    plot_feature_pca(plots_dir, sae, max_act, frac_fire)
    print(f"  [ok] 06_feature_pca.png")

    # 7
    plot_feature_trace(plots_dir, z, ctx_toks, win_idx, pos_idx, vocab)
    print(f"  [ok] 07_feature_trace.png")

    # 8
    plot_position_profile(plots_dir, z, pos_idx, max_act)
    print(f"  [ok] 08_position_profile.png")

    print(f"\nall plots written to {plots_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
