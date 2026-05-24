"""Two more cool plots:
  13_temporal_raster.png — feature activations over wall-clock profile time,
                           with sub-benchmark phase boundaries detected from gaps.
  14_logit_lens.png      — for each top feature, what tokens does its decoder
                           direction promote when projected through the LM head?
                           (text + bar plot of top-k voted tokens per feature)
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

from data import DWELL_BASE, N_DWELL
from model import LMConfig, SAEConfig, StackLM, TopKSAE


def short_label(s: str, n: int = 36) -> str:
    s = re.sub(r"::h[0-9a-f]{8,}", "", s)
    s = s.replace("$", r"\$")  # avoid matplotlib mathtext parsing
    if len(s) <= n:
        return s
    parts = s.split("::")
    if len(parts) >= 3:
        return "..::" + "::".join(parts[-2:])[: n - 4]
    return s[: n - 1] + "…"


# ---------- temporal raster ----------

def plot_temporal_raster(out_dir: Path, plots_dir: Path):
    """For each top SAE feature, plot its mean activation over wall-clock time,
    binned into ~400 time bins. Annotate phase boundaries (long-dwell gaps)."""
    sae_blob = torch.load(out_dir / "sae.pt", map_location="cpu", weights_only=False)
    sae_cfg = SAEConfig(**sae_blob["cfg"])
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_blob["state_dict"], strict=False)
    sae.eval()

    blob = np.load(out_dir / "activations.npz")
    acts = torch.from_numpy(blob["acts"])
    win_idx = blob["win_idx"]
    pos_idx = blob["pos_idx"]
    seq_len = int(blob["seq_len"])

    # Per-position TopK feature activations
    with torch.no_grad():
        sparse, _ = sae.encode_topk(acts)
        z = sparse.cpu().numpy()  # (N, dict_size)

    max_act = z.max(axis=0)
    alive = np.where(max_act > 1e-6)[0]
    print(f"alive features for raster: {len(alive)}")

    # Map each (win, pos) → token-stream index → wall-clock time via events_meta
    meta = np.load(out_dir / "events_meta.npz")
    tok_start = meta["token_start"]
    tok_end = meta["token_end"]
    start_time = meta["start_time_ms"]
    dwell_ms = meta["dwell_ms"]

    # Each window in the harvest covers tokens [win_idx*seq_len, win_idx*seq_len + seq_len).
    abs_tok = win_idx.astype(np.int64) * seq_len + pos_idx.astype(np.int64)

    # For each abs_tok, find the event it belongs to via searchsorted on tok_start.
    # An event spans tok_start[e] .. tok_end[e]. We want the largest e with tok_start[e] <= abs_tok.
    event_idx = np.searchsorted(tok_start, abs_tok, side="right") - 1
    event_idx = np.clip(event_idx, 0, len(tok_start) - 1)
    pos_time = start_time[event_idx]  # ms

    # Detect phase boundaries: criterion sub-benchmarks are separated by ~100ms+ gaps
    # (warmup/measurement/analysis stages). Real bench transitions are very rare and large.
    gaps = np.diff(start_time)
    gap_thresh = max(200.0, float(np.percentile(gaps, 99.99)))
    phase_breaks = start_time[1:][gaps > gap_thresh]
    print(f"detected {len(phase_breaks)} phase boundaries (gap > {gap_thresh:.1f}ms)")

    # Pick top-N features by peak activation
    top_n = 36
    top = alive[np.argsort(-max_act[alive])[:top_n]]

    # Bin time into ~600 bins; per (feature, bin) take mean activation
    n_bins = 600
    t_min, t_max = float(pos_time.min()), float(pos_time.max())
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, pos_time, side="right") - 1, 0, n_bins - 1)

    raster = np.zeros((len(top), n_bins), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(counts, bin_idx, 1)
    for i, f in enumerate(top):
        col = z[:, f]
        sums = np.zeros(n_bins, dtype=np.float64)
        np.add.at(sums, bin_idx, col)
        raster[i] = sums / np.maximum(counts, 1)

    # Row-normalize for visibility
    rmax = raster.max(axis=1, keepdims=True) + 1e-9
    raster_n = raster / rmax

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(
        raster_n, aspect="auto", cmap="magma",
        extent=[t_min / 1000.0, t_max / 1000.0, len(top), 0],
        interpolation="nearest",
    )
    for tb in phase_breaks:
        ax.axvline(tb / 1000.0, color="cyan", lw=0.7, alpha=0.6, linestyle="--")
    ax.set_xlabel("profile time (s)")
    ax.set_ylabel("top features (by peak activation)")
    ax.set_title(
        f"Temporal feature raster — {len(top)} features × {n_bins} time bins  "
        f"(cyan = phase boundaries, {len(phase_breaks)} sub-benchmarks)"
    )
    fig.colorbar(im, ax=ax, label="row-normalized mean activation")
    fig.tight_layout()
    fig.savefig(plots_dir / "13_temporal_raster.png", dpi=130)
    plt.close(fig)
    print("  [ok] 13_temporal_raster.png")


# ---------- logit lens ----------

def plot_logit_lens(out_dir: Path, plots_dir: Path):
    """For each top SAE feature, project its decoder direction through the LM
    head (final LayerNorm + tied unembed) and report the top voted tokens."""
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
    with torch.no_grad():
        sparse, _ = sae.encode_topk(acts)
        z = sparse.cpu().numpy()

    max_act = z.max(axis=0)
    frac_fire = (z > 0).mean(axis=0)
    alive = max_act > 1e-6
    safe = np.maximum(frac_fire, 1.0 / z.shape[0])
    score = max_act * np.log(1.0 / safe)
    score[~alive] = -np.inf
    n_top = 18
    top = np.argsort(-score)[:n_top]

    sym = json.loads((out_dir / "vocab_sym.json").read_text())
    id_to_label = {int(k): v for k, v in sym.items()}

    # Logit lens: head(ln_f(W_dec[:, f]))
    W_dec = sae.W_dec  # (d_model, dict_size)
    with torch.no_grad():
        directions = W_dec[:, top].T  # (n_top, d_model)
        # Apply final LN + tied head
        normed = lm.ln_f(directions)
        logits = lm.head(normed)  # (n_top, vocab_size)
        # Take top-k voted tokens per feature
        top_k = 5
        vals, idx = logits.topk(top_k, dim=-1)

    fig, axes = plt.subplots(n_top // 3, 3, figsize=(18, n_top // 3 * 1.6))
    axes = axes.flat
    for i, f in enumerate(top):
        ax = axes[i]
        v = vals[i].cpu().numpy()
        ii = idx[i].cpu().numpy()
        labels = [short_label(id_to_label.get(int(t), f"id{int(t)}"), 24) for t in ii]
        colors = plt.cm.tab10(i % 10)
        ax.barh(np.arange(top_k)[::-1], v, color=colors, edgecolor="none")
        ax.set_yticks(np.arange(top_k)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            f"f{int(f)}  max={max_act[f]:.2f}  fire={frac_fire[f]*100:.1f}%",
            fontsize=9,
        )
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(
        f"Logit lens — top-{n_top} features × top-5 voted tokens "
        "(decoder direction → ln_f → tied unembed)",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "14_logit_lens.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] 14_logit_lens.png")

    # Also write a text version for reading
    lines = []
    for i, f in enumerate(top):
        v = vals[i].cpu().numpy()
        ii = idx[i].cpu().numpy()
        bits = ", ".join(
            f"{short_label(id_to_label.get(int(t), f'id{int(t)}'), 30)}({v[k]:.2f})"
            for k, t in enumerate(ii)
        )
        lines.append(
            f"f{int(f):4d}  max={max_act[f]:6.2f}  fire={frac_fire[f]*100:5.2f}%  → {bits}"
        )
    (plots_dir / "14_logit_lens.txt").write_text("\n".join(lines) + "\n")
    print("  [ok] 14_logit_lens.txt")


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_temporal_raster(out_dir, plots_dir)
    plot_logit_lens(out_dir, plots_dir)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
