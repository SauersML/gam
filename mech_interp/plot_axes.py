"""Find continuous coherent axes for time / dwell / depth across three substrates:
  A) raw residual stream activations (LM hidden state at the captured layer)
  B) SAE-encoded sparse z (TopK output)
  C) SAE decoder directions W_dec (per-feature, NOT per-position — included as control)

Uses Ridge linear probes + a small MLP non-linear probe on (A) and (B). Reports R².
The substrate where the linear probe wins is where the *continuous* axis lives.
Plus, for the winning substrate, derive the 1-D supervised direction and plot
points projected onto it — that's the "discovered time axis" — colored by truth.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from data import DWELL_BASE, N_DWELL
from model import SAEConfig, TopKSAE


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 64, epochs: int = 80,
            lr: float = 1e-2, batch_size: int = 1024, device: str = "cpu") -> "tuple[float, np.ndarray]":
    """Tiny 1-hidden-layer MLP regression. Returns (test R², predictions on test split)."""
    rng = np.random.default_rng(0)
    n = X.shape[0]
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.randint(0, len(tr), (batch_size,))
        xb = Xt[tr][idx].to(device)
        yb = yt[tr][idx].to(device)
        pred = model(xb).squeeze(-1)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pte = model(Xt[te].to(device)).squeeze(-1).cpu().numpy()
    return float(r2_score(y[te], pte)), pte


def main(out_dir: Path):
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    sae_blob = torch.load(out_dir / "sae.pt", map_location="cpu", weights_only=False)
    sae_cfg = SAEConfig(**sae_blob["cfg"])
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_blob["state_dict"], strict=False)
    sae.eval()

    blob = np.load(out_dir / "activations.npz")
    acts = torch.from_numpy(blob["acts"])  # (N, d_model)
    win_idx = blob["win_idx"]
    pos_idx = blob["pos_idx"]
    seq_len = int(blob["seq_len"])

    # SAE-encoded sparse z
    print("computing SAE-encoded sparse z...")
    z_chunks = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, acts.shape[0], bs):
            sparse, _ = sae.encode_topk(acts[i : i + bs])
            z_chunks.append(sparse.cpu().numpy())
    z = np.concatenate(z_chunks, axis=0)  # (N, dict_size)
    A = acts.numpy()  # (N, d_model)
    # Subsample to keep Ridge fast on large z (100k x 4096 is too slow).
    if A.shape[0] > 25000:
        rng_sub = np.random.default_rng(0)
        sub = rng_sub.choice(A.shape[0], 25000, replace=False)
        A = A[sub]
        z = z[sub]
        win_idx = win_idx[sub]
        pos_idx = pos_idx[sub]
    print(f"A: {A.shape}  z: {z.shape}")

    # Per-position attributes from event metadata
    meta = np.load(out_dir / "events_meta.npz")
    tok_start = meta["token_start"]
    start_time_ms = meta["start_time_ms"]
    dwell_ms = meta["dwell_ms"]
    abs_tok = win_idx.astype(np.int64) * seq_len + pos_idx.astype(np.int64)
    event_idx = np.clip(np.searchsorted(tok_start, abs_tok, side="right") - 1, 0, len(tok_start) - 1)
    pos_time_s = start_time_ms[event_idx] / 1000.0
    pos_log_dwell = np.log(dwell_ms[event_idx] + 1e-6)

    stream = np.load(out_dir / "stream.npz")
    tokens_full = stream["tokens"]
    BOS_ID = 1
    bos_at_idx = np.where(tokens_full == BOS_ID)[0]
    j = np.clip(np.searchsorted(bos_at_idx, abs_tok, side="right") - 1, 0, len(bos_at_idx) - 1)
    pos_depth = (abs_tok - bos_at_idx[j]).astype(np.float64)

    # 80/20 split for probes
    rng = np.random.default_rng(0)
    perm = rng.permutation(A.shape[0])
    split = int(0.8 * A.shape[0])
    tr, te = perm[:split], perm[split:]

    targets = {"time_s": pos_time_s, "log_dwell": pos_log_dwell, "stack_depth": pos_depth}

    print("\n=== Ridge linear probes ===")
    print(f"{'target':<14}{'A→y (residual)':>18}{'z→y (SAE)':>14}{'W_dec→meanY':>14}")
    results = {}
    for name, y in targets.items():
        # A) raw residual
        ra = Ridge(alpha=1.0).fit(A[tr], y[tr])
        r2_A = ra.score(A[te], y[te])
        pred_A_te = ra.predict(A[te])
        # B) sparse z
        rz = Ridge(alpha=1.0).fit(z[tr], y[tr])
        r2_z = rz.score(z[te], y[te])
        pred_z_te = rz.predict(z[te])
        # C) (just for reference) what W_dec → mean-y predicts is in plot_time_color earlier; skip here.
        print(f"{name:<14}{r2_A:>18.3f}{r2_z:>14.3f}")
        results[name] = {
            "ridge_A_R2": float(r2_A),
            "ridge_z_R2": float(r2_z),
            "ridge_A_dir": ra.coef_,  # 1D direction in d_model space
            "y_te": y[te],
            "pred_A_te": pred_A_te,
            "pred_z_te": pred_z_te,
        }

    print("\n=== Nonlinear MLP probes (residual A) ===")
    for name, y in targets.items():
        # Tiny MLP, fewer epochs — just a sanity check vs Ridge.
        r2, pte = fit_mlp(A, y, hidden=32, epochs=30, device="cpu")
        results[name]["mlp_A_R2"] = float(r2)
        results[name]["mlp_A_pred_te"] = pte
        print(f"  A → {name:<14} MLP R² = {r2:.3f}")

    # Save numerical summary
    summary = {
        name: {k: float(v) if not isinstance(v, np.ndarray) else None for k, v in d.items() if k.endswith("R2")}
        for name, d in results.items()
    }
    (plots_dir / "27_axis_probes.json").write_text(json.dumps(summary, indent=2))

    # ---------- 27: bar chart of probe R² across substrates ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    methods = ["W_dec→meanY (per-feat)", "Ridge A→y (residual)", "Ridge z→y (SAE sparse)", "MLP A→y (residual)"]
    # W_dec→meanY scores are from the prior plot_time_color step — recompute here for completeness.
    W_dec = sae.W_dec.detach().cpu().numpy().T
    max_act_proxy = z.max(axis=0)
    alive_idx = np.where(max_act_proxy > 1e-6)[0]
    Wn_alive = W_dec[alive_idx] / (np.linalg.norm(W_dec[alive_idx], axis=1, keepdims=True) + 1e-9)
    feat_act_sum = z[:, alive_idx].sum(axis=0) + 1e-9

    feat_means: dict[str, np.ndarray] = {}
    for name, y in targets.items():
        feat_means[name] = (z[:, alive_idx].T @ y) / feat_act_sum

    rng2 = np.random.default_rng(1)
    perm2 = rng2.permutation(len(alive_idx))
    split2 = int(0.8 * len(alive_idx))
    tr2, te2 = perm2[:split2], perm2[split2:]
    wdec_r2 = {}
    for name in targets:
        rw = Ridge(alpha=1.0).fit(Wn_alive[tr2], feat_means[name][tr2])
        wdec_r2[name] = rw.score(Wn_alive[te2], feat_means[name][te2])
    print(f"\nW_dec→mean(target) R² (per-feat targets):")
    for name, v in wdec_r2.items():
        print(f"  W_dec → {name}  R² = {v:.3f}")

    bars = np.array(
        [
            [wdec_r2[name] for name in targets],
            [results[name]["ridge_A_R2"] for name in targets],
            [results[name]["ridge_z_R2"] for name in targets],
            [results[name]["mlp_A_R2"] for name in targets],
        ]
    )
    x = np.arange(len(targets))
    w = 0.18
    palette = ["#bbbbbb", "#1f77b4", "#2ca02c", "#d62728"]
    for i, (vals, color, lbl) in enumerate(zip(bars, palette, methods)):
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, label=lbl)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(targets.keys()))
    ax.set_ylabel("test R²")
    ax.set_title("How decodable is each attribute? — probes across SAE / residual stream")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "27_probe_r2.png", dpi=130)
    plt.close(fig)
    print("  [ok] 27_probe_r2.png")

    # ---------- 28: discovered axes — predicted vs actual on test set ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for j, (name, y) in enumerate(targets.items()):
        d = results[name]
        # row 0: linear A
        ax = axes[0, j]
        ax.scatter(d["y_te"], d["pred_A_te"], s=4, alpha=0.3, color="#1f77b4")
        lo, hi = min(d["y_te"].min(), d["pred_A_te"].min()), max(d["y_te"].max(), d["pred_A_te"].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.7)
        ax.set_xlabel(f"actual {name}")
        ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"Ridge A → {name}  R²={d['ridge_A_R2']:.3f}")

        # row 1: MLP A
        ax = axes[1, j]
        ax.scatter(d["y_te"], d["mlp_A_pred_te"], s=4, alpha=0.3, color="#d62728")
        lo, hi = min(d["y_te"].min(), d["mlp_A_pred_te"].min()), max(d["y_te"].max(), d["mlp_A_pred_te"].max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.7)
        ax.set_xlabel(f"actual {name}")
        ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"MLP A → {name}  R²={d['mlp_A_R2']:.3f}")
    fig.suptitle("Linear vs nonlinear decodability of attributes from residual stream", y=1.005)
    fig.tight_layout()
    fig.savefig(plots_dir / "28_predicted_vs_actual.png", dpi=130)
    plt.close(fig)
    print("  [ok] 28_predicted_vs_actual.png")

    # ---------- 29: 1D supervised axis — project A onto Ridge direction, color by attribute ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (name, y) in enumerate(targets.items()):
        direction = results[name]["ridge_A_dir"]
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        proj = A @ direction  # (N,)
        # Sort positions by projection for a clean strip plot
        order = np.argsort(proj)
        ax = axes[j]
        ax.scatter(proj[order], y[order], s=2, alpha=0.3, color="#9467bd")
        ax.set_xlabel(f"residual stream projected onto Ridge({name}) direction")
        ax.set_ylabel(f"actual {name}")
        ax.set_title(f"Discovered axis for {name}  (corr = {np.corrcoef(proj, y)[0, 1]:.3f})")
    fig.suptitle("1-D supervised axes in the residual stream", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "29_supervised_axis.png", dpi=130)
    plt.close(fig)
    print("  [ok] 29_supervised_axis.png")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
