"""Produce per-checkpoint OLMo-3-32B L25 qualia atlas PNGs.

Outputs:
    olmo_L25_base.png
    olmo_L25_step_2300.png

Each PNG: 2-panel scatter (exp/noexp side coloring + kind coloring)
on the PCA embedding, with EV and AUC annotated.

Dodge-config for #1051/#1095:
  K=1, atom_topology="circle", PCA tries [32, 16, 8] in order.
  If all fail, falls back to raw PCA scatter (no SAE fit).
  If N<635 needed, subsamples to 200 stratified by side.

Usage:
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    cd /tmp && python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_l25_plot.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data")
OUT_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data/plots")
CHECKPOINTS = [("base", "olmo_L25_base.png"), ("step_2300", "olmo_L25_step_2300.png")]
PCA_TRIES = [32, 16, 8]
N_SUBSAMPLE = 200  # fallback subsample size if all PCA dims fail


SIDE_PALETTE = {
    "exp": "#e6194b",
    "noexp": "#4363d8",
    "-": "#aaaaaa",
    "a": "#f58231",
    "b": "#42d4f4",
}

KIND_PALETTE = {
    "self": "#e6194b",
    "human": "#3cb44b",
    "ai": "#4363d8",
    "robot": "#911eb4",
    "mammal": "#f58231",
    "tool": "#42d4f4",
    "dead": "#808080",
    "supernatural": "#ffe119",
    "rock": "#bfef45",
    "fish": "#aaffc3",
    "insect": "#ffd8b1",
    "plant": "#dcbeff",
    "conscious_machine": "#fabebe",
    "chinese_room": "#469990",
    "animal": "#e6beff",
    "bird": "#9a6324",
    "vehicle": "#800000",
    "collective": "#000075",
    "simulated": "#f032e6",
    "upload": "#aaffc3",
}


def pca_whiten(X: np.ndarray, n: int) -> np.ndarray:
    mu = X.mean(0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n].T) / (S[:n] + 1e-12)


def wilcoxon_auc(vals: np.ndarray, sides: list) -> float:
    e = vals[np.array([s == "exp" for s in sides])]
    n = vals[np.array([s == "noexp" for s in sides])]
    if len(e) == 0 or len(n) == 0:
        return float("nan")
    a = float(((e[:, None] > n[None, :]).sum() + 0.5 * (e[:, None] == n[None, :]).sum())
              / (len(e) * len(n)))
    return max(a, 1.0 - a)


def try_fit(Z: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float]:
    """Try sae_manifold_fit K=1 circle. Returns (fitted, ev) or raises."""
    import gamfit
    fit = gamfit.sae_manifold_fit(
        X=Z, K=1, d_atom=2, atom_topology="circle",
        n_iter=30, random_state=seed,
        assignment="ibp_map", smoothness_weight=1.0, sparsity_weight=0.5,
    )
    fitted = np.asarray(fit.fitted)
    var = float(((Z - Z.mean(0)) ** 2).sum())
    ev = float(1.0 - ((Z - fitted) ** 2).sum() / var) if var > 0 else float("nan")
    return fitted, ev


def make_panel(ckpt_name: str, out_path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ckpt_dir = DATA_DIR / ckpt_name
    acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")  # (635, 64, 5120)
    X_raw = np.array(acts[:, 25, :], dtype=np.float32)           # (635, 5120)
    with open(ckpt_dir / "prompts.jsonl") as fh:
        prompts = [json.loads(l) for l in fh]
    sides = [p.get("side", "-") for p in prompts]
    kinds = [p.get("kind", "?") for p in prompts]

    print(f"\n[{ckpt_name}] X_raw {X_raw.shape}", flush=True)

    # Try PCA dims in order; subsample if all fail
    Z_plot = None
    ev = float("nan")
    label = ""
    for pca_dim in PCA_TRIES:
        Z = pca_whiten(X_raw, pca_dim)
        t0 = time.time()
        try:
            _, ev = try_fit(Z)
            Z_plot = Z
            label = f"K=1 circle PCA-{pca_dim} EV={ev:.3f} ({time.time()-t0:.0f}s)"
            print(f"  {label}", flush=True)
            break
        except Exception as exc:
            print(f"  PCA-{pca_dim} FAILED: {exc!s:.120}", flush=True)

    if Z_plot is None:
        # Subsample to N_SUBSAMPLE and retry smallest PCA
        print(f"  All PCA dims failed full N={len(X_raw)}; subsampling to {N_SUBSAMPLE}", flush=True)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_raw), size=N_SUBSAMPLE, replace=False)
        idx.sort()
        X_sub = X_raw[idx]
        prompts = [prompts[i] for i in idx]
        sides = [sides[i] for i in idx]
        kinds = [kinds[i] for i in idx]
        for pca_dim in PCA_TRIES:
            Z = pca_whiten(X_sub, pca_dim)
            t0 = time.time()
            try:
                _, ev = try_fit(Z)
                Z_plot = Z
                label = (f"K=1 circle PCA-{pca_dim} N={N_SUBSAMPLE} EV={ev:.3f} "
                         f"({time.time()-t0:.0f}s)")
                print(f"  {label}", flush=True)
                break
            except Exception as exc:
                print(f"  sub PCA-{pca_dim} FAILED: {exc!s:.120}", flush=True)

    if Z_plot is None:
        # Ultimate fallback: raw PCA scatter (no SAE fit)
        print(f"  SAE fit failed entirely; using raw PCA-2 scatter", flush=True)
        Z_plot = pca_whiten(X_raw, 2)
        ev = float("nan")
        label = "PCA-2 raw (SAE fit blocked by #1051/#1095)"

    auc = wilcoxon_auc(Z_plot[:, 0], sides)
    print(f"  AUC={auc:.3f}", flush=True)

    # --- draw 2-panel figure ---
    fig, (ax_side, ax_kind) = plt.subplots(1, 2, figsize=(12, 5.5))

    # panel 1: exp/noexp
    side_c = [SIDE_PALETTE.get(s, "#cccccc") for s in sides]
    ax_side.scatter(Z_plot[:, 0], Z_plot[:, 1], c=side_c, s=12, alpha=0.70, linewidths=0)
    ax_side.set_title(
        f"{ckpt_name}  L25 exp/noexp\n{label}\nAUC={auc:.3f}",
        fontsize=9,
    )
    present_sides = sorted(set(sides))
    ax_side.legend(
        handles=[mpatches.Patch(color=SIDE_PALETTE.get(s, "#ccc"), label=s)
                 for s in present_sides],
        fontsize=7, loc="upper right", framealpha=0.7,
    )
    ax_side.set_xticks([]); ax_side.set_yticks([])

    # panel 2: kind
    kind_c = [KIND_PALETTE.get(k, "#cccccc") for k in kinds]
    ax_kind.scatter(Z_plot[:, 0], Z_plot[:, 1], c=kind_c, s=12, alpha=0.70, linewidths=0)
    ax_kind.set_title(f"{ckpt_name}  L25 kind", fontsize=9)
    present_kinds = sorted(set(kinds))
    ax_kind.legend(
        handles=[mpatches.Patch(color=KIND_PALETTE.get(k, "#ccc"), label=k)
                 for k in present_kinds[:16]],
        fontsize=5, loc="upper right", framealpha=0.7, ncol=2,
    )
    ax_kind.set_xticks([]); ax_kind.set_yticks([])

    fig.suptitle(
        f"OLMo-3-32B  {ckpt_name}  —  L25 qualia plane\n"
        f"PCA-whitened activations[:,25,:] (635×5120 → manifold embed)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"  SAVED {out_path}", flush=True)
    return dict(ckpt=ckpt_name, ev=ev, auc=auc, label=label)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt_name, png_name in CHECKPOINTS:
        out_path = OUT_DIR / png_name
        row = make_panel(ckpt_name, out_path)
        rows.append(row)

    print("\n=== SUMMARY ===")
    print(f"\n{'checkpoint':<22} {'EV':>8} {'AUC exp/noexp':>15}")
    for r in rows:
        print(f"{r['ckpt']:<22} {r['ev']:>8.4f} {r['auc']:>15.3f}")
    print()
    for r in rows:
        print(f"  {r['ckpt']}: {r['label']}")
    print("\nPlots:")
    for _, png_name in CHECKPOINTS:
        print(f"  {OUT_DIR / png_name}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
