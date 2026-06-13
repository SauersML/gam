"""Fit gam's manifold-SAE on OLMo-3-32B residual-stream activations.

Checkpoints in olmo_data/:
    base          — pretrained base model (no RLVR)
    step_2300     — final RLVR checkpoint (OLMo-3-32B instruct ancestor)
    stage1-step0  — SFT stage1 init
    stage3-step11921 — SFT stage3 end
    instruct      — final SFT instruct

Layout per checkpoint:
    activations.npy   (635, 64, 5120) float32  — qualia bank (L25 = idx 25)
    prompts.jsonl     635 prompts with role/side/kind
    extra/ (some ckpts) activations.npy (180, 64, 5120) float32 — color bank (L44 = idx 44)
    extra/ prompts.jsonl with color/hex/rgb fields

Analyses:
  1. L25 qualia atlas — base vs step_2300 (4 panels: side, kind × 2 checkpoints)
  2. L44 color trajectory — 3 checkpoints with extra/, colored by hex
  3. RGB-R² curve across those checkpoints

Usage (MSI, from /tmp to avoid gamfit source-tree shadowing):
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    cd /tmp && python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_sae_atlas.py \\
        --data_dir /projects/standard/hsiehph/sauer354/olmo_data \\
        --out_dir  /projects/standard/hsiehph/sauer354/olmo_data/plots
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def load_l25(ckpt_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (635, 5120) float32 L25 slice and prompts list."""
    acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")  # (635, 64, 5120)
    l25 = np.array(acts[:, 25, :], dtype=np.float32)
    with open(ckpt_dir / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh]
    return l25, prompts


def load_l44_color(ckpt_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (180, 5120) float32 L44 slice and color prompts."""
    extra = ckpt_dir / "extra"
    acts = np.load(extra / "activations.npy", mmap_mode="r")  # (180, 64, 5120)
    l44 = np.array(acts[:, 44, :], dtype=np.float32)
    with open(extra / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh]
    return l44, prompts


# ---------------------------------------------------------------------------
# PCA whitening
# ---------------------------------------------------------------------------

def pca_project(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z, Vt, mu): PCA-whitened (N, n_components) float64."""
    mu = X.mean(axis=0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:n_components]
    S_trunc = S[:n_components]
    Z = (Xc @ Vt.T) / (S_trunc[np.newaxis, :] + 1e-12)
    return Z, Vt, mu


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def exp_noexp_auc(coord: np.ndarray, sides: list[str]) -> float:
    """Wilcoxon AUC separating exp from noexp on first coordinate."""
    exp_mask = np.array([s == "exp" for s in sides])
    noexp_mask = np.array([s == "noexp" for s in sides])
    if exp_mask.sum() == 0 or noexp_mask.sum() == 0:
        return float("nan")
    ev = coord[exp_mask]
    nv = coord[noexp_mask]
    n = len(ev) * len(nv)
    auc = float(
        ((ev[:, None] > nv[None, :]).sum() + 0.5 * (ev[:, None] == nv[None, :]).sum())
        / n
    )
    return max(auc, 1.0 - auc)


def color_rgb_r2(Z: np.ndarray, prompts: list[dict]) -> float:
    """R² of linear fit (PC1, PC2) → (R, G, B)/255."""
    try:
        rgb = np.array([p["rgb"] for p in prompts], dtype=np.float64) / 255.0
        Xf = np.column_stack([np.ones(len(Z)), Z[:, :2]])
        coef, _, _, _ = np.linalg.lstsq(Xf, rgb, rcond=None)
        pred = Xf @ coef
        ss_res = float(((rgb - pred) ** 2).sum())
        ss_tot = float(((rgb - rgb.mean(0)) ** 2).sum())
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# SAE fit wrapper
# ---------------------------------------------------------------------------

def fit_slice(
    X_raw: np.ndarray,
    pca_dim: int,
    n_atoms: int,
    n_iter: int,
    seed: int,
) -> dict:
    import gamfit

    t0 = time.time()
    Z, Vt, mu = pca_project(X_raw, pca_dim)
    fit = gamfit.sae_manifold_fit(
        X=Z,
        K=n_atoms,
        d_atom=2,
        atom_topology="circle",
        n_iter=n_iter,
        random_state=seed,
        assignment="ibp_map",
        smoothness_weight=1.0,
        sparsity_weight=0.5,
    )
    fitted = np.asarray(fit.fitted)
    total_var = float(((Z - Z.mean(0)) ** 2).sum())
    ev = float(1.0 - ((Z - fitted) ** 2).sum() / total_var) if total_var > 0 else float("nan")
    asn = np.asarray(fit.assignments)
    hard = asn.argmax(axis=1) if asn.ndim == 2 else asn
    k_active = int(asn.shape[1]) if asn.ndim == 2 else n_atoms
    return dict(
        Z=Z, fitted=fitted, hard=hard,
        ev=ev, seconds=time.time() - t0, n_atoms=k_active,
    )


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

SIDE_PALETTE: dict[str, str] = {
    "exp": "#e6194b",
    "noexp": "#4363d8",
    "-": "#aaaaaa",
    "a": "#f58231",
    "b": "#42d4f4",
}

KIND_PALETTE: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def draw_qualia_side(ax, Z: np.ndarray, prompts: list[dict], ckpt: str, ev: float, secs: float) -> None:
    import matplotlib.patches as mpatches

    sides = [p.get("side", "-") for p in prompts]
    colors = [SIDE_PALETTE.get(s, "#cccccc") for s in sides]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=10, alpha=0.65, linewidths=0)
    auc = exp_noexp_auc(Z[:, 0], sides)
    ax.set_title(f"{ckpt}  L25 side\nEV={ev:.3f}  AUC={auc:.3f}  {secs:.0f}s", fontsize=8)
    present = sorted(set(sides))
    handles = [mpatches.Patch(color=SIDE_PALETTE.get(s, "#cccccc"), label=s) for s in present]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])


def draw_qualia_kind(ax, Z: np.ndarray, prompts: list[dict], ckpt: str) -> None:
    import matplotlib.patches as mpatches

    kinds = [p.get("kind", "?") for p in prompts]
    colors = [KIND_PALETTE.get(k, "#cccccc") for k in kinds]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=10, alpha=0.65, linewidths=0)
    ax.set_title(f"{ckpt}  L25 kind", fontsize=8)
    present = sorted(set(kinds))
    handles = [mpatches.Patch(color=KIND_PALETTE.get(k, "#cccccc"), label=k) for k in present[:16]]
    ax.legend(handles=handles, fontsize=5, loc="upper right", framealpha=0.7, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])


def draw_color_tile(ax, Z: np.ndarray, prompts: list[dict], ckpt: str, ev: float,
                    rgb_r2: float, secs: float) -> None:
    hexcols = [p.get("hex", "#cccccc") for p in prompts]
    # ensure #-prefixed
    hexcols = [h if h.startswith("#") else "#" + h for h in hexcols]
    ax.scatter(Z[:, 0], Z[:, 1], c=hexcols, s=18, alpha=0.80, linewidths=0)
    ax.set_title(f"{ckpt}  L44 color\nEV={ev:.3f}  RGB-R²={rgb_r2:.3f}  {secs:.0f}s", fontsize=8)
    # label color names at centroids
    color_names = [p.get("color", "") for p in prompts]
    for c_name in sorted(set(color_names)):
        mask = np.array([cn == c_name for cn in color_names])
        if mask.sum() == 0:
            continue
        cx, cy = float(Z[mask, 0].mean()), float(Z[mask, 1].mean())
        ax.text(cx, cy, c_name, fontsize=5, ha="center", va="bottom", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5))
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/projects/standard/hsiehph/sauer354/olmo_data")
    parser.add_argument("--out_dir", default="/projects/standard/hsiehph/sauer354/olmo_data/plots")
    parser.add_argument("--pca_dim", type=int, default=32,
                        help="PCA components (32 recommended to condition inner block)")
    parser.add_argument("--n_atoms", type=int, default=1,
                        help="number of atoms (1 = dodge #1051 multi-atom timeout)")
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # 1. L25 qualia: base vs step_2300
    # -----------------------------------------------------------------------
    qualia_ckpts = [("base", data / "base"), ("step_2300", data / "step_2300")]
    qualia_rows: list[dict] = []
    for name, ckpt_dir in qualia_ckpts:
        print(f"\n=== L25 qualia: {name} ===", flush=True)
        X_raw, prompts = load_l25(ckpt_dir)
        print(f"  shape {X_raw.shape} → PCA-{args.pca_dim}", flush=True)
        res = fit_slice(X_raw, args.pca_dim, args.n_atoms, args.n_iter, args.seed)
        sides = [p.get("side", "-") for p in prompts]
        auc = exp_noexp_auc(res["Z"][:, 0], sides)
        print(f"  EV={res['ev']:.4f}  AUC={auc:.3f}  K={res['n_atoms']}  {res['seconds']:.1f}s",
              flush=True)
        qualia_rows.append(dict(name=name, res=res, prompts=prompts, auc=auc))

    # draw 2×2 qualia figure: rows = side/kind, cols = base/step_2300
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    for col_i, row in enumerate(qualia_rows):
        r = row["res"]
        draw_qualia_side(axes1[0, col_i], r["Z"], row["prompts"], row["name"],
                         r["ev"], r["seconds"])
        draw_qualia_kind(axes1[1, col_i], r["Z"], row["prompts"], row["name"])
    fig1.suptitle(
        "OLMo-3-32B — L25 qualia plane: base (pretrained) vs step_2300 (RLVR)\n"
        f"PCA-{args.pca_dim} → sae_manifold_fit K={args.n_atoms}",
        fontsize=11,
    )
    fig1.tight_layout()
    p1 = out / "olmo_qualia_atlas.png"
    fig1.savefig(str(p1), dpi=130)
    plt.close(fig1)
    print(f"\nQUALIA ATLAS SAVED {p1}", flush=True)

    # -----------------------------------------------------------------------
    # 2. L44 color trajectory
    # -----------------------------------------------------------------------
    color_ckpts = [
        ("stage1-step0", data / "stage1-step0"),
        ("stage3-step11921", data / "stage3-step11921"),
        ("step_2300", data / "step_2300"),
    ]
    color_rows: list[dict] = []
    for name, ckpt_dir in color_ckpts:
        extra = ckpt_dir / "extra"
        if not (extra / "activations.npy").exists():
            print(f"  SKIP {name}: no extra/activations.npy", flush=True)
            continue
        print(f"\n=== L44 color: {name} ===", flush=True)
        X_raw, prompts = load_l44_color(ckpt_dir)
        print(f"  shape {X_raw.shape} → PCA-{args.pca_dim}", flush=True)
        res = fit_slice(X_raw, args.pca_dim, args.n_atoms, args.n_iter, args.seed)
        r2 = color_rgb_r2(res["Z"], prompts)
        print(f"  EV={res['ev']:.4f}  RGB-R²={r2:.3f}  K={res['n_atoms']}  {res['seconds']:.1f}s",
              flush=True)
        color_rows.append(dict(name=name, res=res, prompts=prompts, rgb_r2=r2))

    # draw color trajectory grid
    if color_rows:
        n_cols = len(color_rows)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
        ax_flat = np.asarray(axes2).ravel()
        for i, row in enumerate(color_rows):
            r = row["res"]
            draw_color_tile(ax_flat[i], r["Z"], row["prompts"], row["name"],
                            r["ev"], row["rgb_r2"], r["seconds"])
        fig2.suptitle(
            f"OLMo-3-32B — L44 color geometry across SFT→RLVR stages\n"
            f"PCA-{args.pca_dim} → sae_manifold_fit K={args.n_atoms}  "
            f"(RGB-R² = PC1/PC2 → color alignment)",
            fontsize=11,
        )
        fig2.tight_layout()
        p2 = out / "olmo_color_trajectory.png"
        fig2.savefig(str(p2), dpi=130)
        plt.close(fig2)
        print(f"\nCOLOR TRAJECTORY SAVED {p2}", flush=True)

        # RGB-R² curve
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        xs = list(range(len(color_rows)))
        labels = [r["name"] for r in color_rows]
        r2s = [r["rgb_r2"] for r in color_rows]
        evs = [r["res"]["ev"] for r in color_rows]
        ax3.plot(xs, r2s, "o-", label="RGB-R² (color alignment)", color="#e6194b", lw=2)
        ax3.plot(xs, evs, "s--", label="L44 EV (SAE reconstruction)", color="#4363d8", lw=2)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylabel("metric")
        ax3.set_title("Color geometry across OLMo-3-32B training stages")
        ax3.legend()
        ax3.set_ylim(0, 1)
        fig3.tight_layout()
        p3 = out / "olmo_color_r2_curve.png"
        fig3.savefig(str(p3), dpi=130)
        plt.close(fig3)
        print(f"R² CURVE SAVED {p3}", flush=True)

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n=== SUMMARY ===", flush=True)
    print(f"\n{'checkpoint':<22} {'layer':<6} {'EV':>8} {'AUC':>8} {'RGB-R²':>8} {'K':>4}")
    for row in qualia_rows:
        r = row["res"]
        print(f"{row['name']:<22} {'L25':<6} {r['ev']:>8.4f} {row['auc']:>8.3f} {'n/a':>8} {r['n_atoms']:>4}")
    for row in color_rows:
        r = row["res"]
        print(f"{row['name']:<22} {'L44':<6} {r['ev']:>8.4f} {'n/a':>8} {row['rgb_r2']:>8.3f} {r['n_atoms']:>4}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
