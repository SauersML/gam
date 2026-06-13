"""Fit gam's manifold-SAE on OLMo-3-32B residual-stream activations.

Two analyses:
  1. Qualia (step10790 only, the one checkpoint with the full qualia bank):
     L25 colored by exp/noexp and by kind.
  2. Color trajectory (all 11 SFT checkpoints, extra/ color bank at L44):
     Compares color-representation geometry across SFT fine-tuning steps.

PCA-whitens to pca_dim before fitting.

Usage (on MSI):
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_sae_atlas.py \\
        --banks_dir /projects/standard/hsiehph/sauer354/gam_data/banks \\
        --out_dir  /projects/standard/hsiehph/sauer354/olmo_data/plots
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_qualia_l25(bank_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (N, 5120) float32 L25 slice + prompts list. Requires activations.npy."""
    acts = np.load(bank_dir / "activations.npy")   # (635, 64, 5120)
    l25 = acts[:, 25, :].astype(np.float32)
    with open(bank_dir / "prompts.jsonl") as f:
        prompts = [json.loads(line) for line in f]
    return l25, prompts


def load_color_l44(bank_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (180, 5120) float32 L44 color bank slice + color prompts."""
    extra = bank_dir / "extra"
    acts = np.load(extra / "activations.npy")       # (180, 64, 5120)
    l44 = acts[:, 44, :].astype(np.float32)
    with open(extra / "prompts.jsonl") as f:
        prompts = [json.loads(line) for line in f]
    return l44, prompts


def pca_project(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z, Vt, mean): PCA-whitened (N, n_components) float64."""
    mu = X.mean(axis=0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:n_components]
    S = S[:n_components]
    Z = (Xc @ Vt.T) / S[np.newaxis, :]
    return Z, Vt, mu


def exp_noexp_auc(coord: np.ndarray, sides: list[str]) -> float:
    """Wilcoxon AUC separating 'exp' from 'noexp' rows on a 1-D coordinate."""
    exp_mask = np.array([s == "exp" for s in sides])
    noexp_mask = np.array([s == "noexp" for s in sides])
    if exp_mask.sum() == 0 or noexp_mask.sum() == 0:
        return float("nan")
    ev = coord[exp_mask]
    nv = coord[noexp_mask]
    auc = float(
        ((ev[:, None] > nv[None, :]).sum() + 0.5 * (ev[:, None] == nv[None, :]).sum())
        / (len(ev) * len(nv))
    )
    return max(auc, 1.0 - auc)


# ---------------------------------------------------------------------------
# fit wrapper
# ---------------------------------------------------------------------------

def fit_slice(X_raw: np.ndarray, pca_dim: int, n_atoms: int, atom_d: int,
              atom_topology: str, n_iter: int, seed: int) -> dict:
    import gamfit

    t0 = time.time()
    Z, Vt, mu = pca_project(X_raw, pca_dim)
    fit = gamfit.sae_manifold_fit(
        X=Z,
        K=n_atoms,
        d_atom=atom_d,
        atom_topology=atom_topology,
        n_iter=n_iter,
        random_state=seed,
        assignment="ibp_map",
        smoothness_weight=1.0,
        sparsity_weight=0.5,
    )
    fitted = np.asarray(fit.fitted)
    variance_total = float(((Z - Z.mean(0)) ** 2).sum())
    ev = float(1.0 - ((Z - fitted) ** 2).sum() / variance_total) if variance_total > 0 else float("nan")
    assignments = np.asarray(fit.assignments)   # (N, K) soft assignment logits
    hard = assignments.argmax(axis=1) if assignments.ndim == 2 else assignments
    return dict(
        fit=fit,
        Z=Z,
        fitted=fitted,
        hard=hard,
        ev=ev,
        seconds=time.time() - t0,
        n_atoms=int(assignments.shape[1]) if assignments.ndim == 2 else n_atoms,
    )


# ---------------------------------------------------------------------------
# color helpers
# ---------------------------------------------------------------------------

COLOR_RGB: dict[str, tuple[int, int, int]] = {}   # filled from prompts


def hex_to_mpl(h: str) -> str:
    return h if h.startswith("#") else "#" + h


def color_rgb_alignment(Z: np.ndarray, color_prompts: list[dict]) -> float:
    """
    Measure how well the PC1-PC2 embedding aligns with RGB space.
    Returns R² of a linear fit from (PC1, PC2) → (R, G, B) target.
    """
    try:
        rgb = np.array([p.get("rgb", [128, 128, 128]) for p in color_prompts],
                       dtype=np.float64) / 255.0
        Xf = np.column_stack([np.ones(len(Z)), Z[:, :2]])
        coef, _, _, _ = np.linalg.lstsq(Xf, rgb, rcond=None)
        pred = Xf @ coef
        ss_res = ((rgb - pred) ** 2).sum()
        ss_tot = ((rgb - rgb.mean(0)) ** 2).sum()
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

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
    "self_control": "#9a6324",
}

SIDE_PALETTE: dict[str, str] = {
    "exp": "#e6194b",
    "noexp": "#4363d8",
    "-": "#aaaaaa",
    "a": "#f58231",
    "b": "#42d4f4",
}


# ---------------------------------------------------------------------------
# draw the two-panel figure
# ---------------------------------------------------------------------------

def draw_qualia_panel(ax_side, ax_kind, res: dict, prompts: list[dict], ckpt: str) -> None:
    import matplotlib.patches as mpatches

    Z = res["Z"]
    sides = [p["side"] for p in prompts]
    kinds = [p["kind"] for p in prompts]

    # side panel
    side_colors = [SIDE_PALETTE.get(s, "#cccccc") for s in sides]
    ax_side.scatter(Z[:, 0], Z[:, 1], c=side_colors, s=12, alpha=0.65, linewidths=0)
    auc = exp_noexp_auc(Z[:, 0], sides)
    ax_side.set_title(
        f"{ckpt}  L25 exp/noexp\nEV={res['ev']:.3f}  AUC={auc:.3f}  {res['seconds']:.0f}s",
        fontsize=8,
    )
    handles = [mpatches.Patch(color=v, label=k)
               for k, v in SIDE_PALETTE.items() if k in set(sides)]
    ax_side.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
    ax_side.set_xticks([]); ax_side.set_yticks([])

    # kind panel
    kind_colors = [KIND_PALETTE.get(k, "#cccccc") for k in kinds]
    ax_kind.scatter(Z[:, 0], Z[:, 1], c=kind_colors, s=12, alpha=0.65, linewidths=0)
    ax_kind.set_title(f"{ckpt}  L25 by kind", fontsize=8)
    present = sorted(set(kinds))
    handles = [mpatches.Patch(color=KIND_PALETTE.get(k, "#cccccc"), label=k)
               for k in present[:14]]
    ax_kind.legend(handles=handles, fontsize=6, loc="upper right",
                   framealpha=0.7, ncol=2)
    ax_kind.set_xticks([]); ax_kind.set_yticks([])


def draw_color_tile(ax, res: dict, color_prompts: list[dict], ckpt: str) -> None:
    Z = res["Z"]
    hexcols = [hex_to_mpl(p.get("hex", "#cccccc")) for p in color_prompts]
    ax.scatter(Z[:, 0], Z[:, 1], c=hexcols, s=18, alpha=0.75, linewidths=0)
    rgb_r2 = color_rgb_alignment(Z, color_prompts)
    ax.set_title(
        f"{ckpt}  L44 color\nEV={res['ev']:.3f}  RGB-R²={rgb_r2:.3f}  {res['seconds']:.0f}s",
        fontsize=8,
    )
    # label each unique color at its centroid
    color_names = [p.get("color", "") for p in color_prompts]
    unique_colors = sorted(set(color_names))
    for c_name in unique_colors:
        mask = np.array([cn == c_name for cn in color_names])
        cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        ax.text(cx, cy, c_name, fontsize=5, ha="center", va="bottom", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5))
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pca_dim", type=int, default=10)
    parser.add_argument("--n_atoms", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--color_steps", nargs="*", default=None,
                        help="subset of checkpoint dirs to use for color trajectory")
    args = parser.parse_args()

    banks = Path(args.banks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 1. Qualia analysis on step10790 ---
    qualia_ckpt = banks / "5e-5-step10790"
    if not (qualia_ckpt / "activations.npy").exists():
        print(f"WARNING: qualia bank missing at {qualia_ckpt}, skipping qualia panel", flush=True)
        qualia_res = None
        qualia_prompts = []
    else:
        print(f"\n=== Qualia bank: {qualia_ckpt.name} ===", flush=True)
        l25_raw, qualia_prompts = load_qualia_l25(qualia_ckpt)
        print(f"  L25 shape {l25_raw.shape}  pca→{args.pca_dim}", flush=True)
        qualia_res = fit_slice(l25_raw, args.pca_dim, args.n_atoms, 2,
                               "euclidean", args.n_iter, args.seed)
        sides = [p["side"] for p in qualia_prompts]
        auc = exp_noexp_auc(qualia_res["Z"][:, 0], sides)
        print(f"  EV={qualia_res['ev']:.4f}  AUC={auc:.3f}  {qualia_res['seconds']:.1f}s", flush=True)

    # --- 2. Color trajectory across all SFT steps ---
    all_ckpts = sorted([p for p in banks.iterdir() if p.is_dir()], key=lambda p: p.name)
    if args.color_steps:
        color_ckpts = [banks / s for s in args.color_steps]
    else:
        # all that have extra/activations.npy
        color_ckpts = [c for c in all_ckpts if (c / "extra" / "activations.npy").exists()]

    print(f"\n=== Color trajectory: {len(color_ckpts)} checkpoints ===", flush=True)
    color_rows: list[dict] = []
    for ckpt_dir in color_ckpts:
        print(f"  {ckpt_dir.name} …", flush=True)
        l44_raw, col_prompts = load_color_l44(ckpt_dir)
        res = fit_slice(l44_raw, args.pca_dim, args.n_atoms, 2,
                        "euclidean", args.n_iter, args.seed)
        rgb_r2 = color_rgb_alignment(res["Z"], col_prompts)
        print(f"    EV={res['ev']:.4f}  RGB-R²={rgb_r2:.3f}  {res['seconds']:.1f}s", flush=True)
        color_rows.append(dict(ckpt=ckpt_dir.name, res=res, prompts=col_prompts, rgb_r2=rgb_r2))

    # --- Summary table ---
    print("\n=== SUMMARY ===", flush=True)
    if qualia_res is not None:
        sides = [p["side"] for p in qualia_prompts]
        auc = exp_noexp_auc(qualia_res["Z"][:, 0], sides)
        print(f"Qualia L25 ({qualia_ckpt.name}): EV={qualia_res['ev']:.4f}  "
              f"exp/noexp AUC={auc:.3f}  K={qualia_res['n_atoms']}", flush=True)
    print(f"\n{'checkpoint':<20} {'L44 EV':>8} {'RGB-R²':>8}")
    for row in color_rows:
        print(f"{row['ckpt']:<20} {row['res']['ev']:>8.4f} {row['rgb_r2']:>8.3f}")

    # --- Figure 1: qualia atlas (2 panels) ---
    if qualia_res is not None:
        fig1, (ax_s, ax_k) = plt.subplots(1, 2, figsize=(11, 5))
        draw_qualia_panel(ax_s, ax_k, qualia_res, qualia_prompts, qualia_ckpt.name)
        fig1.suptitle(
            "OLMo-3-32B SFT end (5e-5-step10790) — L25 qualia plane\n"
            f"PCA-{args.pca_dim} → sae_manifold_fit K={args.n_atoms}",
            fontsize=11,
        )
        fig1.tight_layout()
        p1 = out_dir / "olmo_qualia_atlas.png"
        fig1.savefig(str(p1), dpi=130)
        print(f"QUALIA ATLAS SAVED {p1}", flush=True)
        plt.close(fig1)

    # --- Figure 2: color trajectory grid ---
    if color_rows:
        n_cols = min(4, len(color_rows))
        n_rows = (len(color_rows) + n_cols - 1) // n_cols
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        ax_flat = np.asarray(axes).ravel()
        for i, row in enumerate(color_rows):
            draw_color_tile(ax_flat[i], row["res"], row["prompts"], row["ckpt"])
        for j in range(len(color_rows), len(ax_flat)):
            ax_flat[j].set_visible(False)
        fig2.suptitle(
            f"OLMo-3-32B SFT trajectory — L44 color geometry (PCA-{args.pca_dim} → "
            f"sae_manifold_fit K={args.n_atoms})\nRGB-R² = variance in RGB explained by PC1/PC2",
            fontsize=11,
        )
        fig2.tight_layout()
        p2 = out_dir / "olmo_color_trajectory.png"
        fig2.savefig(str(p2), dpi=130)
        print(f"COLOR TRAJECTORY SAVED {p2}", flush=True)
        plt.close(fig2)

        # --- Figure 3: RGB-R² over SFT steps ---
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        steps = [int(r["ckpt"].split("step")[-1]) for r in color_rows]
        r2s = [r["rgb_r2"] for r in color_rows]
        evs = [r["res"]["ev"] for r in color_rows]
        ax3.plot(steps, r2s, "o-", label="RGB-R² (color alignment)", color="#e6194b")
        ax3.plot(steps, evs, "s--", label="L44 EV (reconstruction)", color="#4363d8")
        ax3.set_xlabel("SFT step")
        ax3.set_ylabel("metric")
        ax3.set_title("Color geometry vs SFT step — OLMo-3-32B")
        ax3.legend()
        ax3.set_ylim(0, 1)
        fig3.tight_layout()
        p3 = out_dir / "olmo_color_r2_curve.png"
        fig3.savefig(str(p3), dpi=130)
        print(f"R² CURVE SAVED {p3}", flush=True)
        plt.close(fig3)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
