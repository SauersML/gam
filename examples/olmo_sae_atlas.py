"""Fit gam's manifold-SAE on OLMo-3-32B residual-stream activations.

Loads L25 (qualia) and L44 (color) activation slices across SFT checkpoints,
PCA-whitens to a manageable dimension, fits sae_manifold_fit, and renders a
2×3 atlas: early vs late SFT, L25 colored by kind/side, L44 colored by color.

Reports reconstruction EV, atom count, and exp vs noexp separation at each
checkpoint.

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

def load_activations_l25(bank_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (N, 5120) float32 L25 slice + prompts list."""
    acts = np.load(bank_dir / "activations.npy")   # (635, 64, 5120)
    l25 = acts[:, 25, :].astype(np.float32)
    with open(bank_dir / "prompts.jsonl") as f:
        prompts = [json.loads(line) for line in f]
    return l25, prompts


def load_activations_l44_color(bank_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (180, 5120) float32 L44 color slice + color prompts."""
    extra = bank_dir / "extra"
    acts = np.load(extra / "activations.npy")       # (180, 64, 5120)
    l44 = acts[:, 44, :].astype(np.float32)
    with open(extra / "prompts.jsonl") as f:
        prompts = [json.loads(line) for line in f]
    return l44, prompts


def pca_whiten(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-whiten X to n_components, return (Z, components, mean)."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # use randomised SVD via numpy for speed at D=5120
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # keep top n_components
    Vt = Vt[:n_components]
    S = S[:n_components]
    Z = (Xc @ Vt.T) / S[np.newaxis, :]   # (N, n_components) zero-mean unit-var
    return Z.astype(np.float64), Vt, mu


def exp_noexp_auc(coords_1d: np.ndarray, sides: list[str]) -> float:
    """Wilcoxon AUC: how well the 1-D coordinate separates exp from noexp rows."""
    exp_mask = np.array([s == "exp" for s in sides])
    noexp_mask = np.array([s == "noexp" for s in sides])
    if exp_mask.sum() == 0 or noexp_mask.sum() == 0:
        return float("nan")
    exp_vals = coords_1d[exp_mask]
    noexp_vals = coords_1d[noexp_mask]
    # AUC = P(exp > noexp)
    auc = ((exp_vals[:, None] > noexp_vals[None, :]).sum()
           + 0.5 * (exp_vals[:, None] == noexp_vals[None, :]).sum())
    auc /= len(exp_vals) * len(noexp_vals)
    return float(max(auc, 1 - auc))


# ---------------------------------------------------------------------------
# fit one checkpoint slice
# ---------------------------------------------------------------------------

def fit_one(X_raw: np.ndarray, pca_dim: int, n_atoms: int, atom_d: int,
            atom_topology: str, n_iter: int, seed: int) -> dict:
    import gamfit

    t0 = time.time()
    Z, _, _ = pca_whiten(X_raw, pca_dim)
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
    ev = float(1.0 - ((Z - fitted) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum())
    assignments = np.asarray(fit.assignments)           # (N, K) logit-ish
    hard = assignments.argmax(axis=1) if assignments.ndim == 2 else assignments
    return dict(
        fit=fit,
        Z=Z,
        fitted=fitted,
        hard=hard,
        ev=ev,
        seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

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
}

SIDE_PALETTE = {"exp": "#e6194b", "noexp": "#4363d8", "-": "#aaaaaa",
                "a": "#f58231", "b": "#42d4f4"}


def _scatter_ax(ax, Z_2d, colors, labels=None, title="", ev=None, secs=None):
    ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=colors, s=12, alpha=0.65, linewidths=0)
    ev_str = f"  EV={ev:.3f}" if ev is not None else ""
    time_str = f"  {secs:.0f}s" if secs is not None else ""
    ax.set_title(f"{title}{ev_str}{time_str}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_atlas(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_rows = len(rows)   # one row per checkpoint
    n_cols = 3           # [L25 by side] [L25 by kind] [L44 by color]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, row in enumerate(rows):
        ckpt = row["checkpoint"]

        # --- L25 side panel ---
        ax = axes[ri, 0]
        res = row["l25"]
        Z = res["Z"]
        sides = row["sides"]
        hard = res["hard"]
        side_colors = [SIDE_PALETTE.get(s, "#cccccc") for s in sides]
        _scatter_ax(ax, Z[:, :2], side_colors,
                    title=f"{ckpt}  L25 by exp/noexp",
                    ev=res["ev"], secs=res["seconds"])
        # legend
        handles = [mpatches.Patch(color=v, label=k)
                   for k, v in SIDE_PALETTE.items() if k in set(sides)]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

        # AUC annotation
        coord0 = Z[:, 0]
        auc = exp_noexp_auc(coord0, sides)
        ax.text(0.02, 0.02, f"PC1 AUC={auc:.3f}", transform=ax.transAxes,
                fontsize=7, color="black", va="bottom")

        # --- L25 kind panel ---
        ax = axes[ri, 1]
        kinds = row["kinds"]
        kind_colors = [KIND_PALETTE.get(k, "#cccccc") for k in kinds]
        _scatter_ax(ax, Z[:, :2], kind_colors,
                    title=f"{ckpt}  L25 by kind")
        present_kinds = sorted(set(kinds))
        handles = [mpatches.Patch(color=KIND_PALETTE.get(k, "#cccccc"), label=k)
                   for k in present_kinds[:12]]
        ax.legend(handles=handles, fontsize=6, loc="upper right",
                  framealpha=0.7, ncol=2)

        # --- L44 color panel ---
        ax = axes[ri, 2]
        col_res = row.get("l44_color")
        if col_res is not None:
            Zc = col_res["Z"]
            # color prompts have hex field
            color_prompts = row["color_prompts"]
            hexcols = [p.get("hex", "#cccccc") for p in color_prompts]
            _scatter_ax(ax, Zc[:, :2], hexcols,
                        title=f"{ckpt}  L44 by color",
                        ev=col_res["ev"], secs=col_res["seconds"])
            # label by color name (unique)
            seen = set()
            for xi, p in enumerate(color_prompts):
                c_name = p.get("color", "")
                if c_name not in seen:
                    seen.add(c_name)
                    ax.text(Zc[xi, 0], Zc[xi, 1], c_name,
                            fontsize=5, ha="center", va="bottom", alpha=0.8)
        else:
            ax.set_visible(False)

    fig.suptitle(
        "OLMo-3-32B SFT trajectory — gam manifold-SAE atlas\n"
        "L25 = self/qualia layer, L44 = color layer; PCA-10 → sae_manifold_fit",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130)
    print(f"ATLAS SAVED {out_path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks_dir", required=True,
                        help="directory containing checkpoint subdirs")
    parser.add_argument("--out_dir", required=True,
                        help="directory to write PNGs")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="explicit list of checkpoint names; default = first + last")
    parser.add_argument("--pca_dim", type=int, default=10,
                        help="PCA components before fitting (default 10)")
    parser.add_argument("--n_atoms", type=int, default=4,
                        help="number of SAE atoms K (default 4)")
    parser.add_argument("--n_iter", type=int, default=25,
                        help="REML iterations (default 25)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    banks = Path(args.banks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick checkpoints to compare
    all_ckpts = sorted(banks.iterdir(), key=lambda p: p.name)
    all_ckpts = [p for p in all_ckpts if p.is_dir()]
    if args.checkpoints:
        selected = [banks / c for c in args.checkpoints]
    else:
        # default: first, middle, last SFT step
        n = len(all_ckpts)
        idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        selected = [all_ckpts[i] for i in idxs]

    print(f"Fitting {len(selected)} checkpoints: {[p.name for p in selected]}", flush=True)

    atlas_rows = []
    for ckpt_dir in selected:
        ckpt_name = ckpt_dir.name
        print(f"\n=== {ckpt_name} ===", flush=True)

        # L25 qualia
        l25_raw, prompts = load_activations_l25(ckpt_dir)
        sides = [p["side"] for p in prompts]
        kinds = [p["kind"] for p in prompts]
        print(f"  L25 raw shape {l25_raw.shape}", flush=True)
        print(f"  fitting L25 K={args.n_atoms} d=2 euclidean pca={args.pca_dim}…", flush=True)
        l25_res = fit_one(l25_raw, args.pca_dim, args.n_atoms, 2,
                          "euclidean", args.n_iter, args.seed)
        print(f"  L25 EV={l25_res['ev']:.4f}  {l25_res['seconds']:.1f}s", flush=True)
        auc = exp_noexp_auc(l25_res["Z"][:, 0], sides)
        print(f"  L25 PC1 exp/noexp AUC={auc:.3f}", flush=True)

        # L44 color (in extra/)
        l44_res = None
        color_prompts = []
        extra = ckpt_dir / "extra"
        if extra.exists() and (extra / "activations.npy").exists():
            l44_raw, color_prompts = load_activations_l44_color(ckpt_dir)
            print(f"  L44 color raw shape {l44_raw.shape}", flush=True)
            print(f"  fitting L44 K={args.n_atoms} d=2 euclidean pca={args.pca_dim}…", flush=True)
            l44_res = fit_one(l44_raw, args.pca_dim, args.n_atoms, 2,
                              "euclidean", args.n_iter, args.seed)
            print(f"  L44 EV={l44_res['ev']:.4f}  {l44_res['seconds']:.1f}s", flush=True)

        atlas_rows.append(dict(
            checkpoint=ckpt_name,
            l25=l25_res,
            sides=sides,
            kinds=kinds,
            l44_color=l44_res,
            color_prompts=color_prompts,
        ))

    # summary table
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'checkpoint':<20} {'L25 EV':>8} {'L25 AUC':>9} {'L44 EV':>8}")
    for row in atlas_rows:
        auc = exp_noexp_auc(row["l25"]["Z"][:, 0], row["sides"])
        l44_ev = row["l44_color"]["ev"] if row["l44_color"] else float("nan")
        print(f"{row['checkpoint']:<20} {row['l25']['ev']:>8.4f} {auc:>9.3f} {l44_ev:>8.4f}")

    # render atlas
    atlas_path = out_dir / "olmo_sae_atlas.png"
    draw_atlas(atlas_rows, atlas_path)

    # also per-checkpoint quick saves
    for row in atlas_rows:
        ckpt = row["checkpoint"]
        np.save(str(out_dir / f"l25_Z_{ckpt}.npy"), row["l25"]["Z"])
        if row["l44_color"]:
            np.save(str(out_dir / f"l44_Z_{ckpt}.npy"), row["l44_color"]["Z"])

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
