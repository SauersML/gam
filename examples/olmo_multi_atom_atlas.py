"""Multi-atom manifold-SAE atlas for OLMo-3-32B L25 residual-stream activations.

Goals
-----
1. K=1 sweep across all 5 checkpoints to build the full dev-trajectory curve.
2. K=2, 3, 5 on the qualia (L25, N=635) slice — same 5 checkpoints.
3. Outputs:
   - plots/olmo_multi_atom_grid.png  — K × checkpoint heatmap + scatter grids
   - plots/olmo_trajectory_curve.png — EV + exp/noexp AUC vs training step
   - plots/olmo_k_sweep_ev.png       — EV vs K per checkpoint
   - olmo_multi_atom_results.csv     — one row per (K, checkpoint)

Bug workarounds
---------------
- #1094: euclidean K>1 hangs → use atom_topology="circle" for all fits
- #1095: circle K=1 fails on N=180 color bank → skip L44 color for now

Usage (MSI):
    source /projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/activate
    cd /tmp && python3 /projects/standard/hsiehph/sauer354/gam/examples/olmo_multi_atom_atlas.py \\
        --data_dir /projects/standard/hsiehph/sauer354/olmo_data \\
        --out_dir  /projects/standard/hsiehph/sauer354/olmo_data/plots \\
        --n_iter 40
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint registry — ordered by training step so the trajectory is
# chronological on the x-axis.
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    ("stage1-step0",      "SFT-init",  0),
    ("base",              "base",      None),   # pretrain baseline
    ("stage3-step11921",  "SFT-end",   11921),
    ("instruct",          "instruct",  None),   # post-SFT
    ("step_2300",         "RLVR",      None),   # RLVR final
]

K_SWEEP = [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_l25(ckpt_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Return (635, 5120) float32 L25 slice and prompt metadata."""
    acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")
    l25 = np.array(acts[:, 25, :], dtype=np.float32)
    with open(ckpt_dir / "prompts.jsonl") as fh:
        prompts = [json.loads(line) for line in fh]
    return l25, prompts


def pca_project(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Z_whitened, Vt, mu) with shape (N, n_components)."""
    mu = X.mean(axis=0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:n_components]
    S_trunc = S[:n_components]
    Z = (Xc @ Vt.T) / (S_trunc[np.newaxis, :] + 1e-12)
    return Z, Vt, mu


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exp_noexp_auc(coord: np.ndarray, sides: list[str]) -> float:
    """Wilcoxon AUC for exp vs noexp separation on first coordinate."""
    exp_mask = np.array([s == "exp" for s in sides])
    noexp_mask = np.array([s == "noexp" for s in sides])
    if exp_mask.sum() == 0 or noexp_mask.sum() == 0:
        return float("nan")
    ev = coord[exp_mask]
    nv = coord[noexp_mask]
    n = len(ev) * len(nv)
    auc = float(
        ((ev[:, None] > nv[None, :]).sum() + 0.5 * (ev[:, None] == nv[None, :]).sum()) / n
    )
    return max(auc, 1.0 - auc)


def kind_purity(hard: np.ndarray, kinds: list[str]) -> float:
    """Mean per-atom majority-class purity across active atoms."""
    unique_atoms = np.unique(hard)
    purities = []
    for a in unique_atoms:
        mask = hard == a
        if mask.sum() == 0:
            continue
        ks = [kinds[i] for i in range(len(kinds)) if mask[i]]
        most_common = max(set(ks), key=ks.count)
        purities.append(ks.count(most_common) / len(ks))
    return float(np.mean(purities)) if purities else float("nan")


# ---------------------------------------------------------------------------
# SAE fit wrapper
# ---------------------------------------------------------------------------

def fit_slice(
    Z: np.ndarray,
    n_atoms: int,
    n_iter: int,
    seed: int,
    pca_dim: int,
) -> dict | None:
    """
    Run sae_manifold_fit on pre-whitened Z (N, pca_dim).
    Returns dict with fitted, hard, ev, seconds, k_active — or None on failure.
    """
    import gamfit

    t0 = time.time()
    try:
        fit = gamfit.sae_manifold_fit(
            X=Z,
            K=n_atoms,
            d_atom=2,
            # Use circle for all K values: euclidean K>1 hangs (#1094)
            atom_topology="circle",
            n_iter=n_iter,
            random_state=seed,
            assignment="ibp_map",
            smoothness_weight=1.0,
            sparsity_weight=0.5,
        )
    except Exception as exc:
        print(f"    FAILED (K={n_atoms}): {type(exc).__name__}: {exc}", flush=True)
        return None

    fitted = np.asarray(fit.fitted)
    total_var = float(((Z - Z.mean(0)) ** 2).sum())
    ev = float(1.0 - ((Z - fitted) ** 2).sum() / total_var) if total_var > 1e-12 else float("nan")
    asn = np.asarray(fit.assignments)
    hard = asn.argmax(axis=1) if asn.ndim == 2 else asn
    k_active = int(len(np.unique(hard)))
    return dict(
        Z=Z,
        fitted=fitted,
        hard=hard,
        assignments=asn,
        ev=ev,
        seconds=time.time() - t0,
        k_requested=n_atoms,
        k_active=k_active,
    )


# ---------------------------------------------------------------------------
# Colour palettes
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

# Atom colours — up to 8 atoms
ATOM_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#ffe119", "#808080",
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _scatter_side(ax, Z, prompts, title):
    import matplotlib.patches as mpatches
    sides = [p.get("side", "-") for p in prompts]
    colors = [SIDE_PALETTE.get(s, "#cccccc") for s in sides]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=8, alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    present = sorted(set(sides))
    handles = [mpatches.Patch(color=SIDE_PALETTE.get(s, "#cccccc"), label=s) for s in present]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.6)


def _scatter_atom(ax, Z, hard, title):
    import matplotlib.patches as mpatches
    colors = [ATOM_PALETTE[int(h) % len(ATOM_PALETTE)] for h in hard]
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=8, alpha=0.65, linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    unique = sorted(set(int(h) for h in hard))
    handles = [mpatches.Patch(color=ATOM_PALETTE[a % len(ATOM_PALETTE)], label=f"atom {a}")
               for a in unique]
    ax.legend(handles=handles, fontsize=5, loc="upper right", framealpha=0.6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/projects/standard/hsiehph/sauer354/olmo_data")
    parser.add_argument("--out_dir",  default="/projects/standard/hsiehph/sauer354/olmo_data/plots")
    parser.add_argument("--pca_dim",  type=int, default=32)
    parser.add_argument("--n_iter",   type=int, default=40)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    data = Path(args.data_dir)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # Accumulate results: one entry per (ckpt_key, K)
    # results[ckpt_key][K] = fit_dict | None
    # -----------------------------------------------------------------------
    results: dict[str, dict[int, dict | None]] = {}
    prompts_by_ckpt: dict[str, list[dict]] = {}

    for ckpt_key, ckpt_label, _step in CHECKPOINTS:
        ckpt_dir = data / ckpt_key
        if not ckpt_dir.exists():
            print(f"\nSKIP {ckpt_key}: directory not found", flush=True)
            continue

        print(f"\n=== {ckpt_label} ({ckpt_key}) ===", flush=True)
        X_raw, prompts = load_l25(ckpt_dir)
        print(f"  L25 shape: {X_raw.shape}  PCA-{args.pca_dim}", flush=True)
        prompts_by_ckpt[ckpt_key] = prompts

        Z, _, _ = pca_project(X_raw, args.pca_dim)
        results[ckpt_key] = {}

        for K in K_SWEEP:
            print(f"  K={K} ...", end="", flush=True)
            r = fit_slice(Z, K, args.n_iter, args.seed, args.pca_dim)
            results[ckpt_key][K] = r
            if r is not None:
                sides = [p.get("side", "-") for p in prompts]
                auc = exp_noexp_auc(r["Z"][:, 0], sides)
                kinds = [p.get("kind", "?") for p in prompts]
                purity = kind_purity(r["hard"], kinds)
                r["auc"] = auc
                r["kind_purity"] = purity
                print(f" EV={r['ev']:.4f}  AUC={auc:.3f}  purity={purity:.3f}"
                      f"  k_active={r['k_active']}  {r['seconds']:.1f}s", flush=True)
            else:
                print(" FAILED", flush=True)

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    csv_path = data / "olmo_multi_atom_results.csv"
    csv_fields = ["checkpoint", "label", "K_requested", "K_active",
                  "EV", "AUC_exp_noexp", "kind_purity", "seconds"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for ckpt_key, ckpt_label, _step in CHECKPOINTS:
            if ckpt_key not in results:
                continue
            for K in K_SWEEP:
                r = results[ckpt_key].get(K)
                writer.writerow({
                    "checkpoint": ckpt_key,
                    "label": ckpt_label,
                    "K_requested": K,
                    "K_active": r["k_active"] if r else "",
                    "EV": f"{r['ev']:.4f}" if r else "",
                    "AUC_exp_noexp": f"{r.get('auc', float('nan')):.3f}" if r else "",
                    "kind_purity": f"{r.get('kind_purity', float('nan')):.3f}" if r else "",
                    "seconds": f"{r['seconds']:.1f}" if r else "",
                })
    print(f"\nCSV saved: {csv_path}", flush=True)

    # -----------------------------------------------------------------------
    # Plot 1: K-sweep EV grid (K on x-axis, one line per checkpoint)
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_ev, ax_auc, ax_purity = axes1

    # colours per checkpoint
    ckpt_colors = {ckpt_key: plt.cm.tab10(i / max(len(CHECKPOINTS) - 1, 1))
                   for i, (ckpt_key, _, _) in enumerate(CHECKPOINTS)}

    for ckpt_key, ckpt_label, _step in CHECKPOINTS:
        if ckpt_key not in results:
            continue
        xs, evs, aucs, purities = [], [], [], []
        for K in K_SWEEP:
            r = results[ckpt_key].get(K)
            if r is None:
                continue
            xs.append(K)
            evs.append(r["ev"])
            aucs.append(r.get("auc", float("nan")))
            purities.append(r.get("kind_purity", float("nan")))
        c = ckpt_colors[ckpt_key]
        if xs:
            ax_ev.plot(xs, evs, "o-", color=c, label=ckpt_label, lw=2)
            ax_auc.plot(xs, aucs, "s-", color=c, label=ckpt_label, lw=2)
            ax_purity.plot(xs, purities, "^-", color=c, label=ckpt_label, lw=2)

    for ax, ylabel, title in [
        (ax_ev, "EV (explained variance)", "Reconstruction EV vs K-atoms"),
        (ax_auc, "AUC (exp vs noexp)", "exp/noexp separability vs K-atoms"),
        (ax_purity, "Atom kind-purity", "Concept purity per atom vs K"),
    ]:
        ax.set_xlabel("K (number of atoms)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(K_SWEEP)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig1.suptitle(
        "OLMo-3-32B L25 qualia — multi-atom manifold-SAE K sweep across all checkpoints\n"
        f"(atom_topology=circle, PCA-{args.pca_dim})",
        fontsize=10,
    )
    fig1.tight_layout()
    p1 = out / "olmo_k_sweep_metrics.png"
    fig1.savefig(str(p1), dpi=130)
    plt.close(fig1)
    print(f"K-sweep metrics plot saved: {p1}", flush=True)

    # -----------------------------------------------------------------------
    # Plot 2: Trajectory curve (K=1) — EV + AUC across training steps
    # -----------------------------------------------------------------------
    # Order checkpoints for the x-axis
    traj_order = ["stage1-step0", "base", "stage3-step11921", "instruct", "step_2300"]
    traj_labels = {k: lbl for k, lbl, _ in CHECKPOINTS}

    traj_xs, traj_evs, traj_aucs = [], [], []
    for i, ck in enumerate(traj_order):
        if ck not in results:
            continue
        r = results[ck].get(1)
        if r is None:
            continue
        traj_xs.append(i)
        traj_evs.append(r["ev"])
        traj_aucs.append(r.get("auc", float("nan")))

    if traj_xs:
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        ax2_r = ax2.twinx()
        xlabels = [traj_labels.get(traj_order[x], traj_order[x]) for x in traj_xs]
        ax2.plot(traj_xs, traj_evs, "o-", color="#e6194b", lw=2.5, label="L25 EV (K=1)")
        ax2_r.plot(traj_xs, traj_aucs, "s--", color="#4363d8", lw=2.5, label="exp/noexp AUC (K=1)")
        ax2.set_xticks(traj_xs)
        ax2.set_xticklabels(xlabels, fontsize=9)
        ax2.set_ylabel("Explained Variance", color="#e6194b")
        ax2_r.set_ylabel("exp/noexp AUC", color="#4363d8")
        ax2.set_title(
            "OLMo-3-32B L25 qualia: representational geometry across training\n"
            f"(circle K=1, PCA-{args.pca_dim})",
            fontsize=10,
        )
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        p2 = out / "olmo_trajectory_curve.png"
        fig2.savefig(str(p2), dpi=130)
        plt.close(fig2)
        print(f"Trajectory curve saved: {p2}", flush=True)

    # -----------------------------------------------------------------------
    # Plot 3: Scatter grid — rows=checkpoints, cols=K values
    # (only when we have results; skip missing)
    # -----------------------------------------------------------------------
    active_ckpts = [ck for ck in traj_order if ck in results]
    active_Ks = [K for K in K_SWEEP if any(
        results.get(ck, {}).get(K) is not None for ck in active_ckpts
    )]

    if active_ckpts and active_Ks:
        n_rows = len(active_ckpts)
        # Two panels per K: atom-coloured + side-coloured
        n_cols = len(active_Ks) * 2
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.6))
        if n_rows == 1:
            axes3 = axes3[np.newaxis, :]
        if n_cols == 1:
            axes3 = axes3[:, np.newaxis]

        for r_idx, ck in enumerate(active_ckpts):
            prompts = prompts_by_ckpt.get(ck, [])
            label = traj_labels.get(ck, ck)
            for c_idx, K in enumerate(active_Ks):
                r = results[ck].get(K)
                col_atom = c_idx * 2
                col_side = c_idx * 2 + 1
                ax_a = axes3[r_idx, col_atom]
                ax_s = axes3[r_idx, col_side]
                if r is None:
                    ax_a.text(0.5, 0.5, "FAILED", ha="center", va="center",
                              transform=ax_a.transAxes, fontsize=9, color="red")
                    ax_s.text(0.5, 0.5, "FAILED", ha="center", va="center",
                              transform=ax_s.transAxes, fontsize=9, color="red")
                    ax_a.set_xticks([]); ax_a.set_yticks([])
                    ax_s.set_xticks([]); ax_s.set_yticks([])
                    continue
                Z = r["Z"]
                _scatter_atom(ax_a, Z, r["hard"],
                              f"{label} K={K}\nEV={r['ev']:.3f} purity={r.get('kind_purity', float('nan')):.2f}")
                _scatter_side(ax_s, Z, prompts,
                              f"{label} K={K}\nAUC={r.get('auc', float('nan')):.3f}")

        fig3.suptitle(
            "OLMo-3-32B L25 qualia — atom assignments (left) and exp/noexp sides (right)\n"
            f"Across checkpoints (rows) and K values (cols). atom_topology=circle, PCA-{args.pca_dim}",
            fontsize=10,
        )
        fig3.tight_layout()
        p3 = out / "olmo_multi_atom_grid.png"
        fig3.savefig(str(p3), dpi=110)
        plt.close(fig3)
        print(f"Multi-atom scatter grid saved: {p3}", flush=True)

    # -----------------------------------------------------------------------
    # Plot 4: K=1 across all checkpoints — kind-coloured scatter (5 panels)
    # -----------------------------------------------------------------------
    import matplotlib.patches as mpatches

    active_k1 = [(ck, traj_labels.get(ck, ck)) for ck in traj_order
                 if ck in results and results[ck].get(1) is not None]
    if active_k1:
        n = len(active_k1)
        fig4, axes4 = plt.subplots(1, n, figsize=(n * 3.5, 4.0))
        if n == 1:
            axes4 = [axes4]
        for i, (ck, label) in enumerate(active_k1):
            ax = axes4[i]
            r = results[ck][1]
            prompts = prompts_by_ckpt[ck]
            kinds = [p.get("kind", "?") for p in prompts]
            colors = [KIND_PALETTE.get(k, "#cccccc") for k in kinds]
            Z = r["Z"]
            ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=9, alpha=0.65, linewidths=0)
            ax.set_title(
                f"{label}\nEV={r['ev']:.3f} AUC={r.get('auc', float('nan')):.3f}",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])
            if i == n - 1:
                present = sorted(set(kinds))
                handles = [mpatches.Patch(color=KIND_PALETTE.get(k, "#cccccc"), label=k)
                           for k in present[:14]]
                ax.legend(handles=handles, fontsize=5, loc="lower right", framealpha=0.6, ncol=2)
        fig4.suptitle(
            "OLMo-3-32B L25 qualia — K=1 kind-coloured scatter across all training checkpoints",
            fontsize=10,
        )
        fig4.tight_layout()
        p4 = out / "olmo_k1_all_checkpoints.png"
        fig4.savefig(str(p4), dpi=130)
        plt.close(fig4)
        print(f"K=1 all-checkpoints plot saved: {p4}", flush=True)

    # -----------------------------------------------------------------------
    # Terminal summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'checkpoint':<22} {'K_req':>6} {'K_act':>6} {'EV':>8} {'AUC':>8} {'purity':>8} {'secs':>6}"
    print(header)
    print("-" * 78)
    for ck in traj_order:
        if ck not in results:
            continue
        label = traj_labels.get(ck, ck)
        for K in K_SWEEP:
            r = results[ck].get(K)
            if r is None:
                print(f"{label:<22} {K:>6} {'—':>6} {'FAILED':>8}")
                continue
            print(f"{label:<22} {K:>6} {r['k_active']:>6} {r['ev']:>8.4f}"
                  f" {r.get('auc', float('nan')):>8.3f}"
                  f" {r.get('kind_purity', float('nan')):>8.3f}"
                  f" {r['seconds']:>6.0f}s")
    print("=" * 78)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
