"""Functorial layer transport: does the qualia manifold geometry survive L25 → L44?

SCIENCE QUESTION: The "self/qualia" geometry identified at L25 — does it
transport isometrically to L44 (the "color" layer)? A small isometry defect
means the manifold structure is preserved (the geometry *is* the concept);
a large defect means the representation reorganizes fundamentally between layers.

METHOD: We have both L25 and L44 activations for stage1-step0 and stage3-step11921.
1. Extract 1D circular chart from each layer via PCA + fit_circle.
2. Use gamfit.layer_transport_fit() to estimate winding degree + isometry defect
   with profile-likelihood CIs.
3. Compare across checkpoints: does training make the geometry more isometric?

Also run layer_transport_ladder across all available layers in the activations
(shapes: 635 x 64 x 5120 — 64 transformer layers) to find where geometry
concentrates and where it breaks.

Outputs:
  /projects/standard/hsiehph/sauer354/olmo_data/plots/layer_transport.png
  /projects/standard/hsiehph/sauer354/olmo_data/plots/layer_transport.json
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

DATA_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data")
OUT_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data/plots")
# Checkpoints that have both L25 and L44 data
TRANSPORT_CHECKPOINTS = ["stage1-step0", "stage3-step11921"]
# Layer pairs to test for the ladder (subsample for speed)
LADDER_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 44, 50, 55, 63]
PCA_DIM_FOR_CIRCLE = 2


def pca_whiten(X: np.ndarray, n: int) -> np.ndarray:
    mu = X.mean(0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n].T) / (S[:n] + 1e-12)


def to_circular_coord(Z2d: np.ndarray) -> np.ndarray:
    """Convert 2D PCA embedding to circular angle coordinate."""
    return np.arctan2(Z2d[:, 1], Z2d[:, 0])


def fit_l25_to_l44_transport(ckpt: str) -> dict:
    """Test transport of qualia geometry from L25 to L44."""
    import gamfit

    ckpt_dir = DATA_DIR / ckpt
    acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")
    X_l25 = np.array(acts[:, 25, :], dtype=np.float32)
    X_l44 = np.array(acts[:, 44, :], dtype=np.float32)

    # Get circular chart for each layer
    Z25 = pca_whiten(X_l25, PCA_DIM_FOR_CIRCLE)
    Z44 = pca_whiten(X_l44, PCA_DIM_FOR_CIRCLE)
    theta25 = to_circular_coord(Z25)
    theta44 = to_circular_coord(Z44)

    print(f"  [{ckpt}] L25→L44 transport fit ...", flush=True)
    t0 = time.time()
    try:
        result = gamfit.layer_transport_fit(
            theta25, theta44,
            topology_from="circle", topology_to="circle",
            layer_from=25, layer_to=44,
        )
        elapsed = time.time() - t0
        print(f"  [{ckpt}] done {elapsed:.1f}s  "
              f"degree={result.get('degree')}  "
              f"isometry_defect={result.get('isometry_defect', float('nan')):.4f} "
              f"±{result.get('isometry_defect_se', float('nan')):.4f}", flush=True)
        return dict(ckpt=ckpt, layer_from=25, layer_to=44, elapsed=elapsed,
                    error=None, **{k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in result.items()})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{ckpt}] FAILED {elapsed:.1f}s: {e}", flush=True)
        traceback.print_exc()
        return dict(ckpt=ckpt, layer_from=25, layer_to=44, elapsed=elapsed,
                    error=str(e)[:300], isometry_defect=float("nan"))


def fit_layer_ladder(ckpt: str) -> list[dict]:
    """Fit transport ladder across sampled layers."""
    import gamfit

    ckpt_dir = DATA_DIR / ckpt
    acts = np.load(ckpt_dir / "activations.npy", mmap_mode="r")
    n_layers = acts.shape[1]

    # Filter to available layers
    layers = [l for l in LADDER_LAYERS if l < n_layers]
    print(f"  [{ckpt}] ladder over layers {layers}", flush=True)

    coords = []
    for l in layers:
        X = np.array(acts[:, l, :], dtype=np.float32)
        Z = pca_whiten(X, PCA_DIM_FOR_CIRCLE)
        theta = to_circular_coord(Z)
        coords.append(theta)

    t0 = time.time()
    try:
        result = gamfit.layer_transport_ladder(coords, topology="circle", layers=layers)
        elapsed = time.time() - t0
        print(f"  [{ckpt}] ladder done {elapsed:.1f}s", flush=True)

        # Extract adjacent pairs
        adjacent = result.get("adjacent", [])
        rows = []
        for r in adjacent:
            row = dict(ckpt=ckpt, error=None)
            for k, v in r.items():
                row[k] = float(v) if isinstance(v, (int, float)) else v
            rows.append(row)
        return rows
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{ckpt}] ladder FAILED {elapsed:.1f}s: {e}", flush=True)
        traceback.print_exc()
        return [dict(ckpt=ckpt, error=str(e)[:300])]


def plot_results(transport_results: list[dict], ladder_results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: L25→L44 isometry defect by checkpoint
    ax = axes[0]
    valid = [r for r in transport_results if not np.isnan(r.get("isometry_defect", float("nan")))]
    if valid:
        ckpts = [r["ckpt"] for r in valid]
        defects = [r.get("isometry_defect", float("nan")) for r in valid]
        defect_ses = [r.get("isometry_defect_se", 0.0) for r in valid]
        degrees = [r.get("degree", 0) for r in valid]
        colors = ["#E91E63" if abs(d - 1.0) < 0.1 else "#2196F3"
                  for d in degrees]  # red if winding=1 (preserved)

        ax.bar(range(len(ckpts)), defects, color=colors,
               yerr=defect_ses, capsize=5, alpha=0.8)
        ax.set_xticks(range(len(ckpts)))
        ax.set_xticklabels([c.replace("stage", "s").replace("-step", "\nstep") for c in ckpts],
                           fontsize=8)
        for i, (d, deg) in enumerate(zip(defects, degrees)):
            ax.text(i, d + 0.01, f"deg={deg}", ha="center", fontsize=8)

    ax.set_ylabel("Isometry defect ∫(|h'|−1)² dP̂")
    ax.set_title("L25→L44 isometry defect\n(lower = geometry preserved)", fontweight="bold")
    ax.set_xlabel("Checkpoint")

    # Panel 2 & 3: Ladder isometry defect vs layer pair (per checkpoint)
    for ax, ckpt in zip(axes[1:], TRANSPORT_CHECKPOINTS):
        rows = ladder_results.get(ckpt, [])
        valid_rows = [r for r in rows if not r.get("error") and
                      not np.isnan(r.get("isometry_defect", float("nan")))]
        if valid_rows:
            layer_pairs = [f"{r.get('layer_from','')}→{r.get('layer_to','')}"
                           for r in valid_rows]
            defects = [r.get("isometry_defect", float("nan")) for r in valid_rows]
            ses = [r.get("isometry_defect_se", 0.0) for r in valid_rows]
            xs = range(len(layer_pairs))
            ax.plot(xs, defects, "o-", color="#2196F3", linewidth=1.5, markersize=5)
            ax.fill_between(xs,
                            [d - s for d, s in zip(defects, ses)],
                            [d + s for d, s in zip(defects, ses)],
                            alpha=0.2, color="#2196F3")
            ax.axvline([i for i, lp in enumerate(layer_pairs) if "25→" in lp][0]
                       if any("25→" in lp for lp in layer_pairs) else -1,
                       color="#E91E63", linestyle="--", linewidth=1, label="L25")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(layer_pairs, rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Layer pair")
            ax.set_ylabel("Isometry defect")
        ax.set_title(f"Transport ladder: {ckpt}\n(where does qualia geometry concentrate?)",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(
        "OLMo-3-32B — Functorial layer transport & isometry defect\n"
        "Circular chart (PCA-2 → angle), REML smoothing, delta-method SE",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"\nSAVED {out_path}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    transport_results = []
    ladder_results = {}

    for ckpt in TRANSPORT_CHECKPOINTS:
        ckpt_dir = DATA_DIR / ckpt
        if not ckpt_dir.exists():
            print(f"[SKIP] {ckpt}", flush=True)
            continue
        # Check L44 data available
        acts_path = ckpt_dir / "activations.npy"
        if not acts_path.exists():
            print(f"[SKIP] {ckpt} — no activations.npy", flush=True)
            continue

        print(f"\n=== {ckpt} ===", flush=True)
        transport_results.append(fit_l25_to_l44_transport(ckpt))
        ladder_rows = fit_layer_ladder(ckpt)
        ladder_results[ckpt] = ladder_rows

    # Save JSON
    output = dict(l25_l44_transport=transport_results, ladder=ladder_results)
    json_path = OUT_DIR / "layer_transport.json"
    with open(json_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSAVED {json_path}", flush=True)

    plot_results(transport_results, ladder_results, OUT_DIR / "layer_transport.png")

    # Print summary
    print("\n=== LAYER TRANSPORT SUMMARY ===")
    print(f"{'checkpoint':<30} {'isometry_defect':>18} {'degree':>8}")
    for r in transport_results:
        print(f"  {r['ckpt']:<28} "
              f"{r.get('isometry_defect', float('nan')):>18.4f} "
              f"{r.get('degree', '?'):>8}")
    print()


if __name__ == "__main__":
    main()
