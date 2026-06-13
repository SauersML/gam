"""Curvature-as-estimand science on OLMo-3-32B L25 qualia activations.

SCIENCE QUESTION: Is the self/qualia representational manifold measurably
curved (κ̂ CI excludes 0), and does curvature change across training?

METHOD: For each checkpoint, take L25 activations (635×5120), PCA-whiten to
d dimensions, fit a gam Model with the curv() smooth term to estimate κ̂
(signed sectional curvature) via exact REML, report CI + flatness LR test.

This uses gamfit.Model.curvature() which returns:
  - kappa_hat: float
  - ci_lo, ci_hi: profile-likelihood 95% CI
  - verdict: "spherical" / "hyperbolic" / "flat" / "indistinguishable"
  - flatness_lr_pvalue: χ² test of κ=0

We also separate exp vs noexp subgroups to ask: does the qualia-attribution
manifold have different curvature from the control (non-experience) manifold?

Outputs:
  /projects/standard/hsiehph/sauer354/olmo_data/plots/curvature_by_checkpoint.png
  /projects/standard/hsiehph/sauer354/olmo_data/plots/curvature_results.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

DATA_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data")
OUT_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data/plots")
CHECKPOINTS = ["base", "step_2300", "stage1-step0", "stage3-step11921"]
PCA_DIM = 4  # 2D curv() smooth over top-2 PCs after response
RANDOM_STATE = 42


def pca_whiten(X: np.ndarray, n: int) -> np.ndarray:
    mu = X.mean(0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n].T) / (S[:n] + 1e-12)


def fit_curvature(Z: np.ndarray, label: str) -> dict:
    """Fit κ̂ on a whitened embedding via gam Model.curvature()."""
    import gamfit
    import pandas as pd

    n, d = Z.shape
    print(f"  [{label}] fitting curvature on {n}x{d} ...", flush=True)

    # Build a pandas DataFrame: curv() smooth operates on the coordinates
    cols = {f"x{i}": Z[:, i] for i in range(d)}
    df = pd.DataFrame(cols)

    # Formula: Gaussian response = first PC as proxy; curv(x1, x2) fits κ̂
    # for the 2D manifold spanned by PCs 2-3 (a proper sectional curvature).
    # Using only 2 predictor dimensions keeps the basis small and fast.
    response = Z[:, 0]
    df["y"] = response
    formula = "y ~ curv(x1, x2)"

    t0 = time.time()
    try:
        m = gamfit.fit(df, formula, family="gaussian")
        curv_list = m.curvature(df)
        # curvature() returns a list of per-curv()-term dicts
        # We use the first term (we only have one curv() block)
        result = curv_list[0] if curv_list else {}
        elapsed = time.time() - t0
        khat = float(result.get("kappa_hat", float("nan")))
        ci_lo = float(result.get("ci_lo", float("nan")))
        ci_hi = float(result.get("ci_hi", float("nan")))
        verdict = str(result.get("verdict", "unknown"))
        print(f"  [{label}] done in {elapsed:.1f}s  κ̂={khat:.4f} "
              f"CI=[{ci_lo:.4f},{ci_hi:.4f}] verdict={verdict}", flush=True)
        return dict(
            label=label,
            n=n, d=d,
            kappa_hat=khat,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            verdict=verdict,
            flatness_lr_pvalue=float(result.get("flatness_p_value",
                                                 result.get("flatness_lr_pvalue", float("nan")))),
            elapsed=elapsed,
            error=None,
        )
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"  [{label}] FAILED in {elapsed:.1f}s: {e}", flush=True)
        print(tb[:500], flush=True)
        return dict(
            label=label, n=n, d=d,
            kappa_hat=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
            verdict="error", flatness_lr_pvalue=float("nan"),
            elapsed=elapsed, error=str(e)[:300],
        )


def run_checkpoint(ckpt: str) -> list[dict]:
    acts = np.load(DATA_DIR / ckpt / "activations.npy", mmap_mode="r")
    X_raw = np.array(acts[:, 25, :], dtype=np.float32)

    with open(DATA_DIR / ckpt / "prompts.jsonl") as fh:
        prompts = [json.loads(l) for l in fh]
    sides = np.array([p.get("side", "-") for p in prompts])

    Z_all = pca_whiten(X_raw, PCA_DIM)

    rows = []

    # 1. All data
    rows.append(fit_curvature(Z_all, f"{ckpt}/all"))

    # 2. exp-only subgroup
    exp_mask = sides == "exp"
    if exp_mask.sum() >= 20:
        rows.append(fit_curvature(Z_all[exp_mask], f"{ckpt}/exp"))

    # 3. noexp-only subgroup
    noexp_mask = sides == "noexp"
    if noexp_mask.sum() >= 20:
        rows.append(fit_curvature(Z_all[noexp_mask], f"{ckpt}/noexp"))

    return rows


def plot_results(all_rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Separate by subgroup type
    all_group = [r for r in all_rows if r["label"].endswith("/all")]
    exp_group = [r for r in all_rows if r["label"].endswith("/exp")]
    noexp_group = [r for r in all_rows if r["label"].endswith("/noexp")]

    ckpt_order = CHECKPOINTS
    ckpt_labels = ["base", "step_2300", "stage1\nstep0", "stage3\nstep11921"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    def plot_group(ax, group, title, color):
        ckpt_to_row = {r["label"].split("/")[0]: r for r in group}
        xs, ys, lo_errs, hi_errs, verdicts = [], [], [], [], []
        for i, ckpt in enumerate(ckpt_order):
            r = ckpt_to_row.get(ckpt)
            if r and not np.isnan(r["kappa_hat"]):
                xs.append(i)
                ys.append(r["kappa_hat"])
                lo_errs.append(r["kappa_hat"] - r["ci_lo"])
                hi_errs.append(r["ci_hi"] - r["kappa_hat"])
                verdicts.append(r["verdict"])

        if xs:
            ax.errorbar(xs, ys, yerr=[lo_errs, hi_errs],
                        fmt="o", color=color, capsize=5, linewidth=1.5,
                        markersize=7, label=title)
            for x, y, v in zip(xs, ys, verdicts):
                ax.annotate(v[:4], (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color=color)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="κ=0 (flat)")
        ax.set_xticks(range(len(ckpt_order)))
        ax.set_xticklabels(ckpt_labels, fontsize=8)
        ax.set_xlabel("Checkpoint")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylabel("κ̂ (signed sectional curvature)")

    plot_group(axes[0], all_group, "All prompts", "#2196F3")
    plot_group(axes[1], exp_group, "Exp (experience-attr.) only", "#E91E63")
    plot_group(axes[2], noexp_group, "Noexp (non-exp.) only", "#4CAF50")

    fig.suptitle(
        "OLMo-3-32B — L25 qualia manifold curvature κ̂ by checkpoint\n"
        f"PCA-{PCA_DIM} whitened activations, 95% profile-likelihood CI",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"\nSAVED {out_path}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for ckpt in CHECKPOINTS:
        ckpt_path = DATA_DIR / ckpt
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt} — directory not found", flush=True)
            continue
        print(f"\n=== {ckpt} ===", flush=True)
        rows = run_checkpoint(ckpt)
        all_rows.extend(rows)

    # Save JSON results
    json_path = OUT_DIR / "curvature_results.json"
    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nSAVED {json_path}", flush=True)

    plot_results(all_rows, OUT_DIR / "curvature_by_checkpoint.png")

    # Print summary table
    print("\n=== CURVATURE SUMMARY ===")
    print(f"{'label':<35} {'κ̂':>8} {'ci_lo':>8} {'ci_hi':>8} {'verdict':<18} {'p_flat':>10}")
    for r in all_rows:
        print(
            f"{r['label']:<35} "
            f"{r['kappa_hat']:>8.4f} "
            f"{r['ci_lo']:>8.4f} "
            f"{r['ci_hi']:>8.4f} "
            f"{r['verdict']:<18} "
            f"{r['flatness_lr_pvalue']:>10.4f}"
        )
    print()


if __name__ == "__main__":
    main()
