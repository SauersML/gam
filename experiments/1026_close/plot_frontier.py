#!/usr/bin/env python3
"""#1026 EV-vs-K frontier — figure + markdown table from results_1026.jsonl.

Reads the frontier rows appended by run_frontier_arm.sbatch (via
frontier_matrix.sh) and renders a SINGLE EV-vs-dictionary-size figure:

  * external_topk (traditional-SAE bar) and hybrid (our contestant) as two
    lines over K on a log-K axis, each seed drawn as its own marker (seed 0
    light, seed 1 dark) with the seed-mean as the connecting line.
  * pca_bar : the linear-optimum yardstick, a horizontal reference line
    (rank-512 held-out PCA EV, the strongest linear reconstruction).
  * w32k_baseline : the published W32K k=100 flat-SAE number (EV=0.523) as a
    horizontal reference line.

It also writes a markdown table (per-K seed values, seed means, and the
hybrid-minus-external margin) so the frontier can be posted to the issue and
read from the logs alone.

Whatever rows are present get plotted — points that have not landed yet are
simply skipped, so this can be run for an early partial figure.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt

SWEEP_ARMS = ("external_topk", "hybrid")
# Okabe-Ito, colorblind-safe: blue for the bar, vermillion for the contestant.
ARM_COLOR = {"external_topk": "#0072B2", "hybrid": "#D55E00"}
ARM_LABEL = {"external_topk": "external TopK (traditional SAE)",
             "hybrid": "hybrid (flat TopK + curved manifold)"}


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_frontier_sweep(r: dict) -> bool:
    """A sweep point for the EV-vs-K curve. Prefer explicitly frontier-tagged
    rows; fall back to any external_topk/hybrid row if none are tagged (so a
    combined results file still yields a curve)."""
    return r.get("arm") in SWEEP_ARMS


def collect(rows: list[dict]):
    """Return (curve, pca_line, w32k_line).

    curve[arm][K][seed] = ev, choosing frontier-tagged rows when a (arm,K,seed)
    cell has several rows, else the last-written row.
    """
    tagged = any(str(r.get("tag", "")).startswith("frontier")
                 for r in rows if is_frontier_sweep(r))
    curve: dict = {a: defaultdict(dict) for a in SWEEP_ARMS}
    for r in rows:
        if not is_frontier_sweep(r):
            continue
        if tagged and not str(r.get("tag", "")).startswith("frontier"):
            continue
        curve[r["arm"]][int(r["K"])][int(r["seed"])] = float(r["ev"])

    pca_line = None
    pca_ranks: dict = {}
    for r in rows:
        if r.get("arm") == "pca_bar":
            pca_ranks = {k: v for k, v in r.items() if k.startswith("pca_ev_r")}
            # strongest available linear reference = highest rank present
            if pca_ranks:
                top = max(pca_ranks, key=lambda k: int(k.split("_r")[1]))
                pca_line = (top, float(pca_ranks[top]))
    w32k_line = next((float(r["ev"]) for r in rows
                      if r.get("arm") == "w32k_baseline"), None)
    return curve, pca_line, pca_ranks, w32k_line


def make_figure(curve, pca_line, w32k_line, out_png: str):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for arm in SWEEP_ARMS:
        ks = sorted(curve[arm])
        if not ks:
            continue
        means = [sum(curve[arm][k].values()) / len(curve[arm][k]) for k in ks]
        ax.plot(ks, means, "-", color=ARM_COLOR[arm], lw=2.0, zorder=3,
                label=ARM_LABEL[arm])
        # seeds as light/dark markers of the same hue
        for k in ks:
            for seed, ev in curve[arm][k].items():
                alpha = 0.45 if seed == 0 else 1.0
                ax.plot(k, ev, "o", color=ARM_COLOR[arm], alpha=alpha,
                        ms=6, mec="white", mew=0.6, zorder=4)
    if pca_line is not None:
        top, val = pca_line
        rank = top.split("_r")[1]
        ax.axhline(val, ls="--", color="#009E73", lw=1.4, zorder=2,
                   label=f"PCA rank-{rank} (linear optimum) EV={val:.3f}")
    if w32k_line is not None:
        ax.axhline(w32k_line, ls=":", color="#555555", lw=1.4, zorder=2,
                   label=f"W32K k=100 baseline EV={w32k_line:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dictionary size K (log$_2$)")
    ax.set_ylabel("held-out explained variance")
    ax.set_title("#1026 EV-vs-K frontier — hybrid manifold SAE vs traditional TopK\n"
                 "(matched per-token active-scalar budget; seed 0 light / seed 1 dark)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def make_table(curve, pca_ranks, w32k_line, out_md: str):
    ks = sorted({k for arm in SWEEP_ARMS for k in curve[arm]})
    lines = ["# #1026 EV-vs-K frontier", ""]
    lines.append("Held-out EV at matched per-token active-scalar budget "
                 "(top_k=32). Curved tier scales as curved_atoms=max(64, K/64).")
    lines.append("")
    lines.append("| K | ext s0 | ext s1 | ext mean | hyb s0 | hyb s1 | hyb mean | hyb−ext |")
    lines.append("|---|-------|-------|---------|-------|-------|---------|--------|")

    def cell(arm, k, seed):
        v = curve[arm].get(k, {}).get(seed)
        return f"{v:.4f}" if v is not None else "—"

    def mean(arm, k):
        d = curve[arm].get(k, {})
        return (sum(d.values()) / len(d)) if d else None

    for k in ks:
        em, hm = mean("external_topk", k), mean("hybrid", k)
        delta = f"{hm - em:+.4f}" if (em is not None and hm is not None) else "—"
        lines.append(
            f"| {k} | {cell('external_topk', k, 0)} | {cell('external_topk', k, 1)} | "
            f"{('%.4f' % em) if em is not None else '—'} | "
            f"{cell('hybrid', k, 0)} | {cell('hybrid', k, 1)} | "
            f"{('%.4f' % hm) if hm is not None else '—'} | {delta} |")

    lines += ["", "## Reference lines", ""]
    if pca_ranks:
        lines.append("PCA held-out EV (linear optimum, K-independent):")
        for rk in sorted(pca_ranks, key=lambda k: int(k.split("_r")[1])):
            lines.append(f"- rank {rk.split('_r')[1]}: EV={float(pca_ranks[rk]):.4f}")
    if w32k_line is not None:
        lines.append(f"- W32K k=100 flat-SAE baseline (published external): EV={w32k_line:.4f}")
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(__file__), "results", "results_1026.jsonl"))
    ap.add_argument("--outdir", default=None,
                    help="where to write the figure + table (default: results dir)")
    args = ap.parse_args()

    rows = load_rows(args.results)
    curve, pca_line, pca_ranks, w32k_line = collect(rows)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, "frontier_ev_vs_k.png")
    out_md = os.path.join(outdir, "frontier_table.md")

    make_figure(curve, pca_line, w32k_line, out_png)
    table = make_table(curve, pca_ranks, w32k_line, out_md)

    print(f"[frontier] wrote {out_png}")
    print(f"[frontier] wrote {out_md}")
    print("\n".join(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
