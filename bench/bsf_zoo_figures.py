#!/usr/bin/env python3
"""Publication figures for the manifold-zoo arena (``bench/bsf_manifold_zoo.py``).

Reads the arena's JSONL results (and the optional NPZ cloud dumps) and renders:

* ``gallery``     -- 3-D true-vs-recovered contribution clouds, one row per
  curved factor, colored by the factor's TRUE intrinsic coordinate so hue
  reports *where on the concept* a point lies (the BSF paper's Fig. 3 reading;
  a featurizer that recovers the manifold reproduces the hue gradient, one that
  shatters it produces hue noise).
* ``leaderboard`` -- per-featurizer mean contribution R^2 with per-kind detail
  dots and the oracle ceiling as a reference rule.
* ``mdl``         -- description length at each distortion floor, grouped per
  featurizer; the support/code/residual split shown as stacked segments.
* ``dimensionality`` -- stable rank of recovered atoms and active coordinates
  per activation (the "a circle is one coordinate" panel).
* ``pareto``      -- recovery R^2 against active coords/token across every
  record in the JSONL (populated by sweep runs).

Design system: the dataviz reference palette; categorical hues in fixed order
(ours_rust blue, ours_torch aqua, flat yellow), the oracle is a neutral
reference rule (not a series), text in ink tokens, recessive grid, thin marks.

Usage::

    python3 bsf_zoo_figures.py --results bsf_zoo_results.jsonl \
        --clouds-dir clouds/ --out-dir figures/bsf_zoo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e8e7e3"
SERIES = {
    "ours_rust": "#2a78d6",
    "ours_torch": "#1baf7a",
    "flat_topk": "#eda100",
}
ORACLE_COLOR = "#8a8984"
LABELS = {
    "ours_rust": "Manifold SAE (Rust REML)",
    "ours_torch": "Manifold SAE (torch)",
    "flat_topk": "Flat TopK SAE",
    "oracle": "Oracle (true subspaces)",
}
KIND_ORDER = ["segment", "circle", "disk", "sphere", "torus", "mobius", "swiss", "helix"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load_results(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    config = None
    results = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "config":
                config = rec
            elif rec.get("record") == "result":
                results.append(rec)
    return config, results


def _latest_per_featurizer(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for rec in results:
        latest[rec["featurizer"]] = rec
    return latest


# --------------------------------------------------------------------------- #
# Leaderboard                                                                 #
# --------------------------------------------------------------------------- #


def fig_leaderboard(results: list[dict[str, Any]], out: Path) -> None:
    latest = _latest_per_featurizer(results)
    oracle = latest.pop("oracle", None)
    order = [k for k in ("ours_rust", "ours_torch", "flat_topk") if k in latest]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.85 * len(order)), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(order))[::-1]
    for y, name in zip(ys, order):
        rec = latest[name]
        r2 = rec["recovery_r2_mean"]
        ax.barh(y, r2, height=0.52, color=SERIES[name], zorder=3)
        ax.text(r2 + 0.012, y, f"{r2:.3f}", va="center", ha="left",
                fontsize=10, color=INK, fontweight="bold")
        ax.text(-0.015, y, LABELS[name], va="center", ha="right",
                fontsize=10, color=INK)
        by_kind = rec.get("recovery_r2_by_kind", {})
        for kind in KIND_ORDER:
            if kind in by_kind:
                ax.plot(by_kind[kind], y - 0.34, marker="o", markersize=3.4,
                        color=SERIES[name], alpha=0.55, zorder=4)
    if oracle is not None:
        ceiling = oracle["recovery_r2_mean"]
        ax.axvline(ceiling, color=ORACLE_COLOR, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(ceiling - 0.012, float(ys[0]) + 0.42, f"oracle {ceiling:.3f}",
                fontsize=8.5, color=INK_2, ha="right", va="bottom")
    ax.set_xlim(0, 1.06)
    ax.set_yticks([])
    ax.set_xlabel("per-factor contribution R² (matched atom alone, held out)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Manifold recovery on the additive manifold-superposition toy",
                 fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Recovery gallery (3-D clouds, hue = true intrinsic coordinate)              #
# --------------------------------------------------------------------------- #


def _cloud_meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(bytes(npz["meta_json"]).decode())


def fig_gallery(clouds_dir: Path, out: Path, *, max_factors: int = 6) -> None:
    files = sorted(clouds_dir.glob("clouds_*.npz"))
    if not files:
        return
    stores = {}
    for path in files:
        npz = np.load(path)
        stores[_cloud_meta(npz)["featurizer"]] = npz
    col_order = [k for k in ("ours_rust", "ours_torch", "flat_topk") if k in stores]
    if not col_order:
        return
    ref = stores[col_order[0]]
    factors = [m for m in _cloud_meta(ref)["factors"]][:max_factors]
    n_rows, n_cols = len(factors), 1 + len(col_order)
    fig = plt.figure(figsize=(2.5 * n_cols, 2.35 * n_rows), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for r, meta in enumerate(factors):
        i = meta["factor"]
        theta = np.asarray(ref[f"theta_{i}"])
        hue = theta[:, 0]
        cyclic = meta["kind"] in ("circle", "sphere", "torus", "mobius", "helix")
        cmap = plt.get_cmap("twilight" if cyclic else "viridis")
        panels = [("true", ref[f"true_{i}"])]
        for name in col_order:
            store = stores[name]
            key = f"rec_{i}"
            panels.append((name, store[key] if key in store else None))
        for c, (name, cloud) in enumerate(panels):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            ax.set_facecolor(SURFACE)
            if cloud is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1],
                           cloud[:, 2] if cloud.shape[1] > 2 else np.zeros(len(cloud)),
                           c=hue[: len(cloud)], cmap=cmap, s=2.2, alpha=0.75,
                           linewidths=0)
            else:
                ax.text2D(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                          color=INK_MUTED, fontsize=9)
            ax.set_axis_off()
            if r == 0:
                title = {"true": "ground truth"}.get(name, LABELS.get(name, name))
                ax.set_title(title, fontsize=9.5, color=INK, pad=2)
            if c == 0:
                ax.text2D(-0.12, 0.5, f"{meta['kind']}\nR²={meta['r2']:.2f}",
                          transform=ax.transAxes, ha="right", va="center",
                          fontsize=8.5, color=INK_2)
    fig.suptitle("Recovered concept manifolds — hue is the TRUE intrinsic coordinate",
                 fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Description length                                                          #
# --------------------------------------------------------------------------- #


def fig_mdl(results: list[dict[str, Any]], out: Path) -> None:
    latest = _latest_per_featurizer(results)
    latest.pop("oracle", None)
    order = [k for k in ("ours_rust", "ours_torch", "flat_topk") if k in latest]
    if not order:
        return
    targets = [0.99, 0.95, 0.90, 0.80]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    xs = np.array([1.0 - t for t in targets])
    for name in order:
        mdl = latest[name]["mdl"]
        ys = [mdl.get(f"bits_at_r2_{t:g}") for t in targets]
        if any(v is None for v in ys):
            continue
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                color=SERIES[name], label=LABELS[name])
        ax.annotate(LABELS[name], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8.5, color=INK, va="center")
        native = mdl.get("native_bits_per_token")
        if native is not None:
            ax.axhline(native, color=SERIES[name], linewidth=1.0,
                       linestyle=(0, (2, 3)), alpha=0.7)
            ax.text(xs[0], native, f" native bits/token {native:.1f}",
                    fontsize=8, color=SERIES[name], va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{100 * x:g}%" for x in xs])
    ax.set_xlabel("distortion floor (1 − target R²)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("description length (bits / token)", fontsize=9.5, color=INK_2)
    ax.set_title("Description length: fewer bits at every distortion floor",
                 fontsize=12, color=INK, loc="left", pad=10)
    if len(order) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dimensionality: stable rank + coords per activation                         #
# --------------------------------------------------------------------------- #


def fig_dimensionality(results: list[dict[str, Any]], out: Path) -> None:
    latest = _latest_per_featurizer(results)
    latest.pop("oracle", None)
    order = [k for k in ("ours_rust", "ours_torch", "flat_topk") if k in latest]
    if not order:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
        ax.grid(axis="x", visible=False)
    xs = np.arange(len(order))
    ranks = [latest[n]["dimensionality"]["stable_rank_mean"] for n in order]
    coords = [latest[n]["coords_per_activation_mean"] for n in order]
    for ax, vals, title in (
        (axes[0], ranks, "mean stable rank of recovered atoms"),
        (axes[1], coords, "active coordinates per activation"),
    ):
        for x, name, v in zip(xs, order, vals):
            ax.bar(x, v, width=0.5, color=SERIES[name], zorder=3)
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color=INK, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[n].replace(" SAE", "\nSAE").replace(" (", "\n(")
                            for n in order], fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    fig.suptitle("Concept dimensionality: intrinsic coordinates, not block budget",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pareto across sweep records                                                 #
# --------------------------------------------------------------------------- #


def fig_pareto(results: list[dict[str, Any]], out: Path) -> None:
    pts: dict[str, list[tuple[float, float]]] = {}
    for rec in results:
        name = rec["featurizer"]
        if name == "oracle":
            continue
        pts.setdefault(name, []).append(
            (rec["coords_per_activation_mean"], rec["recovery_r2_mean"])
        )
    if sum(len(v) for v in pts.values()) < 4:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for name, vals in pts.items():
        arr = np.array(sorted(vals))
        ax.plot(arr[:, 0], arr[:, 1], marker="o", markersize=6, linewidth=1.8,
                color=SERIES.get(name, INK_MUTED), label=LABELS.get(name, name))
    ax.set_xlabel("active coordinates per activation", fontsize=9.5, color=INK_2)
    ax.set_ylabel("recovery R²", fontsize=9.5, color=INK_2)
    ax.set_title("Recovery per transmitted coordinate", fontsize=12, color=INK,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--clouds-dir", default=None)
    parser.add_argument("--out-dir", default="figures/bsf_zoo")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _config, results = load_results(Path(args.results))
    if not results:
        raise SystemExit("no result records found")
    fig_leaderboard(results, out_dir / "leaderboard.png")
    fig_mdl(results, out_dir / "mdl.png")
    fig_dimensionality(results, out_dir / "dimensionality.png")
    fig_pareto(results, out_dir / "pareto.png")
    if args.clouds_dir:
        fig_gallery(Path(args.clouds_dir), out_dir / "gallery.png")
    print(f"figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
