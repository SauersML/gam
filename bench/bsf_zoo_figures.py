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
ATLAS_KINDS = ["segment", "circle", "disk", "sphere", "torus", "mobius", "swiss", "helix"]
ATLAS_LABELS = {
    "segment": ("Segment", "bounded · 1D"),
    "circle": ("Circle", "closed · 1D"),
    "disk": ("Disk", "bounded · 2D"),
    "sphere": ("Sphere", "closed · 2D"),
    "torus": ("Torus", "product · 2D"),
    "mobius": ("Möbius", "non-orientable · 2D"),
    "swiss": ("Swiss roll", "open · 2D"),
    "helix": ("Helix", "open · 1D"),
}
ATLAS_CAMERA = {
    "segment": (20, -58),
    "circle": (28, -58),
    "disk": (26, -55),
    "sphere": (22, -42),
    "torus": (30, -52),
    "mobius": (28, -62),
    "swiss": (24, -60),
    "helix": (22, -52),
}


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


def fig_gallery(clouds_dir: Path, out: Path, *, max_factors: int = 8) -> None:
    files = sorted(clouds_dir.glob("clouds_*.npz"))
    if not files:
        return
    stores = {}
    for path in files:
        npz = np.load(path)
        stores[_cloud_meta(npz)["featurizer"]] = npz
    col_order = [k for k in ("ours_rust", "ours_torch", "flat_topk", "oracle") if k in stores]
    if not col_order:
        return
    ref = stores[col_order[0]]
    factors = [m for m in _cloud_meta(ref)["factors"]][:max_factors]
    n_rows, n_cols = len(factors), 1 + len(col_order)
    fig = plt.figure(figsize=(2.4 * n_cols, 2.0 * n_rows), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for r, meta in enumerate(factors):
        i = meta["factor"]
        theta = np.asarray(ref[f"theta_{i}"])
        hue = theta[:, 0]
        cyclic = meta["kind"] in ("circle", "torus", "mobius")
        cmap = plt.get_cmap("twilight" if cyclic else "viridis")
        panels = [("true", ref[f"true_{i}"])]
        for name in col_order:
            store = stores[name]
            key = f"rec_{i}"
            panels.append((name, store[key] if key in store else None))
        # One shared cube per row sized by the union. This keeps every column on
        # the same honest scale without silently clipping a bad reconstruction.
        finite_clouds = [cloud for _, cloud in panels if cloud is not None]
        lim = max(float(np.max(np.abs(cloud))) for cloud in finite_clouds) * 1.08 + 1e-9
        for c, (name, cloud) in enumerate(panels):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            ax.set_facecolor(SURFACE)
            if cloud is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1],
                           cloud[:, 2] if cloud.shape[1] > 2 else np.zeros(len(cloud)),
                           c=hue[: len(cloud)], cmap=cmap, s=2.2, alpha=0.75,
                           linewidths=0)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
            else:
                ax.text2D(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                          color=INK_MUTED, fontsize=9)
            ax.set_axis_off()
            if r == 0:
                title = {"true": "ground truth"}.get(name, LABELS.get(name, name))
                ax.set_title(title, fontsize=9.5, color=INK, pad=2)
            if c == 0:
                ax.text2D(-0.12, 0.5, meta["kind"],
                          transform=ax.transAxes, ha="right", va="center",
                          fontsize=8.5, color=INK_2)
    fig.suptitle("Recovered concept manifolds — hue is the TRUE intrinsic coordinate",
                 fontsize=12.5, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.01,
                        hspace=0.02, wspace=0.02)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _atlas_cloud_records(
    clouds_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load one joint newest-code ours_rust fit containing every atlas kind."""
    paths = sorted(clouds_dir.rglob("clouds_ours_rust*.npz"))
    if len(paths) != 1:
        raise ValueError(
            "the atlas requires exactly one joint ours_rust cloud file; "
            f"found {len(paths)}"
        )
    path = paths[0]
    records: dict[str, dict[str, Any]] = {}
    with np.load(path) as store:
        meta = _cloud_meta(store)
        if meta.get("schema") != "joint-manifold-sae-analytic-clouds-v2":
            raise ValueError("cloud file does not use the native analytic joint-fit schema")
        if meta.get("coordinate_space") != "native-analytic":
            raise ValueError("cloud file does not contain native analytic coordinates")
        if meta.get("featurizer") != "ours_rust" or meta.get("joint_fit") is not True:
            raise ValueError("cloud file is not a joint Rust Manifold SAE fit")
        data_config = meta.get("data_config") or {}
        if data_config.get("dgp") != "toy":
            raise ValueError("atlas ground truth must use the nuisance-free analytic toy DGP")
        if data_config.get("kinds") != ATLAS_KINDS:
            raise ValueError(
                "joint fit must contain the eight analytic zoo factors in atlas order"
            )
        fit_config = meta.get("fit_config") or {}
        if fit_config.get("assignment") != "topk":
            raise ValueError("joint fit must use a shared Top-K sparse assignment")
        if fit_config.get("atoms") != len(ATLAS_KINDS):
            raise ValueError("joint fit must use one shared atom bank of size eight")
        if fit_config.get("top_k") != data_config.get("l0"):
            raise ValueError("joint fit Top-K must match the planted row support")
        if meta.get("matching") != "hungarian-exact-one-to-one":
            raise ValueError("joint fit must use exact one-to-one atom/factor matching")
        if meta.get("n_unique_matched_atoms") != len(ATLAS_KINDS):
            raise ValueError("joint fit has collapsed or unmatched atoms")
        matched_atoms: set[int] = set()
        for factor in meta["factors"]:
            kind = str(factor["kind"])
            if kind not in ATLAS_KINDS:
                continue
            if kind in records:
                raise ValueError(f"joint cloud file contains duplicate kind {kind}")
            index = int(factor["factor"])
            matched_atom = int(factor["matched_atom"])
            if matched_atom in matched_atoms:
                raise ValueError(f"joint cloud file reuses learned atom {matched_atom}")
            matched_atoms.add(matched_atom)
            records[kind] = {
                "path": path,
                "true": np.asarray(store[f"true_{index}"], dtype=float),
                "recovered": np.asarray(store[f"rec_{index}"], dtype=float),
                "theta": np.asarray(store[f"theta_{index}"], dtype=float),
                "r2": float(factor["r2"]),
            }
    missing = [kind for kind in ATLAS_KINDS if kind not in records]
    if missing:
        raise ValueError(f"atlas is missing ours_rust clouds for: {', '.join(missing)}")
    return records, meta


def _pad_cloud(cloud: np.ndarray) -> np.ndarray:
    if cloud.ndim != 2 or cloud.shape[1] == 0 or cloud.shape[1] > 3:
        raise ValueError(f"atlas cloud must have one to three columns; got {cloud.shape}")
    if cloud.shape[1] == 3:
        return cloud
    return np.pad(cloud, ((0, 0), (0, 3 - cloud.shape[1])))


def _atlas_hue(kind: str, theta: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(theta[:, 0], dtype=float)
    if kind in ("circle", "torus", "mobius"):
        return np.mod(values, 2.0 * np.pi) / (2.0 * np.pi), "twilight"
    lo, hi = float(values.min()), float(values.max())
    scale = hi - lo
    return (values - lo) / (scale if scale > 0.0 else 1.0), "viridis"


def _atlas_scatter(
    ax: Any,
    cloud: np.ndarray,
    hue: np.ndarray,
    cmap: str,
    center: np.ndarray,
    radius: float,
    camera: tuple[float, float],
) -> None:
    ax.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2], c=hue, cmap=cmap,
        vmin=0.0, vmax=1.0, s=3.2, alpha=0.80, linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.24)
    ax.set_proj_type("ortho")
    ax.view_init(elev=camera[0], azim=camera[1])
    ax.set_axis_off()


def fig_all_zoos_atlas(
    clouds_dir: Path,
    smooth_zoo: Path,
    amm_r2: Path,
    amm_topology: Path,
    out: Path,
    *,
    source_sha: str,
    amm_source_sha: str,
) -> None:
    """One fitted atlas spanning the manifold, smooth, and 28-factor AMM zoos."""
    if not source_sha or not amm_source_sha:
        raise ValueError("both source SHAs are required for a provenance-bearing atlas")
    records, cloud_meta = _atlas_cloud_records(clouds_dir)
    data_config = cloud_meta["data_config"]
    fit_config = cloud_meta["fit_config"]
    smooth = plt.imread(smooth_zoo)
    amm_r2_image = plt.imread(amm_r2)
    amm_topology_image = plt.imread(amm_topology)

    n_kinds = len(ATLAS_KINDS)
    fig = plt.figure(figsize=(21.0, 18.8), dpi=210, facecolor=SURFACE)
    grid = fig.add_gridspec(
        4, n_kinds,
        height_ratios=(1.0, 1.0, 1.36, 1.36),
        hspace=0.035,
        wspace=0.015,
        left=0.055,
        right=0.995,
        top=0.85,
        bottom=0.035,
    )
    for column, kind in enumerate(ATLAS_KINDS):
        record = records[kind]
        true = _pad_cloud(record["true"])
        recovered = _pad_cloud(record["recovered"])
        union = np.vstack((true, recovered))
        lower = union.min(axis=0)
        upper = union.max(axis=0)
        center = 0.5 * (lower + upper)
        radius = 0.54 * float(np.max(upper - lower)) + 1.0e-9
        hue, cmap = _atlas_hue(kind, record["theta"])

        planted_ax = fig.add_subplot(grid[0, column], projection="3d")
        recovered_ax = fig.add_subplot(grid[1, column], projection="3d")
        for ax, cloud in ((planted_ax, true), (recovered_ax, recovered)):
            ax.set_facecolor(SURFACE)
            _atlas_scatter(
                ax, cloud, hue[: len(cloud)], cmap, center, radius,
                ATLAS_CAMERA[kind],
            )
        title, subtitle = ATLAS_LABELS[kind]
        planted_ax.set_title(
            f"{title}\n{subtitle}", fontsize=11.2, color=INK, pad=0,
            fontweight="semibold", linespacing=1.35,
        )
        recovered_ax.text2D(
            0.07, 0.07, f"held-out R²  {record['r2']:.3f}",
            transform=recovered_ax.transAxes, ha="left", va="bottom",
            fontsize=8.3, color=INK,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white",
                  "edgecolor": GRID, "linewidth": 0.6, "alpha": 0.92},
        )

    fig.text(
        0.017, 0.764, "PLANTED", rotation=90, ha="center", va="center",
        fontsize=8.5, color=INK_MUTED, fontweight="bold",
    )
    fig.text(
        0.017, 0.602, "ONE JOINT MANIFOLD SAE", rotation=90, ha="center", va="center",
        fontsize=8.5, color=SERIES["ours_rust"], fontweight="bold",
    )

    smooth_ax = fig.add_subplot(grid[2, :])
    smooth_ax.imshow(smooth)
    smooth_ax.set_axis_off()
    fig.text(
        0.017, 0.405, "GAM SMOOTH ZOO", rotation=90, ha="center", va="center",
        fontsize=8.5, color=INK_MUTED, fontweight="bold",
    )

    amm_r2_ax = fig.add_subplot(grid[3, : n_kinds // 2])
    amm_topology_ax = fig.add_subplot(grid[3, n_kinds // 2 :])
    for ax, image in ((amm_r2_ax, amm_r2_image), (amm_topology_ax, amm_topology_image)):
        ax.imshow(image)
        ax.set_axis_off()
    fig.text(
        0.017, 0.158, "AMM 28-FACTOR ZOO", rotation=90, ha="center", va="center",
        fontsize=8.5, color=INK_MUTED, fontweight="bold",
    )

    fig.text(
        0.055, 0.974, "ALL GEOMETRY ZOOS", ha="left", va="top",
        fontsize=23, color=INK, fontweight="bold",
    )
    fig.text(
        0.055, 0.936,
        f"Eight exact analytic factor types, {data_config['l0']} superposed per sample, "
        f"recovered by one shared {fit_config['atoms']}-atom "
        f"Top-{fit_config['top_k']} Rust Manifold SAE",
        ha="left", va="top", fontsize=11.2, color=INK_2,
    )
    fig.text(
        0.055, 0.908,
        f"Nuisance-free toy DGP · row support {data_config['l0']} · "
        f"ambient R^{data_config['ambient']} · held-out atom images from native decoding",
        ha="left", va="top", fontsize=8.8, color=INK_MUTED,
    )
    fig.text(
        0.995, 0.018,
        f"gam {source_sha[:12]}  ·  Manifold-SAE {amm_source_sha[:12]}  ·  "
        "held-out clouds  ·  color = planted intrinsic coordinate",
        ha="right", va="bottom", fontsize=7.5, color=INK_MUTED,
    )
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)
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
    parser.add_argument("--smooth-zoo", default=None)
    parser.add_argument("--amm-r2", default=None)
    parser.add_argument("--amm-topology", default=None)
    parser.add_argument("--source-sha", default=None)
    parser.add_argument("--amm-source-sha", default=None)
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
    if args.smooth_zoo:
        required = {
            "--clouds-dir": args.clouds_dir,
            "--amm-r2": args.amm_r2,
            "--amm-topology": args.amm_topology,
            "--source-sha": args.source_sha,
            "--amm-source-sha": args.amm_source_sha,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"--smooth-zoo also requires {', '.join(missing)}")
        fig_all_zoos_atlas(
            Path(args.clouds_dir),
            Path(args.smooth_zoo),
            Path(args.amm_r2),
            Path(args.amm_topology),
            out_dir / "all_zoos_atlas.png",
            source_sha=args.source_sha,
            amm_source_sha=args.amm_source_sha,
        )
    print(f"figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
