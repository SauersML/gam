#!/usr/bin/env python3
"""Render the eight native analytic zoo objects before embedding or mixing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.manifold_zoo_geometry import (
    ZOO,
    ZOO_ORDER,
    first_coordinate_hue,
    validate_analytic_sample,
)


CAMERAS = {
    "segment": (20, -58),
    "circle": (90, -90),
    "disk": (90, -90),
    "sphere": (20, -42),
    "torus": (28, -52),
    "mobius": (26, -62),
    "swiss": (24, -60),
    "helix": (18, -58),
}
SUBTITLES = {
    "segment": "x = t",
    "circle": "x² + y² = 1",
    "disk": "x² + y² ≤ 1",
    "sphere": "x² + y² + z² = 1",
    "torus": "(√(x²+y²)−1)² + z² = 0.4²",
    "mobius": "one half-twist",
    "swiss": "radius = angle\nheight ∈ [−10.5, 10.5]",
    "helix": "radius = 1\nz = 0.25 angle",
}


def _pad(points: np.ndarray) -> np.ndarray:
    return np.pad(points, ((0, 0), (0, 3 - points.shape[1])))


def _scatter(ax: Any, kind: str, points: np.ndarray, parameters: np.ndarray) -> None:
    cloud = _pad(points)
    hue = first_coordinate_hue(kind, parameters)
    cmap = "twilight" if kind in {"circle", "torus", "mobius"} else "viridis"
    ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c=hue,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=2.4,
        alpha=0.76,
        linewidths=0,
        rasterized=True,
    )
    lower, upper = cloud.min(axis=0), cloud.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.54 * float(np.max(upper - lower)) + 1.0e-9
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.set_proj_type("ortho")
    ax.view_init(*CAMERAS[kind])
    ax.set_axis_off()


def render(out: Path, *, seed: int, points_per_object: int) -> None:
    rng = np.random.default_rng(seed)
    figure = plt.figure(figsize=(20.5, 4.4), dpi=220, facecolor="#fcfcfb")
    grid = figure.add_gridspec(1, len(ZOO_ORDER), left=0.025, right=0.995, top=0.72, bottom=0.02)
    for column, kind in enumerate(ZOO_ORDER):
        points, parameters = ZOO[kind].sampler(rng, points_per_object)
        validate_analytic_sample(kind, points, parameters)
        axis = figure.add_subplot(grid[0, column], projection="3d")
        _scatter(axis, kind, points, parameters)
        axis.set_title(
            f"{kind.replace('swiss', 'Swiss roll').title()}\n{SUBTITLES[kind]}",
            fontsize=9.8,
            fontweight="semibold",
            color="#111111",
            linespacing=1.2,
            pad=0,
        )
    figure.text(
        0.025,
        0.965,
        "ANALYTIC MANIFOLD ZOO · GROUND TRUTH",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#111111",
    )
    figure.text(
        0.025,
        0.86,
        "Native coordinates only · before centering, random isometry, superposition, or fitting",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#595754",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=figure.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--points", type=int, default=3500)
    args = parser.parse_args()
    render(args.out, seed=args.seed, points_per_object=args.points)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
