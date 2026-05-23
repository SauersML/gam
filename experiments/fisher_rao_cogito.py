#!/usr/bin/env python3
"""Fisher-Rao latent GP-LVM demo for Cogito color centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

import gamfit
from _pca_basis import load_pc_basis


AUTO_54_BASELINE = 0.52


def load_cogito_centroids(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        labels = [str(x) for x in data["labels"]]
        centroids = np.asarray(data["centroids"], dtype=np.float64)
    else:
        centroids = np.asarray(data, dtype=np.float64)
        labels = [f"color_{i:03d}" for i in range(centroids.shape[0])]
    if centroids.shape != (886, 7168):
        raise ValueError(f"expected Cogito centroids shape (886, 7168), got {centroids.shape}")
    return labels, centroids


def project_to_cached_pcs(centroids: np.ndarray, k_pc: int) -> np.ndarray:
    basis = load_pc_basis(k_pc)
    pcs = np.asarray(getattr(basis, "components_", basis), dtype=np.float64)[:k_pc]
    return centroids @ pcs.T


def intervention_client(url: str | None, p: int) -> Callable[[np.ndarray], np.ndarray]:
    if url is None:
        rng = np.random.default_rng(7)
        hue_axis = rng.normal(size=p)
        hue_axis /= np.linalg.norm(hue_axis)

        def mock(pc_point: np.ndarray) -> np.ndarray:
            score = float(pc_point @ hue_axis)
            return np.array([np.sin(score), np.cos(score), score], dtype=np.float64)

        return mock

    import requests

    def call(pc_point: np.ndarray) -> np.ndarray:
        resp = requests.post(
            f"{url.rstrip('/')}/intervene",
            json={"pc": pc_point.tolist()},
            timeout=10,
        )
        resp.raise_for_status()
        return np.asarray(resp.json()["metric"], dtype=np.float64)

    return call


def empirical_fisher_blocks(points: np.ndarray, metric: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    n, p = points.shape
    blocks = np.empty((n, p, p), dtype=np.float64)
    eps = 1.0e-3
    for i, x in enumerate(points):
        jac_cols = []
        for a in range(p):
            step = np.zeros(p, dtype=np.float64)
            step[a] = eps
            jac_cols.append((metric(x + step) - metric(x - step)) / (2.0 * eps))
        jac = np.stack(jac_cols, axis=1)
        blocks[i] = jac.T @ jac + 1.0e-5 * np.eye(p)
    return blocks


def fit_latent(points: np.ndarray, fisher_w: np.ndarray) -> dict:
    n, p = points.shape
    rng = np.random.default_rng(54)
    t0 = rng.normal(scale=0.05, size=(n, 2))
    centers = t0.copy()
    penalty = np.eye(centers.shape[0] + 3, dtype=np.float64)
    penalty[0, 0] = 0.0
    return gamfit.gaussian_reml_fit_latent(
        t0.ravel(),
        points,
        n,
        2,
        centers,
        penalty,
        m=2,
        fisher_w=fisher_w,
        init_lambda=1.0,
    )


def tangent_alignment(fit: dict, labels: list[str], hue_scores: np.ndarray) -> float:
    fitted = np.asarray(fit["fitted"], dtype=np.float64)
    red_idx = labels.index("red") if "red" in labels else int(np.argmax(hue_scores))
    local = fitted[red_idx] - fitted.mean(axis=0)
    local /= np.linalg.norm(local) + 1.0e-12
    hue = hue_scores - hue_scores.mean()
    hue_direction = hue @ fitted
    hue_direction /= np.linalg.norm(hue_direction) + 1.0e-12
    return float(local @ hue_direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--k-pc", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("fisher_rao_cogito_results.json"))
    args = parser.parse_args()

    labels, centroids = load_cogito_centroids(args.centroids)
    pc_points = project_to_cached_pcs(centroids, args.k_pc)
    metric = intervention_client(args.server_url, args.k_pc)
    fisher_w = empirical_fisher_blocks(pc_points, metric)

    plain_w = np.broadcast_to(np.eye(args.k_pc), fisher_w.shape).copy()
    plain = fit_latent(pc_points, plain_w)
    fisher = fit_latent(pc_points, fisher_w)

    hue_scores = np.array([metric(x)[0] for x in pc_points], dtype=np.float64)
    plain_align = tangent_alignment(plain, labels, hue_scores)
    fisher_align = tangent_alignment(fisher, labels, hue_scores)
    result = {
        "auto_54_baseline": AUTO_54_BASELINE,
        "plain_gaussian_alignment": plain_align,
        "fisher_rao_alignment": fisher_align,
        "beats_auto_54": fisher_align > AUTO_54_BASELINE,
        "delta_vs_plain": fisher_align - plain_align,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
