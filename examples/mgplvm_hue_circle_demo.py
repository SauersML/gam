"""Smoke demo for Riemannian LatentCoord updates on hue-like ring data.

This script is intentionally parameterized and not run by the implementation
patch. It compares the old Euclidean latent update against the new circle
manifold update on a synthetic 64-D ring, matching the mGPLVM head-direction /
hue use case where the 2*pi -> 0 seam is the failure mode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import gamfit


@dataclass
class FitMetrics:
    manifold: str
    train_mse: float
    procrustes_r2: float
    seam_jump: float


def make_hue_ring(n: int, p: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=p)
    harmonics = rng.integers(1, 5, size=p)
    amplitudes = rng.uniform(0.4, 1.2, size=p)
    y = np.empty((n, p), dtype=float)
    for j in range(p):
        y[:, j] = amplitudes[j] * np.cos(harmonics[j] * theta + phases[j])
        y[:, j] += 0.35 * np.sin((harmonics[j] + 1) * theta - phases[j])
    y += noise * rng.standard_normal(y.shape)
    y -= y.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(y, axis=1, keepdims=True).clip(min=1e-12)
    return theta, y / scale


def circular_procrustes_r2(theta_true: np.ndarray, theta_hat: np.ndarray) -> float:
    z_true = np.column_stack([np.cos(theta_true), np.sin(theta_true)])
    z_hat = np.column_stack([np.cos(theta_hat), np.sin(theta_hat)])
    z_true -= z_true.mean(axis=0, keepdims=True)
    z_hat -= z_hat.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(z_hat.T @ z_true, full_matrices=False)
    aligned = z_hat @ (u @ vt)
    sse = np.sum((aligned - z_true) ** 2)
    sst = np.sum(z_true**2)
    return float(1.0 - sse / sst)


def extract_latent_angle(model: object, n: int) -> np.ndarray:
    values = np.asarray(model.latent("t"), dtype=float)
    values = values.reshape(n, -1)
    return np.mod(values[:, 0], 2.0 * np.pi)


def fit_one(theta: np.ndarray, y: np.ndarray, manifold: str, k: int) -> FitMetrics:
    n, p = y.shape
    data: dict[str, object] = {
        "y0": y[:, 0].tolist(),
        "t": theta.tolist(),
    }
    for j in range(p):
        data[f"y{j}"] = y[:, j].tolist()
    latent = gamfit.LatentCoord(
        n=n,
        d=1,
        init=theta[:, None],
        manifold=manifold,
        aux_prior={"u": theta[:, None], "family": "ridge", "strength": 1.0},
    )
    model = gamfit.fit(
        data,
        f"y0 ~ s(t, type='periodic', k={k}, period={2.0 * np.pi}, origin=0)",
        family="gaussian",
        latents={"t": latent},
        response_geometry="spherical",
        response_columns=[f"y{j}" for j in range(p)],
    )
    pred = np.asarray(model.predict(data), dtype=float)
    theta_hat = extract_latent_angle(model, n)
    order = np.argsort(theta_hat)
    seam_jump = float(abs(np.diff(np.unwrap(theta_hat[order]))).max())
    return FitMetrics(
        manifold=manifold,
        train_mse=float(np.mean((pred - y) ** 2)),
        procrustes_r2=circular_procrustes_r2(theta, theta_hat),
        seam_jump=seam_jump,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=886)
    parser.add_argument("--p", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    theta, y = make_hue_ring(args.n, args.p, args.noise, args.seed)
    rows = [
        fit_one(theta, y, "euclidean", args.k),
        fit_one(theta, y, "circle", args.k),
    ]
    print("manifold,train_mse,procrustes_r2,seam_jump")
    for row in rows:
        print(f"{row.manifold},{row.train_mse:.8g},{row.procrustes_r2:.8g},{row.seam_jump:.8g}")


if __name__ == "__main__":
    main()
