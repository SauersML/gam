"""Sample a noisy 3D circle, fit cyclic GAM coordinates, and plot the curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit


def make_circle_3d(n: int = 220, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Tilted noisy circle in R^3, parameterized by theta in [0, 2*pi)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)

    theta_spike = 2.0 * np.pi / 3.0
    d = np.arctan2(np.sin(theta - theta_spike), np.cos(theta - theta_spike))
    r = 1.0 + 0.55 * np.exp(-0.5 * (d / 0.16) ** 2)
    cx = r * np.cos(theta)
    cy = r * np.sin(theta)
    tilt = np.deg2rad(30.0)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    clean = R @ np.vstack([cx, cy, np.zeros_like(theta)])

    noisy = clean + 0.08 * rng.standard_normal(clean.shape)
    return theta, noisy[0], noisy[1], noisy[2], clean[0], clean[1], clean[2]


def fit_cyclic(theta: np.ndarray, y: np.ndarray):
    return gamfit.fit(
        {"theta": theta.tolist(), "y": y.tolist()},
        "y ~ s(theta, periodic=true, period=2*pi)",
    )


def predict_curve(models: tuple[object, ...], n_grid: int = 401) -> tuple[np.ndarray, ...]:
    grid = np.linspace(0.0, 2.0 * np.pi, n_grid)
    payload = {"theta": grid.tolist()}
    fits = [
        np.asarray(model.predict(payload, return_type="dict")["mean"], dtype=float)
        for model in models
    ]
    return grid, *fits


def main() -> Path:
    theta, nx, ny, nz, tx, ty, tz = make_circle_3d()

    mx = fit_cyclic(theta, nx)
    my = fit_cyclic(theta, ny)
    mz = fit_cyclic(theta, nz)

    _, fx, fy, fz = predict_curve((mx, my, mz))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(nx, ny, nz, s=10, color="#9ca3af", alpha=0.6, label="noisy samples")
    order = np.argsort(theta)
    ax.plot(
        tx[order], ty[order], tz[order],
        color="#10b981", linewidth=1.0, linestyle="--", alpha=0.7, label="truth",
    )
    ax.plot(fx, fy, fz, color="#dc2626", linewidth=2.2, label="cyclic GAM fit")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Cyclic GAM fit of a tilted 3D circle")
    ax.legend(loc="upper left")

    for setter, axis in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim),
        (np.concatenate([nx, fx]), np.concatenate([ny, fy]), np.concatenate([nz, fz])),
    ):
        lo, hi = axis.min(), axis.max()
        pad = 0.1 * (hi - lo)
        setter(lo - pad, hi + pad)

    out = Path(__file__).resolve().parent / "circle_3d_cyclic_demo.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
