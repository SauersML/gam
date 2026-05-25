"""Noisy 3D circle with a spike, fit by several cyclic GAM families."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit


# RUSTIFY:
# Rust kernel required for sock rendering:
# - add `crate::geometry::tube::closed_curve_sock_mesh`
# - expose it from `crates/gam-pyffi/src/lib.rs` as `gamfit.closed_curve_sock_mesh`
# - inputs: closed curve samples `(K, 3)`, observed points `(N, 3)`,
#   angular neighborhood width, radius floor, and cross-section count
# - outputs: tube mesh coordinate arrays `(X, Y, Z)`
# - behavior: compute cyclic tangents, a closed Bishop frame, nearest-curve
#   point radii, cyclic max-filtered/smoothed enclosing radius, and tube mesh
# This Python script intentionally renders only gamfit outputs and plotting.


def make_circle_3d(n: int = 260, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.arctan2(np.sin(theta - 2.0 * np.pi / 3.0),
                   np.cos(theta - 2.0 * np.pi / 3.0))
    r = 1.0 + 0.55 * np.exp(-0.5 * (d / 0.16) ** 2)
    cx, cy, cz = r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)
    tilt = np.deg2rad(30.0)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt),  np.cos(tilt)],
        ]
    )
    clean = R @ np.vstack([cx, cy, cz])
    sigma = 0.07
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    return theta, noisy.T  # noisy: (n, 3)


def fit_closed_curve(
    theta: np.ndarray, P: np.ndarray, formula_body: str
):
    base = {"ct": np.cos(theta).tolist(), "st": np.sin(theta).tolist()}
    models = []
    for j in range(P.shape[1]):
        data = {**base, "y": P[:, j].tolist()}
        models.append(gamfit.fit(data, f"y ~ {formula_body}"))
    return models


def predict_curve(models: list, grid: np.ndarray) -> np.ndarray:
    payload = {"ct": np.cos(grid).tolist(), "st": np.sin(grid).tolist()}
    cols = []
    for m in models:
        pred = m.predict(payload, return_type="dict")
        cols.append(np.asarray(pred["mean"], dtype=float))
    return np.asarray(np.vstack(cols).T)  # (K, 3)


def main() -> Path:
    theta, P = make_circle_3d()

    specs = [
        ("Duchon (centers=80)", "duchon(ct, st, centers=80)"),
        ("Thin-plate", "thinplate(ct, st)"),
        ("Matern", "matern(ct, st)"),
        ("Tensor", "tensor(ct, st)"),
    ]

    grid = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    fits = []
    for name, body in specs:
        models = fit_closed_curve(theta, P, body)
        curve = predict_curve(models, grid)
        fits.append((name, np.vstack([curve, curve[0]])))

    fig = plt.figure(figsize=(13.5, 11))
    for i, (name, curve) in enumerate(fits):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                   s=8, color="#374151", alpha=0.55, label="data")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                color="#dc2626", linewidth=2.0, label="fit")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for setter, vals in zip(
            (ax.set_xlim, ax.set_ylim, ax.set_zlim),
            (np.concatenate([P[:, 0], curve[:, 0]]),
             np.concatenate([P[:, 1], curve[:, 1]]),
             np.concatenate([P[:, 2], curve[:, 2]])),
        ):
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.08 * (hi - lo)
            setter(lo - pad, hi + pad)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cyclic GAM fits for a noisy 3D circle", fontsize=13)
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "circle_3d_cyclic_sock_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
