"""Sample noisy points along a 3D circle, fit cyclic GAMs for each coordinate,
then plot the fitted curve against the noisy data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit
from gamfit import Model, ResponseGeometryModel


def make_circle_3d(n: int = 220, seed: int = 0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Tilted unit circle in R^3, parameterized by theta in [0, 2*pi)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)

    # Circle in a tilted plane: rotate the (cos, sin, 0) circle by ~30 deg about x.
    # Add a localized outward radial spike near theta = 2*pi/3 to break symmetry.
    theta_spike = 2.0 * np.pi / 3.0
    # Wrap-aware angular distance
    d = np.arctan2(np.sin(theta - theta_spike), np.cos(theta - theta_spike))
    spike = 0.55 * np.exp(-0.5 * (d / 0.16) ** 2)
    r = 1.0 + spike
    cx = r * np.cos(theta)
    cy = r * np.sin(theta)
    cz = np.zeros_like(theta)
    tilt = np.deg2rad(30.0)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt),  np.cos(tilt)],
        ]
    )
    clean = R @ np.vstack([cx, cy, cz])

    sigma = 0.08
    noise = sigma * rng.standard_normal(clean.shape)
    noisy = clean + noise
    return theta, noisy[0], noisy[1], noisy[2], clean[0], clean[1], clean[2]


def fit_cyclic(theta: np.ndarray, y: np.ndarray) -> Model | ResponseGeometryModel:
    """Fit a cyclic-in-theta smooth by lifting theta -> (sin, cos) on the unit
    circle and using a 2D thin-plate smooth. Any smooth function of (sin, cos)
    is automatically periodic in theta."""
    data = {
        "ct": np.cos(theta).tolist(),
        "st": np.sin(theta).tolist(),
        "y": y.tolist(),
    }
    return gamfit.fit(data, "y ~ duchon(ct, st)")


def predict_curve(
    models: tuple[Model | ResponseGeometryModel, ...], n_grid: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    payload = {"ct": np.cos(grid).tolist(), "st": np.sin(grid).tolist()}
    out = []
    for m in models:
        pred = m.predict(payload, return_type="dict")
        out.append(np.asarray(pred["mean"], dtype=float))
    return grid, out[0], out[1], out[2]


def main() -> Path:
    theta, nx, ny, nz, tx, ty, tz = make_circle_3d()

    mx = fit_cyclic(theta, nx)
    my = fit_cyclic(theta, ny)
    mz = fit_cyclic(theta, nz)

    _, fx, fy, fz = predict_curve((mx, my, mz))

    # Close the loop visually
    fx = np.append(fx, fx[0])
    fy = np.append(fy, fy[0])
    fz = np.append(fz, fz[0])

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(nx, ny, nz, s=10, color="#9ca3af", alpha=0.6, label="noisy samples")
    # Truth (thin dashed)
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

    # Equal aspect-ish
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

    if "--no-show" not in sys.argv:
        plt.show()
    else:
        plt.close(fig)
    return out


if __name__ == "__main__":
    main()
