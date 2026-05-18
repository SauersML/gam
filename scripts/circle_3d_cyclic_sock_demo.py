"""Noisy 3D circle with a spike, fit by several cyclic GAM families, each
wrapped in a smooth tube ('sock') that contains the data."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gamfit
from gamfit import Model, ResponseGeometryModel


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_circle_3d(n: int = 260, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    # Outward Gaussian spike near 2*pi/3
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


# ---------------------------------------------------------------------------
# Fitting one closed curve via three cyclic 1D GAMs (lifted to (cos,sin))
# ---------------------------------------------------------------------------
def fit_closed_curve(
    theta: np.ndarray, P: np.ndarray, formula_body: str
) -> list[Model | ResponseGeometryModel]:
    """Fit one cyclic GAM per coordinate by lifting theta -> (cos, sin).

    formula_body is e.g. 'duchon(ct, st, centers=80)'."""
    base = {"ct": np.cos(theta).tolist(), "st": np.sin(theta).tolist()}
    models: list[Model | ResponseGeometryModel] = []
    for j in range(3):
        data = {**base, "y": P[:, j].tolist()}
        models.append(gamfit.fit(data, f"y ~ {formula_body}"))
    return models


def predict_curve(
    models: list[Model | ResponseGeometryModel], grid: np.ndarray
) -> np.ndarray:
    payload = {"ct": np.cos(grid).tolist(), "st": np.sin(grid).tolist()}
    cols = []
    for m in models:
        pred = m.predict(payload, return_type="dict")
        cols.append(np.asarray(pred["mean"], dtype=float))
    return np.vstack(cols).T  # (K, 3)


# ---------------------------------------------------------------------------
# Sock: smooth varying-radius tube enclosing the data around the fitted curve
# ---------------------------------------------------------------------------
def bishop_frame(curve: np.ndarray):
    """Parallel-transport (Bishop) frame around a closed curve. Returns
    unit tangent T, and two unit perpendiculars N, B at each sample."""
    K = curve.shape[0]
    # Centered finite differences (cyclic) for the tangent
    fwd = np.roll(curve, -1, axis=0)
    bwd = np.roll(curve, 1, axis=0)
    T = fwd - bwd
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    # Pick a starting perpendicular not parallel to T[0]
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, T[0])) > 0.95:
        seed = np.array([0.0, 1.0, 0.0])
    n0 = seed - np.dot(seed, T[0]) * T[0]
    n0 /= np.linalg.norm(n0)

    N = np.empty_like(curve)
    N[0] = n0
    for k in range(1, K):
        # Parallel-transport N[k-1] from T[k-1] to T[k] by Rodrigues
        t0, t1 = T[k - 1], T[k]
        axis = np.cross(t0, t1)
        s = np.linalg.norm(axis)
        c = np.dot(t0, t1)
        if s < 1e-12:
            N[k] = N[k - 1]
            continue
        axis /= s
        v = N[k - 1]
        N[k] = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
        N[k] /= np.linalg.norm(N[k])
    B = np.cross(T, N)
    return T, N, B


def sock_radius(curve: np.ndarray, P: np.ndarray, win: int = 9) -> np.ndarray:
    """Tight radius r(theta) at each curve sample so that the tube contains
    every data point within an angular neighborhood. Smoothed cyclically."""
    K = curve.shape[0]
    # Nearest curve index for each data point
    # (K=400, n~260 → small enough for direct pairwise)
    d2 = ((P[:, None, :] - curve[None, :, :]) ** 2).sum(axis=2)
    near = d2.argmin(axis=1)
    dist = np.sqrt(d2.min(axis=1))

    # Per-bin max
    raw = np.zeros(K)
    for k, dk in zip(near, dist):
        if dk > raw[k]:
            raw[k] = dk
    # Cyclic max-filter, then a light cyclic smoother
    pad = np.concatenate([raw[-win:], raw, raw[:win]])
    rolled = np.lib.stride_tricks.sliding_window_view(pad, win)[: K]
    env = rolled.max(axis=1)
    # Small cyclic Gaussian smooth
    sigma = 4.0
    width = int(4 * sigma)
    k = np.arange(-width, width + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    env_padded = np.concatenate([env[-width:], env, env[:width]])
    smooth = np.convolve(env_padded, g, mode="valid")
    # Floor at a small value so it's not razor-thin where data is sparse
    return np.maximum(smooth, 0.04)


def build_tube(curve: np.ndarray, N: np.ndarray, B: np.ndarray,
               r: np.ndarray, n_phi: int = 60):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    cphi, sphi = np.cos(phi), np.sin(phi)
    # X[k, j] = curve[k] + r[k] * (cos phi[j] * N[k] + sin phi[j] * B[k])
    offset = (cphi[None, :, None] * N[:, None, :]
              + sphi[None, :, None] * B[:, None, :])
    pts = curve[:, None, :] + r[:, None, None] * offset
    return pts[..., 0], pts[..., 1], pts[..., 2]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main() -> Path:
    theta, P = make_circle_3d()

    specs = [
        ("Duchon (centers=80)", "duchon(ct, st, centers=80)"),
        ("Thin-plate",          "thinplate(ct, st)"),
        ("Matérn",              "matern(ct, st)"),
        ("Tensor",              "tensor(ct, st)"),
    ]

    grid = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)

    fits = []
    for name, body in specs:
        models = fit_closed_curve(theta, P, body)
        curve = predict_curve(models, grid)
        curve_closed = np.vstack([curve, curve[0]])
        T, N, B = bishop_frame(curve)
        r = sock_radius(curve, P)
        Xs, Ys, Zs = build_tube(curve, N, B, r)
        # Close tube
        Xs = np.vstack([Xs, Xs[0]])
        Ys = np.vstack([Ys, Ys[0]])
        Zs = np.vstack([Zs, Zs[0]])
        fits.append((name, curve_closed, (Xs, Ys, Zs)))

    fig = plt.figure(figsize=(13.5, 11))
    for i, (name, curve, (Xs, Ys, Zs)) in enumerate(fits):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                   s=8, color="#374151", alpha=0.55, label="data")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                color="#dc2626", linewidth=2.0, label="fit")
        ax.plot_surface(Xs, Ys, Zs,
                        color="#3b82f6", alpha=0.18,
                        linewidth=0, antialiased=True, shade=True)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Equal-ish bounds
        for setter, vals in zip(
            (ax.set_xlim, ax.set_ylim, ax.set_zlim),
            (np.concatenate([P[:, 0], Xs.ravel()]),
             np.concatenate([P[:, 1], Ys.ravel()]),
             np.concatenate([P[:, 2], Zs.ravel()])),
        ):
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.08 * (hi - lo)
            setter(lo - pad, hi + pad)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Cyclic GAM fits with smooth enclosing tube ('sock')",
                 fontsize=13)
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "circle_3d_cyclic_sock_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    if "--no-show" not in sys.argv:
        plt.show()
    else:
        plt.close(fig)
    return out


if __name__ == "__main__":
    main()
