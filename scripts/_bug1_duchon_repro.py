"""Bug #1 repro: Duchon centers=80 on (cos t, sin t) lift produces grossly
oversmoothed / wandering predictions compared to centers=26 (default) or
thinplate/matern on identical data.

Vary centers, report RMSE of predicted z vs truth on a dense grid.
"""
from __future__ import annotations

import numpy as np
import gamfit


def make_circle_3d(n=260, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    d = np.arctan2(np.sin(theta - 2.0 * np.pi / 3.0),
                   np.cos(theta - 2.0 * np.pi / 3.0))
    r = 1.0 + 0.55 * np.exp(-0.5 * (d / 0.16) ** 2)
    cx, cy, cz = r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)
    tilt = np.deg2rad(30.0)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(tilt), -np.sin(tilt)],
                  [0.0, np.sin(tilt),  np.cos(tilt)]])
    clean = R @ np.vstack([cx, cy, cz])
    sigma = 0.07
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    return theta, clean.T, noisy.T


def fit_eval(formula_body, theta, P_noisy, P_truth, grid):
    """Fit z = f(ct, st), evaluate on dense grid, return RMSE vs truth."""
    data = {
        "ct": np.cos(theta).tolist(),
        "st": np.sin(theta).tolist(),
        "y": P_noisy[:, 2].tolist(),
    }
    m = gamfit.fit(data, f"y ~ {formula_body}")
    pred = m.predict(
        {"ct": np.cos(grid).tolist(), "st": np.sin(grid).tolist()},
        return_type="dict",
    )
    yhat = np.asarray(pred["mean"], dtype=float)

    # Truth on grid: z coord of tilted unit circle (no spike for fairness in test)
    # (here we evaluate against the *clean* generator at the SAME grid points)
    # Reuse the generator formula
    d = np.arctan2(np.sin(grid - 2.0 * np.pi / 3.0),
                   np.cos(grid - 2.0 * np.pi / 3.0))
    r = 1.0 + 0.55 * np.exp(-0.5 * (d / 0.16) ** 2)
    cx_t, cy_t, cz_t = r * np.cos(grid), r * np.sin(grid), np.zeros_like(grid)
    tilt = np.deg2rad(30.0)
    Rmat = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(tilt), -np.sin(tilt)],
                     [0.0, np.sin(tilt),  np.cos(tilt)]])
    truth = (Rmat @ np.vstack([cx_t, cy_t, cz_t])).T
    z_true = truth[:, 2]

    rmse = float(np.sqrt(np.mean((yhat - z_true) ** 2)))
    span = float(yhat.max() - yhat.min())
    return rmse, span, yhat


def main():
    theta, P_clean, P_noisy = make_circle_3d()
    grid = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)

    cases = [
        ("duchon(ct, st)",                   "duchon default"),
        ("duchon(ct, st, centers=20)",       "duchon 20"),
        ("duchon(ct, st, centers=40)",       "duchon 40"),
        ("duchon(ct, st, centers=60)",       "duchon 60"),
        ("duchon(ct, st, centers=80)",       "duchon 80"),
        ("duchon(ct, st, centers=100)",      "duchon 100"),
        ("thinplate(ct, st)",                "thinplate"),
        ("matern(ct, st)",                   "matern"),
    ]
    print(f"{'case':30s}  {'rmse(z)':>10s}  {'pred span':>10s}")
    for body, label in cases:
        try:
            rmse, span, _ = fit_eval(body, theta, P_noisy, P_clean, grid)
            print(f"{label:30s}  {rmse:10.4f}  {span:10.4f}")
        except Exception as exc:
            print(f"{label:30s}  ERROR: {exc}")


if __name__ == "__main__":
    main()
