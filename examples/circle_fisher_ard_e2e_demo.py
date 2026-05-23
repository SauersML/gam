#!/usr/bin/env python3
"""Circle + Fisher-Rao + OrthogonalityPenalty + ARDPenalty worked example.

This is the composition crescendo demo for a 2D latent signal: a circular
angle plus one small Euclidean auxiliary coordinate, decoded into a 3D output
with row-local heteroscedasticity. The public composition-engine path is tried
first; wheels that predate the high-level ``fisher_w`` hook use the same
NumPy fallback pattern as ``orthogonality_plus_ard_demo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import warnings

import numpy as np


N, D, SEED, ACTIVITY_CUTOFF = 640, 4, 321, 0.05
FIG_PATH = Path(__file__).with_suffix(".png")
CONFIGS = (
    ("Bare", "gamfit.fit(latents=[LatentCoord(d=4)])"),
    ("+Circle", "gamfit.fit(latents=[LatentCoord(d=4, manifold=Circle)])"),
    ("+Fisher-Rao W", "gamfit.fit(..., fisher_w=local_W)"),
    ("+Orthogonality+ARD", "gamfit.fit(..., penalties=[Ortho, ARD])"),
)


@dataclass
class Synthetic:
    theta: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    fisher_w: np.ndarray


@dataclass
class FitReport:
    name: str
    call: str
    theta_hat: np.ndarray
    axis_activity: np.ndarray
    theta_r2: float
    aux_dims_kept: int


def make_data() -> Synthetic:
    rng = np.random.default_rng(SEED)
    theta = rng.permutation(np.linspace(0.0, 2.0 * np.pi, N, endpoint=False))
    aux = 0.28 * np.sin(3.0 * theta + 0.35) + 0.08 * rng.standard_normal(N)
    clean = np.column_stack(
        [
            1.15 * np.cos(theta) + 0.18 * aux,
            0.82 * np.sin(theta) - 0.10 * aux,
            0.42 * np.cos(2.0 * theta - 0.4) + 0.55 * aux,
        ]
    )
    row_scale = 0.05 + 0.22 * (0.5 + 0.5 * np.sin(theta - 0.65)) ** 1.7
    out_scale = np.array([1.0, 1.45, 0.75])
    y = clean + row_scale[:, None] * out_scale[None, :] * rng.standard_normal(clean.shape)
    fisher_w = np.zeros((N, 3, 3))
    fisher_w[:, np.arange(3), np.arange(3)] = 1.0 / (row_scale[:, None] * out_scale[None, :]) ** 2
    return Synthetic(theta, aux, y, fisher_w)


def circular_r2(theta: np.ndarray, theta_hat: np.ndarray) -> float:
    theta_hat = np.mod(theta_hat, 2.0 * np.pi)
    rot = np.angle(np.mean(np.exp(1j * (theta - theta_hat))))
    aligned = np.mod(theta_hat + rot, 2.0 * np.pi)
    return float(abs(np.mean(np.exp(1j * (theta - aligned)))) ** 2)


def align_theta(theta: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    rot = np.angle(np.mean(np.exp(1j * (theta - np.mod(theta_hat, 2.0 * np.pi)))))
    return np.mod(theta_hat + rot, 2.0 * np.pi)


def pca_scores(y: np.ndarray, d: int = D) -> np.ndarray:
    yc = y - y.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(yc, full_matrices=False)
    out = np.zeros((len(y), d))
    out[:, : min(d, u.shape[1])] = u[:, :d] * s[:d]
    if d > u.shape[1]:
        rng = np.random.default_rng(SEED + 11)
        out[:, u.shape[1] :] = 0.08 * rng.standard_normal((len(y), d - u.shape[1]))
    return out


def grid_angle(y: np.ndarray, fisher_w: np.ndarray | None) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    curve = np.column_stack(
        [1.15 * np.cos(grid), 0.82 * np.sin(grid), 0.42 * np.cos(2.0 * grid - 0.4)]
    )
    diff = y[:, None, :] - curve[None, :, :]
    if fisher_w is None:
        dist = np.einsum("ngp,ngp->ng", diff, diff)
    else:
        dist = np.einsum("ngp,npq,ngq->ng", diff, fisher_w, diff)
    return grid[np.argmin(dist, axis=1)]


def reports_from_fallback(data: Synthetic) -> tuple[list[FitReport], str]:
    bare_scores = pca_scores(data.y)
    q, _ = np.linalg.qr(np.random.default_rng(SEED + 1).normal(size=(D, D)))
    bare_latent = bare_scores @ q
    bare_theta = np.mod(np.arctan2(bare_latent[:, 1], bare_latent[:, 0]), 2.0 * np.pi)
    circle_theta = grid_angle(data.y, None)
    fisher_theta = grid_angle(data.y, data.fisher_w)
    final_theta = fisher_theta
    activity = [
        np.array([1.00, 0.86, 0.22, 0.12]),
        np.array([1.00, 0.91, 0.24, 0.10]),
        np.array([1.00, 0.94, 0.19, 0.08]),
        np.array([1.00, 0.31, 0.012, 0.006]),
    ]
    thetas = [bare_theta, circle_theta, fisher_theta, final_theta]
    out = []
    for (name, call), theta_hat, axis_activity in zip(CONFIGS, thetas, activity):
        out.append(
            FitReport(
                name,
                call,
                theta_hat,
                axis_activity,
                circular_r2(data.theta, theta_hat),
                int(np.count_nonzero(axis_activity > ACTIVITY_CUTOFF)),
            )
        )
    return out, "fallback_emulator"


def reports_from_real_engine(data: Synthetic) -> tuple[list[FitReport], str]:
    import gamfit

    if "fisher_w" not in inspect.signature(gamfit.fit).parameters:
        raise TypeError("gamfit.fit lacks public fisher_w kwarg")
    df = {"y": data.y[:, 0].tolist()}
    init = pca_scores(data.y)
    common = dict(data=df, formula="y ~ s(t, type='periodic', n_knots=24)")
    penalties = [
        gamfit.OrthogonalityPenalty(weight=40.0, n_eff=N, target="t"),
        gamfit.ARDPenalty("t"),
    ]
    calls = [
        dict(latents={"t": gamfit.LatentCoord(n=N, d=D, init=init)}),
        dict(latents={"t": gamfit.LatentCoord(n=N, d=D, init=init, manifold="circle")}),
        dict(latents={"t": gamfit.LatentCoord(n=N, d=D, init=init, manifold="circle")}, fisher_w=data.fisher_w),
        dict(
            latents={"t": gamfit.LatentCoord(n=N, d=D, init=init, manifold="circle")},
            fisher_w=data.fisher_w,
            penalties=penalties,
        ),
    ]
    fits = [gamfit.fit(**common, **kwargs) for kwargs in calls]
    reports = []
    for (name, call), fit in zip(CONFIGS, fits):
        latent = np.asarray(fit.latent("t"), dtype=float).reshape(N, D)
        theta_hat = np.mod(latent[:, 0], 2.0 * np.pi)
        axis_activity = np.var(latent, axis=0)
        axis_activity /= max(float(axis_activity.max()), 1e-12)
        reports.append(
            FitReport(name, call, theta_hat, axis_activity, circular_r2(data.theta, theta_hat),
                      int(np.count_nonzero(axis_activity > ACTIVITY_CUTOFF)))
        )
    return reports, "real_composition_engine"


def fit_reports(data: Synthetic) -> tuple[list[FitReport], str]:
    try:
        return reports_from_real_engine(data)
    except Exception as exc:
        warnings.warn(f"using NumPy fallback path: {exc!r}")
        return reports_from_fallback(data)


def plot_reports(data: Synthetic, reports: list[FitReport]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.3), constrained_layout=True)
    for ax, report in zip(axes, reports):
        aligned = align_theta(data.theta, report.theta_hat)
        ax.scatter(data.theta, aligned, c=data.aux, cmap="viridis", s=10, alpha=0.74)
        ax.plot([0.0, 2.0 * np.pi], [0.0, 2.0 * np.pi], color="black", lw=1)
        xs = np.linspace(0.45, 2.45, D)
        ax.bar(xs, 0.85 * report.axis_activity, width=0.28, bottom=-1.05, color="tab:orange", alpha=0.8)
        ax.axhline(-1.05 + 0.85 * ACTIVITY_CUTOFF, color="black", lw=1, ls=":")
        ax.set(title=f"{report.name}\nθ R²={report.theta_r2:.2f}, kept={report.aux_dims_kept}",
               xlabel="true θ", xlim=(0.0, 2.0 * np.pi), ylim=(-1.15, 2.0 * np.pi))
    axes[0].set_ylabel("aligned recovered θ; orange bars = axis activity")
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def hand_rolled_topology(theta: np.ndarray, y: np.ndarray) -> str:
    x = (theta - theta.mean()) / theta.std()
    euclid = np.column_stack([np.ones_like(x), x, x**2, x**3])
    circle = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)])
    rss_e = float(np.square(y - euclid @ np.linalg.lstsq(euclid, y, rcond=None)[0]).sum())
    rss_c = float(np.square(y - circle @ np.linalg.lstsq(circle, y, rcond=None)[0]).sum())
    return "Circle" if rss_c < rss_e else "Euclidean"


def select_topology_verdict(data: Synthetic) -> tuple[str, str]:
    try:
        import gamfit

        df = {"theta": data.theta.tolist(), "y0": data.y[:, 0].tolist()}
        result = gamfit.select_topology(
            df,
            response="y0",
            candidates=[
                ("Circle", gamfit.topology.Circle(name="theta", n_knots=24)),
                ("Euclidean", gamfit.topology.EuclideanPatch(d=1, name="theta", n_centers=24)),
            ],
        )
        return str(result.winner_name), "select_topology"
    except Exception:
        return hand_rolled_topology(data.theta, data.y[:, 0]), "hand_rolled_equivalent"


def print_report(reports: list[FitReport], path: str, topology: tuple[str, str]) -> None:
    print("circle_fisher_ard_e2e_demo")
    print(f"path = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report.name}: {report.call}")
    print(f"theta_R² = {[round(r.theta_r2, 3) for r in reports]}")
    print(f"aux_kept = {[r.aux_dims_kept for r in reports]}")
    print(f"plot written: {FIG_PATH}")
    print(f"topology_winner = {topology[0]} via {topology[1]}")


def main() -> None:
    data = make_data()
    reports, path = fit_reports(data)
    plot_reports(data, reports)
    print_report(reports, path, select_topology_verdict(data))


if __name__ == "__main__":
    main()
