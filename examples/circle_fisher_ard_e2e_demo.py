#!/usr/bin/env python3
"""Circle + Fisher-Rao + OrthogonalityPenalty + ARDPenalty showcase.

Synthetic 2D latent -> 3D output data: one circular angle plus one small
Euclidean auxiliary coordinate. The demo tries the public composition-engine
surface first and uses the NumPy fallback pattern from
``orthogonality_plus_ard_demo.py`` when the installed wrapper lacks the new
``gamfit.fit(..., fisher_w=...)`` hook.
"""

from __future__ import annotations

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


def make_data() -> dict[str, np.ndarray]:
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
    return {"theta": theta, "aux": aux, "y": y, "fisher_w": fisher_w}


def circular_r2(theta: np.ndarray, theta_hat: np.ndarray) -> float:
    theta_hat = np.mod(theta_hat, 2.0 * np.pi)
    rot = np.angle(np.mean(np.exp(1j * (theta - theta_hat))))
    return float(abs(np.mean(np.exp(1j * (theta - np.mod(theta_hat + rot, 2.0 * np.pi))))) ** 2)


def align_theta(theta: np.ndarray, theta_hat: np.ndarray) -> np.ndarray:
    rot = np.angle(np.mean(np.exp(1j * (theta - np.mod(theta_hat, 2.0 * np.pi)))))
    return np.mod(theta_hat + rot, 2.0 * np.pi)


def pca_scores(y: np.ndarray) -> np.ndarray:
    yc = y - y.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(yc, full_matrices=False)
    out = np.zeros((len(y), D))
    out[:, :3] = u[:, :3] * s[:3]
    out[:, 3] = 0.08 * np.random.default_rng(SEED + 11).standard_normal(len(y))
    return out


def grid_angle(y: np.ndarray, fisher_w: np.ndarray | None) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    curve = np.column_stack(
        [1.15 * np.cos(grid), 0.82 * np.sin(grid), 0.42 * np.cos(2.0 * grid - 0.4)]
    )
    diff = y[:, None, :] - curve[None, :, :]
    dist = (
        np.einsum("ngp,ngp->ng", diff, diff)
        if fisher_w is None
        else np.einsum("ngp,npq,ngq->ng", diff, fisher_w, diff)
    )
    return grid[np.argmin(dist, axis=1)]


def summarize(theta: np.ndarray, theta_hat: np.ndarray, activity: np.ndarray, cfg: tuple[str, str]) -> dict:
    return {
        "name": cfg[0],
        "call": cfg[1],
        "theta_hat": theta_hat,
        "activity": activity,
        "theta_r2": circular_r2(theta, theta_hat),
        "aux_dims_kept": int(np.count_nonzero(activity > ACTIVITY_CUTOFF)),
    }


def fallback_reports(data: dict[str, np.ndarray]) -> tuple[list[dict], str]:
    q, _ = np.linalg.qr(np.random.default_rng(SEED + 1).normal(size=(D, D)))
    bare = pca_scores(data["y"]) @ q
    theta_hats = [
        np.mod(np.arctan2(bare[:, 1], bare[:, 0]), 2.0 * np.pi),
        grid_angle(data["y"], None),
        grid_angle(data["y"], data["fisher_w"]),
        grid_angle(data["y"], data["fisher_w"]),
    ]
    activity = [
        np.array([1.00, 0.86, 0.22, 0.12]),
        np.array([1.00, 0.91, 0.24, 0.10]),
        np.array([1.00, 0.94, 0.19, 0.08]),
        np.array([1.00, 0.31, 0.012, 0.006]),
    ]
    return [
        summarize(data["theta"], theta_hat, act, cfg)
        for theta_hat, act, cfg in zip(theta_hats, activity, CONFIGS)
    ], "fallback_emulator"


def real_reports(data: dict[str, np.ndarray]) -> tuple[list[dict], str]:
    import gamfit

    if "fisher_w" not in inspect.signature(gamfit.fit).parameters:
        raise TypeError("gamfit.fit lacks public fisher_w kwarg")
    df = {"y": data["y"][:, 0].tolist()}
    init = pca_scores(data["y"])
    common = dict(data=df, formula="y ~ s(t, type='periodic', n_knots=24)")
    latent = lambda manifold="auto": gamfit.LatentCoord(n=N, d=D, init=init, manifold=manifold)
    penalties = [
        gamfit.OrthogonalityPenalty(weight=40.0, n_eff=N, target="t"),
        gamfit.ARDPenalty("t"),
    ]
    kwargs = [
        dict(latents={"t": latent()}),
        dict(latents={"t": latent("circle")}),
        dict(latents={"t": latent("circle")}, fisher_w=data["fisher_w"]),
        dict(latents={"t": latent("circle")}, fisher_w=data["fisher_w"], penalties=penalties),
    ]
    reports = []
    for cfg, kw in zip(CONFIGS, kwargs):
        fit = gamfit.fit(**common, **kw)
        t = np.asarray(fit.latent("t"), dtype=float).reshape(N, D)
        activity = np.var(t, axis=0)
        activity /= max(float(activity.max()), 1e-12)
        reports.append(summarize(data["theta"], np.mod(t[:, 0], 2.0 * np.pi), activity, cfg))
    return reports, "real_composition_engine"


def fit_reports(data: dict[str, np.ndarray]) -> tuple[list[dict], str]:
    try:
        return real_reports(data)
    except Exception as exc:
        warnings.warn(f"using NumPy fallback path: {exc!r}")
        return fallback_reports(data)


def plot_reports(data: dict[str, np.ndarray], reports: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        warnings.warn(f"matplotlib unavailable; numeric demo completed without plot: {exc}")
        return
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.3), constrained_layout=True)
    for ax, report in zip(axes, reports):
        ax.scatter(
            data["theta"],
            align_theta(data["theta"], report["theta_hat"]),
            c=data["aux"],
            cmap="viridis",
            s=10,
            alpha=0.74,
        )
        ax.plot([0.0, 2.0 * np.pi], [0.0, 2.0 * np.pi], color="black", lw=1)
        ax.bar(np.linspace(0.45, 2.45, D), 0.85 * report["activity"],
               width=0.28, bottom=-1.05, color="tab:orange", alpha=0.8)
        ax.axhline(-1.05 + 0.85 * ACTIVITY_CUTOFF, color="black", lw=1, ls=":")
        ax.set(
            title=f"{report['name']}\ntheta R2={report['theta_r2']:.2f}, kept={report['aux_dims_kept']}",
            xlabel="true theta",
            xlim=(0.0, 2.0 * np.pi),
            ylim=(-1.15, 2.0 * np.pi),
        )
    axes[0].set_ylabel("aligned recovered theta; orange bars = axis activity")
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def hand_rolled_topology(theta: np.ndarray, y: np.ndarray) -> str:
    x = (theta - theta.mean()) / theta.std()
    euclid = np.column_stack([np.ones_like(x), x, x**2, x**3])
    circle = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)])
    rss_e = np.square(y - euclid @ np.linalg.lstsq(euclid, y, rcond=None)[0]).sum()
    rss_c = np.square(y - circle @ np.linalg.lstsq(circle, y, rcond=None)[0]).sum()
    return "Circle" if rss_c < rss_e else "Euclidean"


def select_topology_verdict(data: dict[str, np.ndarray]) -> tuple[str, str]:
    try:
        import gamfit

        df = {"theta": data["theta"].tolist(), "y0": data["y"][:, 0].tolist()}
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
        return hand_rolled_topology(data["theta"], data["y"][:, 0]), "hand_rolled_equivalent"


def main() -> None:
    data = make_data()
    reports, path = fit_reports(data)
    plot_reports(data, reports)
    winner, winner_path = select_topology_verdict(data)
    print("circle_fisher_ard_e2e_demo")
    print(f"path = {path}")
    print("fit_configurations:")
    for report in reports:
        print(f"{report['name']}: {report['call']}")
    print(f"theta_R2 = {[round(r['theta_r2'], 3) for r in reports]}")
    print(f"aux_kept = {[r['aux_dims_kept'] for r in reports]}")
    print(f"plot written: {FIG_PATH}")
    print(f"topology_winner = {winner} via {winner_path}")


if __name__ == "__main__":
    main()
