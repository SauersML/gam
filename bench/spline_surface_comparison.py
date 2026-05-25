#!/usr/bin/env python3
import typing

import gamfit
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def true_surface(x: typing.Any, y: typing.Any) -> typing.Any:
    return (
        0.80 * np.sin(1.5 * x)
        + 0.60 * np.cos(1.2 * y)
        + 0.50 * x * y
        + 0.35 * np.exp(-((x - 0.30) ** 2 + (y + 0.20) ** 2) / 0.12)
        + 0.25 * np.sin(2.2 * (x + y))
    )


def xy_table(xy: typing.Any, z: typing.Any | None = None) -> dict[str, typing.Any]:
    table: dict[str, typing.Any] = {
        "x": np.asarray(xy[:, 0], dtype=float).tolist(),
        "y": np.asarray(xy[:, 1], dtype=float).tolist(),
    }
    if z is not None:
        table["z"] = np.asarray(z, dtype=float).tolist()
    return table


def gamfit_surface_predict(
    train_xy: typing.Any,
    z_obs: typing.Any,
    eval_xy: typing.Any,
    formula: str,
) -> typing.Any:
    model = gamfit.fit(xy_table(train_xy, z_obs), formula, family="gaussian")
    pred = model.predict(xy_table(eval_xy), return_type="dict")
    return np.asarray(pred["mean"], dtype=float)


def main() -> None:
    rng = np.random.default_rng(42)

    n = 80
    x = rng.uniform(-1.0, 1.0, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)
    xy = np.c_[x, y]

    z_true = true_surface(x, y)
    sigma = 0.05 + 0.22 * (x + 1.0) / 2.0
    z_obs = z_true + rng.normal(0.0, sigma, size=n)

    g = 180
    gx = np.linspace(-1.0, 1.0, g)
    gy = np.linspace(-1.0, 1.0, g)
    xx, yy = np.meshgrid(gx, gy)
    grid_xy = np.c_[xx.ravel(), yy.ravel()]
    zz_true = true_surface(xx, yy)

    centers = 30
    zz_bs = gamfit_surface_predict(
        xy,
        z_obs,
        grid_xy,
        "z ~ te(x, y, knots=8, double_penalty=true)",
    ).reshape(xx.shape)
    zz_tps = gamfit_surface_predict(
        xy,
        z_obs,
        grid_xy,
        f"z ~ thinplate(x, y, centers={centers}, double_penalty=true)",
    ).reshape(xx.shape)
    zz_mat = gamfit_surface_predict(
        xy,
        z_obs,
        grid_xy,
        f"z ~ matern(x, y, centers={centers}, nu=5/2, length_scale=0.55, double_penalty=true)",
    ).reshape(xx.shape)
    zz_du = gamfit_surface_predict(
        xy,
        z_obs,
        grid_xy,
        f"z ~ duchon(x, y, centers={centers}, power=1, order=0, length_scale=0.55)",
    ).reshape(xx.shape)

    vmin = min(np.min(zz_true), np.min(zz_bs), np.min(zz_tps), np.min(zz_mat), np.min(zz_du))
    vmax = max(np.max(zz_true), np.max(zz_bs), np.max(zz_tps), np.max(zz_mat), np.max(zz_du))

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 3, 1, projection="3d"),
        fig.add_subplot(2, 3, 2, projection="3d"),
        fig.add_subplot(2, 3, 3, projection="3d"),
        fig.add_subplot(2, 3, 4, projection="3d"),
        fig.add_subplot(2, 3, 5, projection="3d"),
    ]
    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.axis("off")

    titles = [
        "Ground Truth Surface",
        "Tensor B-Spline (REML + double penalty)",
        "Thin Plate Spline (REML + double penalty)",
        "Matérn (REML + double penalty)",
        "Duchon p=1 (REML)",
    ]
    fields = [zz_true, zz_bs, zz_tps, zz_mat, zz_du]
    shared_cmap = "viridis"
    shared_view = (33, -55)

    for ax, title, zf in zip(axes[:5], titles, fields):
        ax.plot_surface(
            xx,
            yy,
            zf,
            cmap=shared_cmap,
            rcount=110,
            ccount=110,
            linewidth=0.08,
            edgecolor=(1, 1, 1, 0.08),
            antialiased=True,
            alpha=0.96,
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(x, y, z_obs, s=10, c="black", alpha=0.65, depthshade=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=shared_view[0], azim=shared_view[1])
        ax.set_title(title, fontsize=10.5, weight="bold", pad=10)
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.set_zlabel("Height")

    fig.suptitle(
        "3D Surface Comparison (gamfit Gaussian REML)",
        fontsize=14.5,
        weight="bold",
    )

    out = Path(__file__).resolve().parent.parent / "scripts" / "spline_methods_surface_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
