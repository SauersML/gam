"""Render two hero figures from real `gamfit` fits.

Outputs (both written into docs/images/):
- surface_fit_hero.png : data → fitted mean → predictive SE for one smooth.
- smooth_zoo.png       : same data fit with four smooth families side-by-side
                         (thin-plate, Matérn, Duchon, tensor product).

Run from anywhere; paths are resolved relative to this file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gamfit


# ---------------------------------------------------------------------------
# Editorial style — thin spines, white background, modest fonts
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#1f2937",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#374151",
        "axes.titlecolor": "#0f172a",
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
        "font.family": [
            "Inter",
            "SF Pro Display",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "sans-serif",
        ],
    }
)

DOCS_IMAGES = Path(__file__).resolve().parents[2] / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)


def truth(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """A smooth, fittable 2-D field: two gaussian bumps + a gentle slope."""
    bump_a = 1.2 * np.exp(-((x1 - 0.3) ** 2 + (x2 - 0.65) ** 2) / 0.07)
    bump_b = -0.8 * np.exp(-((x1 - 0.72) ** 2 + (x2 - 0.28) ** 2) / 0.05)
    slope = 0.4 * (x1 - 0.5) + 0.2 * (x2 - 0.5)
    return bump_a + bump_b + slope


def make_data(n: int = 800, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = truth(x1, x2) + 0.18 * rng.standard_normal(n)
    return {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}


def grid_predict(model: gamfit.Model, side: int = 120, *, with_se: bool = False):
    g = np.linspace(0.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)
    payload = {"x1": gx.ravel().tolist(), "x2": gy.ravel().tolist()}
    pred = model.predict(payload, interval=0.95 if with_se else None, return_type="dict")
    mean = np.asarray(pred["mean"], dtype=float).reshape(side, side)
    se = (
        np.asarray(pred["effective_se"], dtype=float).reshape(side, side)
        if with_se and "effective_se" in pred
        else None
    )
    return gx, gy, mean, se


def style_colorbar(cbar) -> None:
    cbar.outline.set_linewidth(0.3)
    cbar.outline.set_edgecolor("#9ca3af")
    cbar.ax.tick_params(width=0.4, length=2.5, labelsize=7.5, color="#9ca3af")


def style_square_axes(ax, *, xlabel: str | None, ylabel: str | None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Figure 1: data → fitted mean → predictive SE
# ---------------------------------------------------------------------------
def render_hero() -> Path:
    data = make_data()
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])
    y = np.asarray(data["y"])

    model = gamfit.fit(data, "y ~ matern(x1, x2)", response_geometry=None)
    gx, gy, mean, se = grid_predict(model, side=140, with_se=True)

    # shared color scale for the data scatter and the fitted mean
    vmin = float(min(y.min(), mean.min()))
    vmax = float(max(y.max(), mean.max()))

    fig, axes = plt.subplots(
        1, 3, figsize=(11.6, 4.0), gridspec_kw={"wspace": 0.18}
    )

    ax = axes[0]
    sc = ax.scatter(
        x1, x2, c=y, cmap="magma", s=18,
        edgecolor="white", linewidth=0.35,
        vmin=vmin, vmax=vmax,
    )
    ax.set_title("Noisy observations", pad=10)
    style_square_axes(ax, xlabel="x₁", ylabel="x₂")
    style_colorbar(fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[1]
    cf = ax.contourf(gx, gy, mean, levels=24, cmap="magma", vmin=vmin, vmax=vmax)
    ax.contour(gx, gy, mean, levels=8, colors="white", linewidths=0.4, alpha=0.55)
    ax.set_title("matern(x₁, x₂) — fitted mean", pad=10)
    style_square_axes(ax, xlabel="x₁", ylabel=None)
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[2]
    cf = ax.contourf(gx, gy, se if se is not None else np.zeros_like(mean),
                     levels=24, cmap="viridis")
    ax.scatter(x1, x2, s=2.5, color="white", alpha=0.5, linewidth=0)
    ax.set_title("Posterior predictive SE", pad=10)
    style_square_axes(ax, xlabel="x₁", ylabel=None)
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    fig.text(
        0.5, -0.03,
        "800 noisy observations  ·  Matérn ν = 5/2 surface fit  ·  smoothing chosen by REML",
        ha="center", va="center", fontsize=8.5, color="#6b7280",
    )

    out = DOCS_IMAGES / "surface_fit_hero.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: four smooth families side by side on the same data
# ---------------------------------------------------------------------------
def render_zoo() -> Path:
    data = make_data()
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])

    specs = [
        ("thin-plate", "y ~ thinplate(x1, x2)"),
        ("Matérn",     "y ~ matern(x1, x2)"),
        ("Duchon",     "y ~ duchon(x1, x2, centers=40)"),
        ("tensor",     "y ~ te(x1, x2)"),
    ]

    side = 110
    surfaces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, formula in specs:
        m = gamfit.fit(data, formula, response_geometry=None)
        gx, gy, mean, _ = grid_predict(m, side=side, with_se=False)
        surfaces.append((name, gx, gy, mean))

    vmin = min(np.percentile(m, 2) for _, _, _, m in surfaces)
    vmax = max(np.percentile(m, 98) for _, _, _, m in surfaces)

    fig, axes = plt.subplots(
        1, 4, figsize=(13.4, 3.7), gridspec_kw={"wspace": 0.10}
    )
    for ax, (name, gx, gy, mean) in zip(axes, surfaces):
        cf = ax.contourf(gx, gy, mean, levels=22, cmap="magma",
                         vmin=vmin, vmax=vmax)
        ax.contour(gx, gy, mean, levels=6, colors="white", linewidths=0.35, alpha=0.55)
        ax.scatter(x1, x2, s=1.6, color="white", alpha=0.35, linewidth=0)
        ax.set_title(name, pad=8)
        style_square_axes(
            ax,
            xlabel="x₁",
            ylabel="x₂" if ax is axes[0] else None,
        )

    cbar = fig.colorbar(cf, ax=axes.tolist(), fraction=0.018, pad=0.02)
    style_colorbar(cbar)

    fig.text(
        0.5, -0.02,
        "Same 800-point dataset, four smooth families.  Smoothing parameters selected by REML in each case.",
        ha="center", va="center", fontsize=8.5, color="#6b7280",
    )

    out = DOCS_IMAGES / "smooth_zoo.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    render_hero()
    render_zoo()


if __name__ == "__main__":
    main()
