"""Render the full README figure portfolio from real `gamfit` fits.

Outputs (written to docs/images/):

  surface_fit_hero.png       2D, 3 panels: data → fitted mean → predictive SE
  smooth_zoo.png             2D, 4 panels: thinplate / matern / duchon / te
  surface_3d_shaded.png      3D shaded surface of the matern fit
  surface_3d_wireframe.png   3D wireframe over the noisy point cloud
  surface_3d_compare.png     3D side-by-side: matern vs duchon
  marginal_slope_3d.png      The headline 2-surface marginal-slope viz

Everything uses real fits. Kick this off once, walk away, and pick up
the PNGs in docs/images/ when it's done.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib import cm

import gamfit


# ---------------------------------------------------------------------------
# Editorial style
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
        "savefig.dpi": 200,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truth_2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Two-bump 2-D field used as the regression target."""
    bump_a = 1.2 * np.exp(-((x1 - 0.30) ** 2 + (x2 - 0.65) ** 2) / 0.07)
    bump_b = -0.8 * np.exp(-((x1 - 0.72) ** 2 + (x2 - 0.28) ** 2) / 0.05)
    slope = 0.4 * (x1 - 0.5) + 0.2 * (x2 - 0.5)
    return bump_a + bump_b + slope


def make_regression(n: int = 800, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = truth_2d(x1, x2) + 0.18 * rng.standard_normal(n)
    return {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}


def grid_eval(model: gamfit.Model, side: int = 110, *, with_se: bool = False):
    g = np.linspace(0.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)
    payload = {"x1": gx.ravel().tolist(), "x2": gy.ravel().tolist()}
    pred = model.predict(
        payload, interval=0.95 if with_se else None, return_type="dict"
    )
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


def style_square_axes_2d(ax, *, xlabel, ylabel) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])


def style_3d_axes(ax, *, xlabel, ylabel, zlabel) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_zlabel(zlabel, labelpad=4)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#d1d5db")
        axis.pane.set_linewidth(0.4)
    ax.tick_params(axis="x", pad=0)
    ax.tick_params(axis="y", pad=0)
    ax.tick_params(axis="z", pad=2)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure 1 — 2D three-panel hero (data → mean → SE)
# ---------------------------------------------------------------------------
def render_2d_hero(data: dict) -> None:
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])
    y = np.asarray(data["y"])

    model = gamfit.fit(data, "y ~ matern(x1, x2)")
    gx, gy, mean, se = grid_eval(model, side=140, with_se=True)

    vmin = float(min(y.min(), mean.min()))
    vmax = float(max(y.max(), mean.max()))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0),
                             gridspec_kw={"wspace": 0.18})

    ax = axes[0]
    sc = ax.scatter(x1, x2, c=y, cmap="magma", s=18,
                    edgecolor="white", linewidth=0.35,
                    vmin=vmin, vmax=vmax)
    ax.set_title("Noisy observations", pad=10)
    style_square_axes_2d(ax, xlabel="x₁", ylabel="x₂")
    style_colorbar(fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[1]
    cf = ax.contourf(gx, gy, mean, levels=24, cmap="magma",
                     vmin=vmin, vmax=vmax)
    ax.contour(gx, gy, mean, levels=8, colors="white",
               linewidths=0.4, alpha=0.55)
    ax.set_title("matern(x₁, x₂) — fitted mean", pad=10)
    style_square_axes_2d(ax, xlabel="x₁", ylabel=None)
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[2]
    cf = ax.contourf(gx, gy, se if se is not None else np.zeros_like(mean),
                     levels=24, cmap="viridis")
    ax.scatter(x1, x2, s=2.5, color="white", alpha=0.5, linewidth=0)
    ax.set_title("Posterior predictive SE", pad=10)
    style_square_axes_2d(ax, xlabel="x₁", ylabel=None)
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    fig.text(0.5, -0.03,
             "800 noisy observations  ·  Matérn ν = 5/2 surface fit"
             "  ·  smoothing chosen by REML",
             ha="center", va="center", fontsize=8.5, color="#6b7280")

    out = DOCS_IMAGES / "surface_fit_hero.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    return mean, gx, gy, model


# ---------------------------------------------------------------------------
# Figure 2 — 2D zoo (4 smooth families on the same data)
# ---------------------------------------------------------------------------
def render_zoo(data: dict) -> dict[str, np.ndarray]:
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
        m = gamfit.fit(data, formula)
        gx, gy, mean, _ = grid_eval(m, side=side, with_se=False)
        surfaces.append((name, gx, gy, mean))

    vmin = min(float(np.percentile(s, 2)) for _, _, _, s in surfaces)
    vmax = max(float(np.percentile(s, 98)) for _, _, _, s in surfaces)

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7),
                             gridspec_kw={"wspace": 0.10})
    for ax, (name, gx, gy, mean) in zip(axes, surfaces):
        cf = ax.contourf(gx, gy, mean, levels=22, cmap="magma",
                         vmin=vmin, vmax=vmax)
        ax.contour(gx, gy, mean, levels=6, colors="white",
                   linewidths=0.35, alpha=0.55)
        ax.scatter(x1, x2, s=1.6, color="white", alpha=0.35, linewidth=0)
        ax.set_title(name, pad=8)
        style_square_axes_2d(
            ax, xlabel="x₁",
            ylabel="x₂" if ax is axes[0] else None,
        )

    cbar = fig.colorbar(cf, ax=axes.tolist(), fraction=0.018, pad=0.02)
    style_colorbar(cbar)

    fig.text(0.5, -0.02,
             "Same 800-point dataset, four smooth families.  "
             "Smoothing parameters selected by REML in each case.",
             ha="center", va="center", fontsize=8.5, color="#6b7280")

    out = DOCS_IMAGES / "smooth_zoo.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"wrote {out}")
    return {name: surf for name, _, _, surf in surfaces}, surfaces


# ---------------------------------------------------------------------------
# Figure 3 — 3D shaded surface
# ---------------------------------------------------------------------------
def render_3d_shaded(gx, gy, mean) -> None:
    fig = plt.figure(figsize=(8.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(mean, cmap=cm.magma,
                   vert_exag=0.6, blend_mode="soft")
    ax.plot_surface(
        gx, gy, mean,
        facecolors=rgb,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False,
    )
    ax.contour(gx, gy, mean, zdir="z",
               offset=float(np.nanmin(mean)) - 0.05,
               levels=12, cmap="magma", linewidths=0.6, alpha=0.7)

    ax.set_title("matern(x₁, x₂) — 3-D fitted surface",
                 pad=2, loc="left")
    ax.set_zlim(float(np.nanmin(mean)) - 0.05, float(np.nanmax(mean)) + 0.1)
    style_3d_axes(ax, xlabel="x₁", ylabel="x₂", zlabel="ŷ")
    ax.view_init(elev=28, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.55))

    fig.text(0.5, 0.02,
             "Same surface as the 2-D panel, shaded with a 315° / 45° "
             "light source for relief.",
             ha="center", fontsize=8.5, color="#6b7280")

    out = DOCS_IMAGES / "surface_3d_shaded.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 — 3D wireframe over scatter
# ---------------------------------------------------------------------------
def render_3d_wireframe(data: dict, gx, gy, mean) -> None:
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])
    y = np.asarray(data["y"])

    fig = plt.figure(figsize=(8.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=float(y.min()), vmax=float(y.max()))
    ax.scatter(x1, x2, y, c=y, cmap="magma", norm=norm,
               s=12, depthshade=True, edgecolor="white",
               linewidth=0.25, alpha=0.95)
    ax.plot_wireframe(gx, gy, mean, rstride=6, cstride=6,
                      color="#111827", linewidth=0.45, alpha=0.5)

    ax.set_title("Wireframe fit over the raw scatter",
                 pad=2, loc="left")
    style_3d_axes(ax, xlabel="x₁", ylabel="x₂", zlabel="y")
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.65))

    fig.text(0.5, 0.02,
             "Points = raw observations.  Wireframe = the smoothed surface "
             "the engine fit through them.",
             ha="center", fontsize=8.5, color="#6b7280")

    out = DOCS_IMAGES / "surface_3d_wireframe.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5 — 3D side-by-side Matérn vs Duchon
# ---------------------------------------------------------------------------
def render_3d_compare(surfaces) -> None:
    pick = {name: (gx, gy, mean) for name, gx, gy, mean in surfaces}
    if "Matérn" not in pick or "Duchon" not in pick:
        print("skip surface_3d_compare (missing matern or duchon)")
        return
    gx1, gy1, mean_m = pick["Matérn"]
    gx2, gy2, mean_d = pick["Duchon"]

    vmin = float(min(np.nanmin(mean_m), np.nanmin(mean_d)))
    vmax = float(max(np.nanmax(mean_m), np.nanmax(mean_d)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(12.4, 5.6))
    for i, (title, gx, gy, mean) in enumerate(
        [("Matérn", gx1, gy1, mean_m), ("Duchon", gx2, gy2, mean_d)],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        rgb = ls.shade(mean, cmap=cm.magma, norm=norm,
                       vert_exag=0.6, blend_mode="soft")
        ax.plot_surface(gx, gy, mean, facecolors=rgb,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)
        ax.contour(gx, gy, mean, zdir="z",
                   offset=vmin - 0.05, levels=10,
                   cmap="magma", linewidths=0.45, alpha=0.7)
        ax.set_title(title, pad=2, loc="left")
        ax.set_zlim(vmin - 0.05, vmax + 0.10)
        style_3d_axes(ax, xlabel="x₁", ylabel="x₂", zlabel="ŷ")
        ax.view_init(elev=26, azim=-55)
        ax.set_box_aspect((1.0, 1.0, 0.55))

    fig.text(0.5, 0.02,
             "Same 800 noisy observations, two surface-smooth families on "
             "the same colour scale.",
             ha="center", fontsize=8.5, color="#6b7280")

    out = DOCS_IMAGES / "surface_3d_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6 — Marginal-slope: baseline + score-elevated surface over (pc1, pc2)
# ---------------------------------------------------------------------------
def render_marginal_slope_3d() -> None:
    rng = np.random.default_rng(2026)
    n = 1500

    # 2-D ancestry-PC-like covariate space
    pc1 = rng.uniform(-1.0, 1.0, n)
    pc2 = rng.uniform(-1.0, 1.0, n)

    # standardised risk score, conditional on PCs (synthetic z, mean 0 / sd 1)
    z = rng.standard_normal(n)

    # baseline log-odds varies smoothly across the PC plane
    baseline_logit = (
        -2.0
        + 1.4 * np.exp(-((pc1 - 0.3) ** 2 + (pc2 + 0.2) ** 2) / 0.22)
        - 0.9 * np.exp(-((pc1 + 0.4) ** 2 + (pc2 - 0.5) ** 2) / 0.18)
    )
    # The score's *slope* is itself a smooth function of (pc1, pc2):
    # strong effect near (+0.5, +0.5), weak near the opposite corner.
    slope_surface = 0.6 + 1.4 * (
        1.0 / (1.0 + np.exp(-2.5 * (pc1 + pc2 - 0.4)))
    )

    eta = baseline_logit + slope_surface * z
    prob = 1.0 / (1.0 + np.exp(-eta))
    case = (rng.uniform(size=n) < prob).astype(float)

    data = {
        "case": case.tolist(),
        "pc1":  pc1.tolist(),
        "pc2":  pc2.tolist(),
        "z":    z.tolist(),
    }

    print("[marginal-slope] fitting...")
    model = gamfit.fit(
        data,
        "case ~ duchon(pc1, pc2, centers=30)",
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="z",
        logslope_formula="duchon(pc1, pc2, centers=30)",
    )
    print("[marginal-slope] fit complete")

    side = 70
    g = np.linspace(-1.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)
    flat_x = gx.ravel().tolist()
    flat_y = gy.ravel().tolist()

    def predict_surface(z_val: float) -> np.ndarray:
        pred = model.predict(
            {"pc1": flat_x, "pc2": flat_y, "z": [z_val] * len(flat_x)},
            return_type="dict",
        )
        if isinstance(pred, dict) and "mean" in pred:
            return np.asarray(pred["mean"], dtype=float).reshape(side, side)
        # bernoulli marginal-slope returns a bare 1-D numpy array by default
        return np.asarray(pred, dtype=float).reshape(side, side)

    p_base = predict_surface(0.0)
    p_high = predict_surface(+2.0)

    vmin = float(min(p_base.min(), p_high.min()))
    vmax = float(max(p_base.max(), p_high.max()))

    fig = plt.figure(figsize=(10.4, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=315, altdeg=45)
    rgb_base = ls.shade(p_base, cmap=cm.viridis,
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        vert_exag=0.6, blend_mode="soft")
    rgb_high = ls.shade(p_high, cmap=cm.magma,
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        vert_exag=0.6, blend_mode="soft")

    ax.plot_surface(
        gx, gy, p_base, facecolors=rgb_base,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
        shade=False, alpha=0.85,
    )
    ax.plot_surface(
        gx, gy, p_high, facecolors=rgb_high,
        rstride=1, cstride=1, linewidth=0, antialiased=True,
        shade=False, alpha=0.85,
    )

    ax.contour(gx, gy, p_base, zdir="z", offset=vmin - 0.02,
               levels=10, cmap="viridis", linewidths=0.5, alpha=0.7)

    # Legend chips: two coloured patches on the title
    fig.text(0.18, 0.92, "  ", backgroundcolor="#3b528b",
             fontsize=10, color="white")
    fig.text(0.205, 0.918,
             "baseline P(case | z = 0)",
             fontsize=9.5, color="#374151", va="center")
    fig.text(0.18, 0.88, "  ", backgroundcolor="#b73779",
             fontsize=10, color="white")
    fig.text(0.205, 0.878,
             "elevated P(case | z = +2)   →  the *slope* surface lifts more "
             "in some regions than others",
             fontsize=9.5, color="#374151", va="center")

    ax.set_title("Marginal-slope GAM:  baseline risk + spatially-varying "
                 "score effect",
                 pad=2, loc="left", fontsize=11)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_zlim(vmin - 0.02, vmax + 0.04)
    ax.set_xlabel("pc₁", labelpad=4)
    ax.set_ylabel("pc₂", labelpad=4)
    ax.set_zlabel("P(case)", labelpad=4)
    style_3d_axes(ax, xlabel="pc₁", ylabel="pc₂", zlabel="P(case)")
    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.65))

    fig.text(
        0.5, 0.015,
        "Two Duchon-smooth surfaces on the joint (pc₁, pc₂) plane:  "
        "baseline prevalence (viridis) and the same population evaluated "
        "at z = +2 (magma).\n"
        "The vertical gap between them is the marginal-slope effect — and "
        "the gap itself is a smooth function of where you stand in PC space.",
        ha="center", fontsize=8.5, color="#6b7280",
    )

    out = DOCS_IMAGES / "marginal_slope_3d.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    data = make_regression()

    # 2D + 3D figures share the matern fit
    _, gx, gy, _ = render_2d_hero(data)
    surfaces_dict, surfaces_list = render_zoo(data)
    # Find the matern surface from the zoo so we don't refit
    matern_mean = surfaces_dict["Matérn"]
    side = matern_mean.shape[0]
    g = np.linspace(0.0, 1.0, side)
    gx_z, gy_z = np.meshgrid(g, g)

    render_3d_shaded(gx_z, gy_z, matern_mean)
    render_3d_wireframe(data, gx_z, gy_z, matern_mean)
    render_3d_compare(surfaces_list)
    render_marginal_slope_3d()


if __name__ == "__main__":
    main()
