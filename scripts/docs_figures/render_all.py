"""Render the README + docs figure portfolio from real `gamfit` fits.

Outputs land in docs/images/:

  surface_fit_hero.png       2D, 3 panels: data → fitted mean → predictive SE
  smooth_zoo.png             2D, 4 panels: thinplate / matern / duchon / te
  surface_3d_shaded.png      3D shaded surface of the matern fit
  surface_3d_wireframe.png   3D wireframe over the noisy point cloud
  surface_3d_compare.png     3D side-by-side: matern vs duchon
  marginal_slope_3d.png      Two-surface marginal-slope viz (baseline + score)

Each render is wrapped in try/except so one failure doesn't take the rest
down. Output is unbuffered. Run with .venv313/bin/python -u.
"""
from __future__ import annotations

import traceback
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib import colormaps

import gamfit


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Editorial style. Helvetica is reliable on macOS; falls back to sans-serif.
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
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    }
)

DOCS_IMAGES = Path(__file__).resolve().parents[2] / "docs" / "images"
DOCS_IMAGES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def truth_2d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """A richer landscape: three peaks + saddle + ripple."""
    peak_a = 1.4 * np.exp(-((x1 - 0.25) ** 2 + (x2 - 0.70) ** 2) / 0.06)
    peak_b = -1.1 * np.exp(-((x1 - 0.75) ** 2 + (x2 - 0.25) ** 2) / 0.05)
    peak_c = 0.7 * np.exp(-((x1 - 0.80) ** 2 + (x2 - 0.78) ** 2) / 0.04)
    saddle = 0.35 * (x1 - x2)
    ripple = 0.18 * np.sin(2.0 * np.pi * (x1 + 0.5 * x2))
    return peak_a + peak_b + peak_c + saddle + ripple


def make_regression(n: int = 800, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = truth_2d(x1, x2) + 0.18 * rng.standard_normal(n)
    return {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}


def grid_eval(model, side=110, with_se=False, lo=0.0, hi=1.0):
    g = np.linspace(lo, hi, side)
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


def style_square_axes_2d(ax, xlabel=None, ylabel=None) -> None:
    ax.set_aspect("equal")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])


def style_3d_axes(ax, xlabel="x1", ylabel="x2", zlabel="z") -> None:
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_zlabel(zlabel, labelpad=4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#d1d5db")
        axis.pane.set_linewidth(0.4)
        axis.pane.fill = False
    ax.tick_params(axis="x", pad=0)
    ax.tick_params(axis="y", pad=0)
    ax.tick_params(axis="z", pad=2)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure 1: 2D three-panel hero
# ---------------------------------------------------------------------------
def render_hero(data: dict, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("[hero] grid eval")
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])
    y = np.asarray(data["y"])
    gx, gy, mean, se = grid_eval(model, side=120, with_se=True)

    vmin = float(min(y.min(), mean.min()))
    vmax = float(max(y.max(), mean.max()))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0),
                             gridspec_kw={"wspace": 0.18})

    ax = axes[0]
    sc = ax.scatter(x1, x2, c=y, cmap="magma", s=18,
                    edgecolor="white", linewidth=0.35,
                    vmin=vmin, vmax=vmax)
    ax.set_title("Noisy observations", pad=10)
    style_square_axes_2d(ax, xlabel="x1", ylabel="x2")
    style_colorbar(fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[1]
    cf = ax.contourf(gx, gy, mean, levels=24, cmap="magma",
                     vmin=vmin, vmax=vmax)
    ax.contour(gx, gy, mean, levels=8, colors="white",
               linewidths=0.4, alpha=0.55)
    ax.set_title("matern(x1, x2) — fitted mean", pad=10)
    style_square_axes_2d(ax, xlabel="x1")
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    ax = axes[2]
    se_plot = se if se is not None else np.zeros_like(mean)
    cf = ax.contourf(gx, gy, se_plot, levels=24, cmap="viridis")
    ax.scatter(x1, x2, s=2.5, color="white", alpha=0.5, linewidth=0)
    ax.set_title("Posterior predictive SE", pad=10)
    style_square_axes_2d(ax, xlabel="x1")
    style_colorbar(fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04))

    out = DOCS_IMAGES / "surface_fit_hero.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    log(f"wrote {out}")
    return gx, gy, mean


# ---------------------------------------------------------------------------
# Figure 2: 2D zoo
# ---------------------------------------------------------------------------
def render_zoo(data: dict, matern_mean: np.ndarray):
    log("[zoo] fitting alternates...")
    x1 = np.asarray(data["x1"])
    x2 = np.asarray(data["x2"])
    side = matern_mean.shape[0]
    g = np.linspace(0.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)

    surfaces = [("Matérn", gx, gy, matern_mean)]
    specs = [
        ("thin-plate", "y ~ thinplate(x1, x2)"),
        ("Duchon",     "y ~ duchon(x1, x2, centers=25)"),
        ("tensor",     "y ~ te(x1, x2)"),
    ]
    for name, formula in specs:
        log(f"[zoo] fit {name}")
        try:
            m = gamfit.fit(data, formula)
            _, _, mean, _ = grid_eval(m, side=side, with_se=False)
            surfaces.append((name, gx, gy, mean))
            log(f"[zoo] {name} done")
        except Exception as exc:
            log(f"[zoo] {name} FAILED: {exc!s}")

    order = ["thin-plate", "Matérn", "Duchon", "tensor"]
    surfaces.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)

    vmin = min(float(np.percentile(s, 2)) for _, _, _, s in surfaces)
    vmax = max(float(np.percentile(s, 98)) for _, _, _, s in surfaces)

    n_panels = len(surfaces)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.35 * n_panels, 3.7),
                             gridspec_kw={"wspace": 0.10})
    if n_panels == 1:
        axes = [axes]
    for ax, (name, gx_, gy_, mean) in zip(axes, surfaces):
        cf = ax.contourf(gx_, gy_, mean, levels=22, cmap="magma",
                         vmin=vmin, vmax=vmax)
        ax.contour(gx_, gy_, mean, levels=6, colors="white",
                   linewidths=0.35, alpha=0.55)
        ax.scatter(x1, x2, s=1.6, color="white", alpha=0.35, linewidth=0)
        ax.set_title(name, pad=8)
        style_square_axes_2d(
            ax, xlabel="x1",
            ylabel="x2" if ax is axes[0] else None,
        )

    cbar = fig.colorbar(cf, ax=axes if isinstance(axes, list) else list(axes),
                        fraction=0.018, pad=0.02)
    style_colorbar(cbar)

    out = DOCS_IMAGES / "smooth_zoo.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    log(f"wrote {out}")
    return {name: surf for name, _, _, surf in surfaces}


# ---------------------------------------------------------------------------
# Figure 3: 3D shaded surface
# ---------------------------------------------------------------------------
def render_3d_shaded(gx, gy, mean) -> None:
    fig = plt.figure(figsize=(8.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(mean, cmap=colormaps["magma"],
                   vert_exag=0.6, blend_mode="soft")
    ax.plot_surface(gx, gy, mean, facecolors=rgb,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)
    ax.contour(gx, gy, mean, zdir="z",
               offset=float(np.nanmin(mean)) - 0.05,
               levels=12, cmap="magma", linewidths=0.6, alpha=0.7)

    ax.set_zlim(float(np.nanmin(mean)) - 0.05,
                float(np.nanmax(mean)) + 0.1)
    style_3d_axes(ax, xlabel="x1", ylabel="x2", zlabel="y")
    ax.view_init(elev=28, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.55))

    out = DOCS_IMAGES / "surface_3d_shaded.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: 3D wireframe + scatter
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

    style_3d_axes(ax, xlabel="x1", ylabel="x2", zlabel="y")
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.65))

    out = DOCS_IMAGES / "surface_3d_wireframe.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5: 3D side-by-side Matérn vs Duchon
# ---------------------------------------------------------------------------
def render_3d_compare(surfaces: dict) -> None:
    if "Matérn" not in surfaces or "Duchon" not in surfaces:
        log("[compare] skipping (missing matern/duchon)")
        return

    side = surfaces["Matérn"].shape[0]
    g = np.linspace(0.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)
    mean_m = surfaces["Matérn"]
    mean_d = surfaces["Duchon"]

    vmin = float(min(np.nanmin(mean_m), np.nanmin(mean_d)))
    vmax = float(max(np.nanmax(mean_m), np.nanmax(mean_d)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(12.4, 5.6))
    for i, (title, mean) in enumerate([("Matérn", mean_m),
                                       ("Duchon", mean_d)], start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        rgb = ls.shade(mean, cmap=colormaps["magma"], norm=norm,
                       vert_exag=0.6, blend_mode="soft")
        ax.plot_surface(gx, gy, mean, facecolors=rgb,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)
        ax.contour(gx, gy, mean, zdir="z",
                   offset=vmin - 0.05, levels=10,
                   cmap="magma", linewidths=0.45, alpha=0.7)
        ax.set_title(title, pad=2, loc="left",
                     fontsize=11, fontweight="semibold")
        ax.set_zlim(vmin - 0.05, vmax + 0.10)
        style_3d_axes(ax, xlabel="x1", ylabel="x2", zlabel="y")
        ax.view_init(elev=26, azim=-55)
        ax.set_box_aspect((1.0, 1.0, 0.55))

    out = DOCS_IMAGES / "surface_3d_compare.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6: Marginal-slope-style two-surface viz over (pc1, pc2)
#
# The current local gamfit build hits a "missing latent_measure" bug on the
# bernoulli-marginal-slope predict path.  We get the same visual story with
# a regular binomial GAM whose RHS is a joint Duchon smooth over
# (pc1, pc2, z), then evaluate at two values of z.  The vertical gap
# between the two surfaces *is* the spatially-varying score effect — exactly
# what the marginal-slope decomposition expresses, computed via the
# capability that's reliable in this build.
# ---------------------------------------------------------------------------
def render_marginal_slope_3d() -> None:
    log("[marginal-slope] generating synthetic data")
    rng = np.random.default_rng(2026)
    n = 2000
    pc1 = rng.uniform(-1.0, 1.0, n)
    pc2 = rng.uniform(-1.0, 1.0, n)
    z = rng.standard_normal(n)

    # Baseline log-odds: three peaks + a valley.  Gives the lower
    # surface a clearly non-flat landscape.
    baseline = (
        -0.6
        + 1.4 * np.exp(-((pc1 - 0.30) ** 2 + (pc2 + 0.15) ** 2) / 0.14)
        - 1.0 * np.exp(-((pc1 + 0.50) ** 2 + (pc2 - 0.55) ** 2) / 0.12)
        + 0.8 * np.exp(-((pc1 + 0.15) ** 2 + (pc2 + 0.60) ** 2) / 0.10)
    )
    # Spatially-varying slope-of-z: always positive but a sharp wedge,
    # ≈ 0.2 in the bottom-left to ≈ 2.6 in the top-right.  Pulls the
    # elevated surface farther from the baseline in one corner than the
    # other without making them cross.
    slope = 0.2 + 2.4 / (1.0 + np.exp(-3.0 * (pc1 + pc2 - 0.0)))
    eta = baseline + slope * z
    prob = 1.0 / (1.0 + np.exp(-eta))
    case = (rng.uniform(size=n) < prob).astype(float)

    data = {
        "case": case.tolist(),
        "pc1":  pc1.tolist(),
        "pc2":  pc2.tolist(),
        "z":    z.tolist(),
    }

    log("[marginal-slope] fitting joint Duchon smooth over (pc1, pc2, z)")
    model = gamfit.fit(
        data,
        "case ~ duchon(pc1, pc2, z, centers=40)",
        scale_dimensions=True,
    )
    log("[marginal-slope] fit complete")

    side = 60
    g = np.linspace(-1.0, 1.0, side)
    gx, gy = np.meshgrid(g, g)
    flat_x = gx.ravel().tolist()
    flat_y = gy.ravel().tolist()

    def surface_at(z_val: float) -> np.ndarray:
        pred = model.predict(
            {"pc1": flat_x, "pc2": flat_y, "z": [z_val] * len(flat_x)},
            return_type="dict",
        )
        m = np.asarray(pred["mean"], dtype=float).reshape(side, side)
        return np.clip(m, 0.0, 1.0)

    log("[marginal-slope] predicting baseline (z=0)")
    p_base = surface_at(0.0)
    log("[marginal-slope] predicting elevated (z=+2)")
    p_high = surface_at(+2.0)

    vmin = float(min(p_base.min(), p_high.min()))
    vmax = float(max(p_base.max(), p_high.max()))

    fig = plt.figure(figsize=(9.6, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    ls = LightSource(azdeg=315, altdeg=45)
    rgb_base = ls.shade(p_base, cmap=colormaps["viridis"],
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        vert_exag=0.6, blend_mode="soft")
    rgb_high = ls.shade(p_high, cmap=colormaps["magma"],
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        vert_exag=0.6, blend_mode="soft")

    ax.plot_surface(gx, gy, p_base, facecolors=rgb_base,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=0.92)
    ax.plot_surface(gx, gy, p_high, facecolors=rgb_high,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=0.92)
    ax.contour(gx, gy, p_base, zdir="z", offset=vmin - 0.02,
               levels=10, cmap="viridis", linewidths=0.5, alpha=0.7)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(vmin - 0.02, vmax + 0.04)
    # `set_zlabel` gets clipped at this camera angle.  Annotate the
    # z-quantity as a figure-coord label instead, on the same row as the
    # two colour chips.
    style_3d_axes(ax, xlabel="pc1", ylabel="pc2", zlabel="")
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.0, 1.0, 0.65))

    # Top-row annotations: quantity name on the left, two colour chips
    # on the right, tightly grouped.
    fig.text(0.08, 0.94, "P(case)",
             fontsize=11, fontweight="semibold", color="#0f172a",
             va="center")
    fig.text(0.45, 0.94, "   ", backgroundcolor="#3b528b",
             fontsize=11, color="white", va="center")
    fig.text(0.495, 0.938, "z = 0",
             fontsize=10, color="#374151", va="center")
    fig.text(0.66, 0.94, "   ", backgroundcolor="#b73779",
             fontsize=11, color="white", va="center")
    fig.text(0.705, 0.938, "z = +2",
             fontsize=10, color="#374151", va="center")

    out = DOCS_IMAGES / "marginal_slope_3d.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log(f"wrote {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def safe(label: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        log(f"[{label}] FAILED")
        traceback.print_exc()
        return None


def main() -> None:
    log("=== render_all start ===")
    data = make_regression()
    log("[main] fitting hero smooth (matern preferred, thin-plate fallback)")
    hero_smooth = "Matérn"
    try:
        model = gamfit.fit(data, "y ~ matern(x1, x2)")
    except Exception as exc:
        log(f"[main] matern failed ({exc!s}); falling back to thin-plate")
        hero_smooth = "thin-plate"
        model = gamfit.fit(data, "y ~ thinplate(x1, x2)")
    log(f"[main] hero {hero_smooth} fit done")

    result = safe("hero", render_hero, data, model)
    if result is None:
        log("[main] hero failed, aborting downstream 3D")
        gx = gy = mean = None
    else:
        gx, gy, mean = result

    surfaces = safe("zoo", render_zoo, data, mean)

    if gx is not None and mean is not None:
        safe("3d_shaded", render_3d_shaded, gx, gy, mean)
        safe("3d_wireframe", render_3d_wireframe, data, gx, gy, mean)
    if surfaces is not None:
        safe("3d_compare", render_3d_compare, surfaces)

    safe("marginal_slope", render_marginal_slope_3d)

    log("=== render_all done ===")


if __name__ == "__main__":
    main()
