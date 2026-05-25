#!/usr/bin/env python3
"""Geometric smooths demo — recover six shapes from noisy 3D point clouds.

Generates a trefoil knot, a wiggly loop, a wobbly cylinder, a lumpy sphere,
a bumpy torus, and a 4π-periodic double-cover of a Möbius
embedding. Each shape is sampled with
known (latent) parameters; we observe noisy (x, y, z) and fit one geometric
smooth per coordinate using the `gam` CLI. The recovered surfaces / curves
are rendered alongside the raw clouds in a single composite frame.

Outputs (siblings of this script):
    geometric_shapes_demo.png   high-resolution still
    geometric_shapes_demo.mp4   browser-friendly rotating animation
    geometric_shapes_demo.gif   GIF version (Pillow-encoded, plays everywhere)

Run from repo root:

    uv run --with numpy --with pyvista --with matplotlib --with imageio \\
           --with imageio-ffmpeg --with pillow python3 \\
           scripts/geometric_shapes_demo.py

Re-uses cached fits/data when present; pass `--regen` to start clean.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent
DATA = SCRIPT.parent / "geometric_shapes_demo_data"
GAM = REPO / "target" / "release" / "gam"

TAU = 2 * np.pi
RNG = np.random.default_rng(2025)

# Per-shape jewel-tone palette
PAL = dict(
    trefoil="#1B6E64", loop="#3D7A2F",
    cylinder="#2A407A", sphere="#7A3463",
    torus="#A87520", mobius="#C9533C",
)


# ---------------------------------------------------------------------------
# Data generation (with markedly higher noise than the in-tree visuals)
# ---------------------------------------------------------------------------
def gen_trefoil(
    n: int = 2200, sig: float = 0.40
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    t = RNG.uniform(0, TAU, n)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return t, (x, y, z), sig


def gen_cyl(
    n: int = 4000, sig: float = 0.14
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
]:
    th = RNG.uniform(0, TAU, n)
    h = RNG.uniform(0, 1, n)
    r = 1.0 + 0.18 * np.sin(4 * th + 3 * np.pi * h)
    return (th, h), (r * np.cos(th), r * np.sin(th), 2 * (h - 0.5)), sig


def gen_sph(
    n: int = 4500, sig: float = 0.10
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
]:
    lat = np.arcsin(RNG.uniform(-1, 1, n))
    lon = RNG.uniform(0, TAU, n)

    def gd(lat0: float, lon0: float) -> np.ndarray:
        c = (np.sin(lat) * np.sin(lat0)
             + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0))
        return np.asarray(np.arccos(np.clip(c, -1, 1)))

    # Several distinct bulges, one deep crater, plus a rippling overlay.
    r: float | np.ndarray = 1.0
    r += 0.55 * np.exp(-(gd(0.95, 0.30) / 0.45) ** 2)
    r += 0.42 * np.exp(-(gd(-0.55, 2.50) / 0.50) ** 2)
    r -= 0.40 * np.exp(-(gd(0.05, 4.20) / 0.55) ** 2)
    r += 0.32 * np.exp(-(gd(-1.20, 5.70) / 0.38) ** 2)
    r += 0.22 * np.sin(4 * lon) * np.cos(3 * lat)
    r += 0.10 * np.sin(6 * lon + 2 * lat)
    return (lat, lon), (
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ), sig


def gen_tor(
    n: int = 4500, sig: float = 0.12
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
]:
    u = RNG.uniform(0, TAU, n)
    v = RNG.uniform(0, TAU, n)
    R = 2.0 + 0.18 * np.sin(3 * v + 2 * u)
    r = 0.65 + 0.10 * np.cos(4 * u)
    return (u, v), (
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v),
    ), sig


def gen_mob(
    n: int = 4000, sig: float = 0.10
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    float,
]:
    # Standard Möbius embedding. Note: F(u+2π, v) = F(u, −v) and
    # F(u+4π, v) = F(u, v). The smoother is given only the 4π periodicity
    # (an orientable cylinder); the Möbius look comes from the embedding,
    # not from a twisted basis identification.
    u = RNG.uniform(0, 2 * TAU, n)
    v = RNG.uniform(-0.8, 0.8, n)
    rim = 1 + 0.5 * v * np.cos(u / 2)
    return (u, v), (rim * np.cos(u), rim * np.sin(u), 0.5 * v * np.sin(u / 2)), sig


def gen_loop(
    n: int = 1000, sig: float = 0.15
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    t = RNG.uniform(0, TAU, n)
    x = np.cos(t) + 0.10 * np.sin(3 * t)
    y = np.sin(t) + 0.10 * np.cos(4 * t)
    z = 0.45 * np.sin(2 * t) + 0.15 * np.cos(t)
    return t, (x, y, z), sig


def add_noise(
    coords: tuple[np.ndarray, np.ndarray, np.ndarray], sig: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = coords
    return (
        a + sig * RNG.standard_normal(a.shape),
        b + sig * RNG.standard_normal(b.shape),
        c + sig * RNG.standard_normal(c.shape),
    )


# ---------------------------------------------------------------------------
# Step 1: write CSVs
# ---------------------------------------------------------------------------
def write_csvs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    # Trefoil (param = t)
    t, coords, sig = gen_trefoil()
    x, y, z = add_noise(coords, sig)
    np.savetxt(DATA / "tref.csv", np.column_stack([t, x, y, z]),
               delimiter=",", header="t,x,y,z", comments="", fmt="%.6f")

    # Cylinder (theta, h)
    (th, h), coords, sig = gen_cyl()
    x, y, z = add_noise(coords, sig)
    np.savetxt(DATA / "cyl.csv", np.column_stack([th, h, x, y, z]),
               delimiter=",", header="theta,h,x,y,z", comments="", fmt="%.6f")

    # Sphere (lat, lon)
    (lat, lon), coords, sig = gen_sph()
    x, y, z = add_noise(coords, sig)
    np.savetxt(DATA / "sph.csv", np.column_stack([lat, lon, x, y, z]),
               delimiter=",", header="lat,lon,x,y,z", comments="", fmt="%.6f")

    # Torus (u, v)
    (u, v), coords, sig = gen_tor()
    x, y, z = add_noise(coords, sig)
    np.savetxt(DATA / "tor.csv", np.column_stack([u, v, x, y, z]),
               delimiter=",", header="u,v,x,y,z", comments="", fmt="%.6f")

    # 4π-periodic Möbius double-cover (u, v)
    (u, v), coords, sig = gen_mob()
    x, y, z = add_noise(coords, sig)
    np.savetxt(DATA / "mob.csv", np.column_stack([u, v, x, y, z]),
               delimiter=",", header="u,v,x,y,z", comments="", fmt="%.6f")

    # Loop (param = t)
    t, coords, sig = gen_loop()
    xn, yn, zn = add_noise(coords, sig)
    np.savetxt(DATA / "loop.csv", np.column_stack([t, xn, yn, zn]),
               delimiter=",", header="t,x,y,z", comments="", fmt="%.6f")


# ---------------------------------------------------------------------------
# Step 2: fit + predict via the gam CLI
# ---------------------------------------------------------------------------
FITS = [
    # (name, csv, formula)
    ("tref_x", "tref.csv",
     "x ~ s(t, periodic=true, period=6.283185307, k=24)"),
    ("tref_y", "tref.csv",
     "y ~ s(t, periodic=true, period=6.283185307, k=24)"),
    ("tref_z", "tref.csv",
     "z ~ s(t, periodic=true, period=6.283185307, k=24)"),
    ("loop_x", "loop.csv",
     "x ~ s(t, periodic=true, period=6.283185307, k=18)"),
    ("loop_y", "loop.csv",
     "y ~ s(t, periodic=true, period=6.283185307, k=18)"),
    ("loop_z", "loop.csv",
     "z ~ s(t, periodic=true, period=6.283185307, k=18)"),
    ("cyl_x", "cyl.csv",
     "x ~ te(theta, h, periodic=[0], period=[6.283185307, None], k=[26,12])"),
    ("cyl_y", "cyl.csv",
     "y ~ te(theta, h, periodic=[0], period=[6.283185307, None], k=[26,12])"),
    ("cyl_z", "cyl.csv",
     "z ~ te(theta, h, periodic=[0], period=[6.283185307, None], k=[26,12])"),
    ("sph_x", "sph.csv",
     "x ~ sphere(lat, lon, radians=true, k=100)"),
    ("sph_y", "sph.csv",
     "y ~ sphere(lat, lon, radians=true, k=100)"),
    ("sph_z", "sph.csv",
     "z ~ sphere(lat, lon, radians=true, k=100)"),
    ("tor_x", "tor.csv",
     "x ~ te(u, v, periodic=[0,1], period=[6.283185307, 6.283185307], k=[20,16])"),
    ("tor_y", "tor.csv",
     "y ~ te(u, v, periodic=[0,1], period=[6.283185307, 6.283185307], k=[20,16])"),
    ("tor_z", "tor.csv",
     "z ~ te(u, v, periodic=[0,1], period=[6.283185307, 6.283185307], k=[20,16])"),
    ("mob_x", "mob.csv",
     "x ~ te(u, v, periodic=[0], period=[12.566370614, None], k=[32,10])"),
    ("mob_y", "mob.csv",
     "y ~ te(u, v, periodic=[0], period=[12.566370614, None], k=[32,10])"),
    ("mob_z", "mob.csv",
     "z ~ te(u, v, periodic=[0], period=[12.566370614, None], k=[32,10])"),
]


def fit_all() -> None:
    if not GAM.exists():
        sys.exit(f"missing release binary: {GAM} — run `cargo build --release`")
    for name, csv, formula in FITS:
        model = DATA / f"{name}.model"
        if model.exists():
            print(f"  skip  {name:8s} (cached)")
            continue
        t0 = time.time()
        r = subprocess.run(
            [str(GAM), "fit", str(DATA / csv), formula, "--out", str(model)],
            capture_output=True, text=True,
        )
        if r.returncode:
            (DATA / f"{name}.fit.log").write_text(r.stdout + r.stderr)
            sys.exit(f"fit failed for {name}; see {DATA}/{name}.fit.log")
        print(f"  fit   {name:8s} {time.time()-t0:5.1f}s")


def write_grids() -> None:
    # 1-D grid for the curves
    np.savetxt(DATA / "grid_1d.csv",
               np.linspace(0, TAU, 800, endpoint=False)[:, None],
               delimiter=",", header="t", comments="", fmt="%.6f")
    # cylinder grid
    NTH, NH = 96, 36
    TH, H = np.meshgrid(np.linspace(0, TAU, NTH, endpoint=False),
                         np.linspace(0, 1, NH))
    np.savetxt(DATA / "grid_cyl.csv",
               np.column_stack([TH.ravel(), H.ravel()]),
               delimiter=",", header="theta,h", comments="", fmt="%.6f")
    np.save(DATA / "grid_cyl_shape.npy", np.array([NTH, NH]))
    # sphere grid
    NLAT, NLON = 80, 160
    LAT, LON = np.meshgrid(
        np.linspace(-np.pi / 2 + 0.02, np.pi / 2 - 0.02, NLAT),
        np.linspace(0, TAU, NLON, endpoint=False),
    )
    np.savetxt(DATA / "grid_sph.csv",
               np.column_stack([LAT.ravel(), LON.ravel()]),
               delimiter=",", header="lat,lon", comments="", fmt="%.6f")
    np.save(DATA / "grid_sph_shape.npy", np.array([NLAT, NLON]))
    # torus grid
    NU, NV = 128, 64
    U, V = np.meshgrid(np.linspace(0, TAU, NU, endpoint=False),
                        np.linspace(0, TAU, NV, endpoint=False))
    np.savetxt(DATA / "grid_tor.csv",
               np.column_stack([U.ravel(), V.ravel()]),
               delimiter=",", header="u,v", comments="", fmt="%.6f")
    np.save(DATA / "grid_tor_shape.npy", np.array([NU, NV]))
    # 4π-periodic Möbius double-cover grid
    NMU, NMV = 240, 36
    UM, VM = np.meshgrid(np.linspace(0, 2 * TAU, NMU, endpoint=False),
                          np.linspace(-0.8, 0.8, NMV))
    np.savetxt(DATA / "grid_mob.csv",
               np.column_stack([UM.ravel(), VM.ravel()]),
               delimiter=",", header="u,v", comments="", fmt="%.6f")
    np.save(DATA / "grid_mob_shape.npy", np.array([NMU, NMV]))


PRED_MAP = {
    "tref_x": "grid_1d.csv", "tref_y": "grid_1d.csv", "tref_z": "grid_1d.csv",
    "loop_x": "grid_1d.csv", "loop_y": "grid_1d.csv", "loop_z": "grid_1d.csv",
    "cyl_x": "grid_cyl.csv", "cyl_y": "grid_cyl.csv", "cyl_z": "grid_cyl.csv",
    "sph_x": "grid_sph.csv", "sph_y": "grid_sph.csv", "sph_z": "grid_sph.csv",
    "tor_x": "grid_tor.csv", "tor_y": "grid_tor.csv", "tor_z": "grid_tor.csv",
    "mob_x": "grid_mob.csv", "mob_y": "grid_mob.csv", "mob_z": "grid_mob.csv",
}


def predict_all() -> None:
    for name, grid in PRED_MAP.items():
        out = DATA / f"{name}_pred.csv"
        if out.exists():
            continue
        r = subprocess.run(
            [str(GAM), "predict", str(DATA / f"{name}.model"),
             str(DATA / grid), "--out", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode:
            (DATA / f"{name}.pred.log").write_text(r.stdout + r.stderr)
            sys.exit(f"predict failed for {name}")


# ---------------------------------------------------------------------------
# Step 3: render
# ---------------------------------------------------------------------------
def cmap_for(hex_color: str, name: str = "c") -> LinearSegmentedColormap:
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    light = np.minimum(1, rgb + 0.55)
    dark = rgb * 0.40
    return LinearSegmentedColormap.from_list(name, [light, rgb, dark], N=128)


def load_pred(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}_pred.csv", delimiter=",", skiprows=1)[:, 1]


def wrap0(A: np.ndarray) -> np.ndarray:
    return np.concatenate([A, A[:1, :]], axis=0)


def wrap1(A: np.ndarray) -> np.ndarray:
    return np.concatenate([A, A[:, :1]], axis=1)


def wrap2(A: np.ndarray) -> np.ndarray:
    A = np.concatenate([A, A[:, :1]], axis=1)
    return np.concatenate([A, A[:1, :]], axis=0)


def make_polydata(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, eps_factor: float = 1e-4
) -> Any:
    """Convert a coord-array surface to a PolyData with shared seam vertices
    and smoothed per-vertex normals. Cleans coincident vertices so wrap-seams
    do not produce shading discontinuities."""
    sg = pv.StructuredGrid(X, Y, Z)
    poly = sg.extract_surface()
    extent = float(np.linalg.norm(
        [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
    ))
    poly = poly.clean(tolerance=extent * eps_factor)
    poly = poly.compute_normals(split_vertices=False, feature_angle=180.0)
    return poly


def build_shapes() -> dict[str, dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Trefoil
    pts = np.loadtxt(DATA / "tref.csv", delimiter=",", skiprows=1)[:, 1:]
    fx, fy, fz = load_pred("tref_x"), load_pred("tref_y"), load_pred("tref_z")
    curve = np.vstack([np.column_stack([fx, fy, fz]),
                        [[fx[0], fy[0], fz[0]]]])
    out.append(dict(name="trefoil", kind="curve", cloud=pts, curve=curve,
                    color=PAL["trefoil"], cmap=cmap_for(PAL["trefoil"]),
                    elev=18, dist_mul=2.0, tube_r=0.12))

    # Loop
    pts = np.loadtxt(DATA / "loop.csv", delimiter=",", skiprows=1)[:, 1:]
    fx, fy, fz = load_pred("loop_x"), load_pred("loop_y"), load_pred("loop_z")
    curve = np.vstack([np.column_stack([fx, fy, fz]),
                        [[fx[0], fy[0], fz[0]]]])
    out.append(dict(name="loop", kind="curve", cloud=pts, curve=curve,
                    color=PAL["loop"], cmap=cmap_for(PAL["loop"]),
                    elev=18, dist_mul=2.0, tube_r=0.05))

    # Cylinder — periodic along θ (axis=1 of LAT-major reshape)
    NTH, NH = np.load(DATA / "grid_cyl_shape.npy")
    shape = (NH, NTH)
    cloud = np.loadtxt(DATA / "cyl.csv", delimiter=",", skiprows=1)[:, 2:]
    X = wrap1(load_pred("cyl_x").reshape(shape))
    Y = wrap1(load_pred("cyl_y").reshape(shape))
    Z = wrap1(load_pred("cyl_z").reshape(shape))
    out.append(dict(name="cylinder", kind="surface", cloud=cloud,
                    mesh=make_polydata(X, Y, Z),
                    color=PAL["cylinder"], cmap=cmap_for(PAL["cylinder"]),
                    elev=10, dist_mul=2.4))

    # Sphere — periodic along LON.  meshgrid(lat_1d, lon_1d) gives arrays of
    # shape (NLON, NLAT) where LON varies along axis=0, so wrap along axis=0.
    NLAT, NLON = np.load(DATA / "grid_sph_shape.npy")
    shape = (NLON, NLAT)
    cloud = np.loadtxt(DATA / "sph.csv", delimiter=",", skiprows=1)[:, 2:]
    X = wrap0(load_pred("sph_x").reshape(shape))
    Y = wrap0(load_pred("sph_y").reshape(shape))
    Z = wrap0(load_pred("sph_z").reshape(shape))
    out.append(dict(name="sphere", kind="surface", cloud=cloud,
                    mesh=make_polydata(X, Y, Z),
                    color=PAL["sphere"], cmap=cmap_for(PAL["sphere"]),
                    elev=15, dist_mul=2.4))

    # Torus — periodic along both axes
    NU, NV = np.load(DATA / "grid_tor_shape.npy")
    shape = (NV, NU)
    cloud = np.loadtxt(DATA / "tor.csv", delimiter=",", skiprows=1)[:, 2:]
    X = wrap2(load_pred("tor_x").reshape(shape))
    Y = wrap2(load_pred("tor_y").reshape(shape))
    Z = wrap2(load_pred("tor_z").reshape(shape))
    out.append(dict(name="torus", kind="surface", cloud=cloud,
                    mesh=make_polydata(X, Y, Z),
                    color=PAL["torus"], cmap=cmap_for(PAL["torus"]),
                    elev=28, dist_mul=2.4))

    # 4π-periodic Möbius double-cover: u wraps ordinarily; v is non-periodic.
    NMU, NMV = np.load(DATA / "grid_mob_shape.npy")
    shape = (NMV, NMU)
    cloud = np.loadtxt(DATA / "mob.csv", delimiter=",", skiprows=1)[:, 2:]
    X = wrap1(load_pred("mob_x").reshape(shape))
    Y = wrap1(load_pred("mob_y").reshape(shape))
    Z = wrap1(load_pred("mob_z").reshape(shape))
    out.append(dict(name="mobius", kind="surface", cloud=cloud,
                    mesh=make_polydata(X, Y, Z),
                    color=PAL["mobius"], cmap=cmap_for(PAL["mobius"]),
                    elev=20, dist_mul=2.4))

    return {s["name"]: s for s in out}


LAYOUT = [
    ("trefoil", "cloud", 0, 0), ("trefoil", "recov", 0, 1),
    ("loop",    "cloud", 0, 2), ("loop",    "recov", 0, 3),
    ("cylinder","cloud", 1, 0), ("cylinder","recov", 1, 1),
    ("sphere",  "cloud", 1, 2), ("sphere",  "recov", 1, 3),
    ("torus",   "cloud", 2, 0), ("torus",   "recov", 2, 1),
    ("mobius",  "cloud", 2, 2), ("mobius",  "recov", 2, 3),
]


def view_dir(elev_deg: float, azim_deg: float) -> np.ndarray:
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    return np.array([np.cos(e) * np.cos(a),
                     np.cos(e) * np.sin(a),
                     np.sin(e)])


def build_scene(
    plotter: Any,
    shapes: dict[str, dict[str, Any]],
    point_size: int,
) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for sname, role, r, c in LAYOUT:
        s = shapes[sname]
        plotter.subplot(r, c)
        plotter.set_background("white")

        centroid = s["cloud"].mean(0)
        extent = np.linalg.norm(s["cloud"].max(0) - s["cloud"].min(0))
        dist = s["dist_mul"] * extent

        if role == "cloud":
            mesh = pv.PolyData(s["cloud"])
            mesh["d"] = s["cloud"][:, 1].copy()
            plotter.add_mesh(
                mesh, scalars="d", cmap=s["cmap"],
                render_points_as_spheres=True, point_size=point_size,
                opacity=0.55, show_scalar_bar=False,
                lighting=True, ambient=0.4, diffuse=0.6, specular=0.2,
            )
            panels.append(dict(r=r, c=c, kind="cloud", mesh=mesh,
                                base=s["cloud"].copy(),
                                centroid=centroid, dist=dist, elev=s["elev"]))
        else:
            # Surfaces and tubes are colored solid with smooth shading.
            # Lighting (ambient + diffuse + specular) gives the depth cue;
            # adding a colormap across them produces visible bands at the
            # silhouette (surfaces) or where the loop closes (tubes).
            if s["kind"] == "curve":
                cpts = s["curve"]
                lines = np.hstack([[len(cpts)], np.arange(len(cpts))])
                poly = pv.PolyData(cpts, lines=lines)
                tube = poly.tube(radius=s["tube_r"], n_sides=28)
                plotter.add_mesh(
                    tube, color=s["color"], smooth_shading=True,
                    show_scalar_bar=False,
                    ambient=0.30, diffuse=0.75,
                    specular=0.55, specular_power=24,
                )
                panels.append(dict(r=r, c=c, kind="tube",
                                    centroid=centroid, dist=dist,
                                    elev=s["elev"]))
            else:
                plotter.add_mesh(
                    s["mesh"], color=s["color"], smooth_shading=True,
                    show_scalar_bar=False,
                    ambient=0.28, diffuse=0.78,
                    specular=0.45, specular_power=22,
                )
                panels.append(dict(r=r, c=c, kind="surface",
                                    centroid=centroid, dist=dist,
                                    elev=s["elev"]))
    return panels


def set_all_cameras(
    plotter: Any, panels: list[dict[str, Any]], azim_deg: float
) -> None:
    # `plotter.camera_position = ...` is silently no-op in subplot mode after
    # the initial render; using set_position/set_focus/set_viewup actually
    # mutates the active renderer's camera.
    for pn in panels:
        plotter.subplot(pn["r"], pn["c"])
        vd = view_dir(pn["elev"], azim_deg)
        eye = pn["centroid"] + pn["dist"] * vd
        plotter.set_position(tuple(eye))
        plotter.set_focus(tuple(pn["centroid"]))
        plotter.set_viewup((0, 0, 1))
        plotter.reset_camera_clipping_range()


def update_depth_scalars(
    panels: list[dict[str, Any]], azim_deg: float
) -> None:
    """Only clouds get camera-relative depth coloring; surfaces and tubes
    derive their depth from solid-color smooth-shading lighting."""
    for pn in panels:
        if pn["kind"] != "cloud":
            continue
        vd = view_dir(pn["elev"], azim_deg)
        pn["mesh"]["d"] = -(pn["base"] @ vd)


# ---------------------------------------------------------------------------
# Render outputs
# ---------------------------------------------------------------------------
def render_still(shapes: dict[str, dict[str, Any]], out_path: Path) -> None:
    W, H = 4000, 3000
    p = pv.Plotter(shape=(3, 4), off_screen=True,
                   window_size=[W, H], border=False)
    panels = build_scene(p, shapes, point_size=14)
    az = 35.0
    set_all_cameras(p, panels, az)
    update_depth_scalars(panels, az)
    p.screenshot(str(out_path), return_img=False)
    p.close()
    print(f"  still → {out_path}  ({W}×{H})")


def render_frames(
    shapes: dict[str, dict[str, Any]],
    w: int,
    h: int,
    n_frames: int,
    point_size: int,
) -> list[np.ndarray]:
    p = pv.Plotter(shape=(3, 4), off_screen=True,
                   window_size=[w, h], border=False)
    panels = build_scene(p, shapes, point_size=point_size)
    frames: list[np.ndarray] = []
    for k, az in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        set_all_cameras(p, panels, az)
        update_depth_scalars(panels, az)
        # PyVista's screenshot returns a view into the renderer's internal
        # buffer that gets overwritten next frame, so we must copy.
        img = np.asarray(p.screenshot(return_img=True)).copy()
        frames.append(img)
        if (k + 1) % 12 == 0:
            print(f"    frame {k+1}/{n_frames}")
    p.close()
    return frames


def render_mp4(shapes: dict[str, dict[str, Any]], out_path: Path) -> None:
    import imageio.v2 as imageio
    W, H = 1920, 1440
    if W % 2 or H % 2:
        raise SystemExit("mp4 dims must be even")
    n = 120
    frames = render_frames(shapes, W, H, n, point_size=8)
    writer = imageio.get_writer(
        str(out_path), fps=30, codec="libx264",
        quality=8, pixelformat="yuv420p", macro_block_size=1,
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  mp4   → {out_path}  ({W}×{H}, {n} frames @ 30 fps)")


def render_gif(shapes: dict[str, dict[str, Any]], out_path: Path) -> None:
    W, H = 1280, 960
    n = 60
    frames = render_frames(shapes, W, H, n, point_size=6)
    # Pillow gives us a properly-encoded animated GIF that plays in browsers,
    # Preview, and on Twitter (imageio's default writer sometimes does not).
    pil_frames = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE,
                                              colors=256) for f in frames]
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=66,           # ms per frame ≈ 15 fps
        loop=0,                # infinite
        disposal=2,            # replace each frame
        optimize=False,        # optimize can break some players
    )
    print(f"  gif   → {out_path}  ({W}×{H}, {n} frames)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                     help="delete cached fits/data and rebuild from scratch")
    ap.add_argument("--still", action="store_true",
                     help="render the still PNG only")
    ap.add_argument("--mp4", action="store_true",
                     help="render the MP4 only")
    ap.add_argument("--gif", action="store_true",
                     help="render the GIF only")
    args = ap.parse_args()

    if args.regen and DATA.exists():
        shutil.rmtree(DATA)
    if not (args.still or args.mp4 or args.gif):
        args.still = args.mp4 = args.gif = True

    if not DATA.exists() or not any(DATA.glob("*.model")):
        print("[1/3] generating noisy data")
        write_csvs()
        write_grids()
        print("[2/3] fitting via gam CLI")
        fit_all()
        print("[3/3] predicting on dense grids")
        predict_all()
    else:
        print("[1/3] using cached data + grids")
        if not any((DATA / "grid_cyl.csv").parent.glob("grid_*")):
            write_grids()
        if not all((DATA / f"{n}.model").exists() for n, *_ in FITS):
            print("[2/3] some models missing — fitting")
            fit_all()
        if not all((DATA / f"{n}_pred.csv").exists() for n in PRED_MAP):
            print("[3/3] some predictions missing — predicting")
            predict_all()
        else:
            print("[2/3] all fits cached")
            print("[3/3] all predictions cached")

    shapes = build_shapes()
    base = SCRIPT.parent / "geometric_shapes_demo"
    if args.still:
        print("[render] still PNG (4000×3000)")
        render_still(shapes, base.with_suffix(".png"))
    if args.mp4:
        print("[render] MP4 (1920×1440 @ 30 fps, 4 s loop)")
        render_mp4(shapes, base.with_suffix(".mp4"))
    if args.gif:
        print("[render] GIF (1280×960, 60 frames, 4 s loop)")
        render_gif(shapes, base.with_suffix(".gif"))


if __name__ == "__main__":
    main()
