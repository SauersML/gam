"""Matplotlib visualization helpers for fitted SAE manifold atoms."""

from __future__ import annotations

from math import ceil, sqrt
from typing import Any
from collections.abc import Mapping

import numpy as np

from ._matplotlib import pyplot


def plot_atom(fit: Any, k: int, ax: Any = None) -> Any:
    """Plot one SAE manifold atom in its leading decoder SVD subspace.

    Parameters
    ----------
    fit
        A fitted :class:`gamfit.sae.ManifoldSAE`-like object with ``atoms``.
    k
        Atom index.
    ax
        Optional Matplotlib axis. If omitted, a new figure and suitable 2D/3D
        axis are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    atom = _atom_at(fit, k)
    decoder = _as_2d(atom.decoder_coefficients, "decoder_coefficients")
    basis = _basis_for(fit, k)
    topology = _topology_for(fit, k)

    shape = _shape_points(fit, atom, k, topology)
    token_points = _token_points(fit, atom, k)

    projector, plot_dim, rank = _decoder_projector(decoder)
    shape_proj = _project(shape, projector)
    token_proj = _project(token_points, projector)
    active = _active_weights(fit, atom, k, token_proj.shape[0])

    if ax is None:
        plt = pyplot()

        fig = plt.figure(figsize=(5.2, 4.2))
        if plot_dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    elif plot_dim == 3 and not hasattr(ax, "zaxis"):
        plot_dim = 2
        shape_proj = shape_proj[:, :2]
        token_proj = token_proj[:, :2]

    _draw_shape(ax, shape_proj, topology, plot_dim)
    _draw_tokens(ax, token_proj, active, plot_dim)
    _decorate_axis(ax, fit, atom, k, topology, basis, rank, plot_dim)
    return ax


def plot_fit(fit: Any) -> Any:
    """Plot all atoms in a compact grid and return the Matplotlib figure."""
    atoms = list(getattr(fit, "atoms", []))
    if not atoms:
        raise ValueError("plot_fit requires fit.atoms to contain at least one atom")

    plt = pyplot()

    n_atoms = len(atoms)
    ncols = min(3, max(1, ceil(sqrt(n_atoms))))
    nrows = ceil(n_atoms / ncols)
    fig = plt.figure(figsize=(4.6 * ncols, 4.0 * nrows), constrained_layout=True)
    for idx in range(n_atoms):
        atom = atoms[idx]
        decoder = _as_2d(atom.decoder_coefficients, "decoder_coefficients")
        _, plot_dim, _ = _decoder_projector(decoder)
        if plot_dim == 3:
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        else:
            ax = fig.add_subplot(nrows, ncols, idx + 1)
        plot_atom(fit, idx, ax=ax)
    return fig


def plot(target: Any, atom: int | None = None, *, ax: Any = None, color_by: str = "assignment") -> Any:
    """Plot an SAE atom or an atom inside a fitted SAE object.

    ``plot(fit, atom=0)`` renders the fitted atom shape in its leading decoder
    SVD subspace. ``plot(atom)`` renders a lightweight coordinate scatter for a
    standalone Rust atom object or atom dictionary.
    Matplotlib is imported only when this function is called.
    """
    if atom is not None or hasattr(target, "atoms"):
        return plot_atom(target, 0 if atom is None else int(atom), ax=ax)
    return _plot_standalone_atom(target, ax=ax, color_by=color_by)


def _plot_standalone_atom(atom: Any, *, ax: Any = None, color_by: str = "assignment") -> Any:
    plt = pyplot()

    coords = _as_2d(_atom_field(atom, "coords"), "coords")
    assignments = np.asarray(_atom_field(atom, "assignments"), dtype=float).reshape(-1)
    basis = str(_atom_field(atom, "basis"))
    if assignments.shape[0] != coords.shape[0]:
        raise ValueError(
            f"atom assignments length {assignments.shape[0]} does not match coords rows {coords.shape[0]}"
        )
    if color_by == "assignment":
        color = assignments
        color_label = "assignment"
    elif color_by == "index":
        color = np.arange(coords.shape[0], dtype=float)
        color_label = "row"
    else:
        raise ValueError("color_by must be 'assignment' or 'index'")
    if ax is None:
        _, ax = plt.subplots()
    if coords.shape[1] == 1:
        scatter = ax.scatter(coords[:, 0], assignments, c=color, cmap="viridis", s=18)
        ax.set_xlabel("coordinate 0")
        ax.set_ylabel("assignment")
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=18)
        ax.set_xlabel("coordinate 0")
        ax.set_ylabel("coordinate 1")
    ax.set_title(f"{basis} atom")
    plt.colorbar(scatter, ax=ax, label=color_label)
    return ax


def _atom_field(atom: Any, name: str) -> Any:
    if isinstance(atom, Mapping) and name in atom:
        return atom[name]
    if hasattr(atom, name):
        return getattr(atom, name)
    raise TypeError("atom must expose the SAE atom fields or be an atom dictionary")


def _atom_at(fit: Any, k: int) -> Any:
    atoms = list(getattr(fit, "atoms", []))
    idx = int(k)
    if idx < 0 or idx >= len(atoms):
        raise IndexError(f"atom index {k} out of range for K={len(atoms)}")
    return atoms[idx]


def _as_2d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return arr


def _geometry_plan_for(fit: Any, k: int) -> Mapping[str, Any]:
    plans = fit.geometry_plans
    idx = int(k)
    if idx < 0 or idx >= len(plans) or not isinstance(plans[idx], Mapping):
        raise ValueError(f"geometry_plans must contain a mapping for atom {idx}")
    return plans[idx]


def _basis_for(fit: Any, k: int) -> str:
    plan = _geometry_plan_for(fit, k)
    kind = plan.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"geometry_plans[{int(k)}].kind must be a non-empty string")
    return kind


def _topology_for(fit: Any, k: int) -> str:
    topologies = fit.atom_topologies
    if int(k) < 0 or int(k) >= len(topologies):
        raise ValueError(f"atom_topologies has no entry for atom {int(k)}")
    return str(topologies[int(k)])


def _shape_points(fit: Any, atom: Any, k: int, topology: str) -> np.ndarray:
    mean = getattr(atom, "shape_band_mean", None)
    if mean is not None:
        arr = _as_optional_2d(mean)
        if arr is not None and arr.shape[0] > 0:
            return arr
    grid = np.ascontiguousarray(_grid_for(atom, topology))
    return np.asarray(fit.atom_curve(int(k), grid), dtype=float)


def _token_points(fit: Any, atom: Any, k: int) -> np.ndarray:
    coords = np.ascontiguousarray(_as_2d(atom.coords, "coords"))
    return np.asarray(fit.atom_curve(int(k), coords), dtype=float)


def _grid_for(atom: Any, topology: str) -> np.ndarray:
    band_coords = getattr(atom, "shape_band_coords", None)
    arr = _as_optional_2d(band_coords)
    if arr is not None and arr.shape[0] > 0:
        return arr

    coords = _as_2d(atom.coords, "coords")
    d = coords.shape[1]
    if topology == "circle":
        return np.linspace(0.0, 1.0, 240, endpoint=True).reshape(-1, 1)
    if topology == "sphere":
        lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 32)
        lon = np.linspace(-np.pi, np.pi, 48)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        return np.column_stack([la.ravel(), lo.ravel()])
    if topology == "torus":
        axes = [np.linspace(0.0, 1.0, 28, endpoint=False) for _ in range(max(1, d))]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    lo, hi = _coordinate_bounds(coords)
    if d == 1:
        return np.linspace(lo[0], hi[0], 240).reshape(-1, 1)
    axes = [np.linspace(lo[j], hi[j], 34) for j in range(min(d, 2))]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.zeros((mesh[0].size, d), dtype=float)
    for j, values in enumerate(mesh):
        grid[:, j] = values.ravel()
    if d > 2:
        grid[:, 2:] = np.median(coords[:, 2:], axis=0)
    return grid


def _coordinate_bounds(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] == 0:
        return np.zeros(coords.shape[1]), np.ones(coords.shape[1])
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    width = np.maximum(hi - lo, 1.0e-6)
    return lo - 0.08 * width, hi + 0.08 * width


def _decoder_projector(decoder: np.ndarray) -> tuple[np.ndarray, int, int]:
    # Plot-axis selection only — no spectral decomposition. The former
    # np.linalg.svd principal-subspace projector was numeric linear algebra
    # computed in Python; a visualization does not need it. Project onto the
    # leading decoder output coordinates instead and report the ambient output
    # width as the (upper-bound) rank used purely for the plot label.
    p = int(decoder.shape[1])
    if p == 0:
        return np.zeros((0, 2)), 2, 0
    plot_dim = 3 if p >= 3 else 2
    keep = min(plot_dim, p)
    projector = np.eye(p, keep, dtype=float)
    if keep < plot_dim:
        pad = np.zeros((p, plot_dim - keep), dtype=float)
        projector = np.column_stack([projector, pad])
    return projector, plot_dim, p


def _project(points: np.ndarray, projector: np.ndarray) -> np.ndarray:
    if points.shape[1] != projector.shape[0]:
        width = min(points.shape[1], projector.shape[0])
        aligned = np.zeros((points.shape[0], projector.shape[0]), dtype=float)
        aligned[:, :width] = points[:, :width]
        points = aligned
    return points @ projector


def _active_weights(fit: Any, atom: Any, k: int, n: int) -> np.ndarray | None:
    assignments = getattr(atom, "assignments", None)
    if assignments is None:
        fit_assignments = getattr(fit, "assignments", None)
        arr = None if fit_assignments is None else np.asarray(fit_assignments, dtype=float)
        if arr is not None and arr.ndim == 2 and int(k) < arr.shape[1]:
            assignments = arr[:, int(k)]
    if assignments is None:
        return None
    weights = np.asarray(assignments, dtype=float).reshape(-1)
    if weights.size != n or not np.all(np.isfinite(weights)):
        return None
    return weights


def _draw_shape(ax: Any, points: np.ndarray, topology: str, plot_dim: int) -> None:
    if points.shape[0] == 0:
        return
    if plot_dim == 3:
        if topology in {"sphere", "torus", "euclidean"} and points.shape[0] > 3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.18, color="tab:blue")
        else:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], lw=1.6, color="tab:blue")
    else:
        if topology in {"sphere", "torus"}:
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.18, color="tab:blue")
        else:
            ax.plot(points[:, 0], points[:, 1], lw=1.8, color="tab:blue")


def _draw_tokens(ax: Any, points: np.ndarray, weights: np.ndarray | None, plot_dim: int) -> None:
    if points.shape[0] == 0:
        return
    color = "tab:orange" if weights is None else weights
    kwargs = {"s": 18, "alpha": 0.72, "color": color} if weights is None else {
        "s": 18,
        "alpha": 0.78,
        "c": color,
        "cmap": "viridis",
    }
    if plot_dim == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], **kwargs)


def _decorate_axis(
    ax: Any,
    fit: Any,
    atom: Any,
    k: int,
    topology: str,
    basis: str,
    rank: int,
    plot_dim: int,
) -> None:
    trust = _trust_for(fit, atom, k)
    title = f"atom {int(k)}: {topology}"
    if trust is not None:
        title += f" trust={trust:.3g}"
    ax.set_title(title)
    ax.set_xlabel("decoder output 1")
    ax.set_ylabel("decoder output 2")
    if plot_dim == 3:
        ax.set_zlabel("decoder output 3")
    text = f"basis={basis}\nrank={rank}"
    ax.text2D(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top") if plot_dim == 3 else ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    if plot_dim == 2:
        ax.set_aspect("equal", adjustable="datalim")


def _trust_for(fit: Any, atom: Any, k: int) -> float | None:
    for name in ("trust", "trust_score"):
        value = getattr(atom, name, None)
        if value is not None:
            return _finite_scalar(value)
    for name in ("atom_trust", "atom_trust_scores", "trust", "trust_scores"):
        value = getattr(fit, name, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if int(k) < arr.size:
            return _finite_scalar(arr[int(k)])
    return None


def _finite_scalar(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _as_optional_2d(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return _as_2d(value, "optional array")
    except ValueError:
        return None


__all__ = ["plot", "plot_atom", "plot_fit"]
