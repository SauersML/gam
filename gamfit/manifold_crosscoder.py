"""Thin public facade for the unified-engine manifold crosscoder."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._binding import rust_module


def sae_crosscoder_fit(
    anchor: Any,
    targets: Sequence[tuple[str, Any]],
    *,
    anchor_label: str = "anchor",
    n_atoms: int = 8,
    n_harmonics: int = 3,
    sparsity_strength: float = 1.0,
    smoothness: float = 1.0,
    max_iter: int = 50,
    learning_rate: float = 0.05,
    ridge_ext_coord: float = 1.0e-6,
    ridge_beta: float = 1.0e-6,
    random_state: int = 0,
    run_outer_rho_search: bool = True,
    grid_resolution: int = 256,
    law_gap_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Fit one shared-chart manifold dictionary across row-aligned layers.

    Parameters
    ----------
    anchor
        The leading ``(N, P_anchor)`` activation target.
    targets
        Ordered ``[(label, array), ...]`` non-anchor targets. Every array must
        carry the same rows as ``anchor``; widths may differ, although drift and
        phase-transport measurements are defined only between equal-width
        consecutive layers.

    Returns
    -------
    dict
        The Rust-owned fit report: selected block relevance, honest-unit layer
        reconstructions/decoders, shared assignments and coordinates,
        cross-layer drift, and per-atom phase-transport measurements.
    """
    anchor_array = np.ascontiguousarray(np.asarray(anchor, dtype=np.float64))
    if anchor_array.ndim != 2:
        raise ValueError(f"anchor must be a 2-D array; got shape {anchor_array.shape}")
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for index, item in enumerate(targets):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(
                f"targets[{index}] must be a (label, array) pair; got {type(item).__name__}"
            )
        label, values = item
        labels.append(str(label))
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        if array.ndim != 2:
            raise ValueError(
                f"targets[{index}] ({label!r}) must be 2-D; got shape {array.shape}"
            )
        arrays.append(array)
    return dict(
        rust_module().sae_crosscoder_fit(
            anchor_array,
            str(anchor_label),
            labels,
            arrays,
            int(n_atoms),
            int(n_harmonics),
            float(sparsity_strength),
            float(smoothness),
            int(max_iter),
            float(learning_rate),
            float(ridge_ext_coord),
            float(ridge_beta),
            int(random_state),
            bool(run_outer_rho_search),
            int(grid_resolution),
            float(law_gap_tolerance),
        )
    )


__all__ = ["sae_crosscoder_fit"]
