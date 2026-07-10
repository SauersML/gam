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
    n_atoms: int,
    n_harmonics: int,
    sparsity_strength: float | None = None,
    smoothness: float | None = None,
    max_iter: int | None = None,
    learning_rate: float | None = None,
    ridge_ext_coord: float | None = None,
    ridge_beta: float | None = None,
    random_state: int | None = None,
    run_outer_rho_search: bool | None = None,
    transport_grid_resolution: int | None = None,
    law_gap_tolerance: float | None = None,
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
    labels = [str(label) for label, _ in targets]
    arrays = [
        np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        for _, values in targets
    ]
    return dict(
        rust_module().sae_crosscoder_fit(
            anchor_array,
            str(anchor_label),
            labels,
            arrays,
            int(n_atoms),
            int(n_harmonics),
            sparsity_strength,
            smoothness,
            max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            random_state,
            run_outer_rho_search,
            transport_grid_resolution,
            law_gap_tolerance,
        )
    )


__all__ = ["sae_crosscoder_fit"]
