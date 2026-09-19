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
    random_state: int | None = None,
) -> Any:
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
    ManifoldCrosscoderCore
        Rust-owned fitted model. ``to_dict()`` materializes the shared report;
        ``steer_layer_delta`` and ``steer_layer_decode`` apply shared-coordinate
        interventions in any fitted layer's honest units.
    """
    anchor_array = np.ascontiguousarray(np.asarray(anchor, dtype=np.float64))
    labels = [str(label) for label, _ in targets]
    arrays = [
        np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        for _, values in targets
    ]
    return rust_module().sae_crosscoder_fit(
        anchor_array,
        str(anchor_label),
        labels,
        arrays,
        int(n_atoms),
        int(n_harmonics),
        random_state,
    )


__all__ = ["sae_crosscoder_fit"]
