"""Functorial inter-layer concept transport maps (issue #1013).

Thin wrapper over the Rust core ``gam::inference::layer_transport``: all math
(REML smoothing, winding-degree estimation, isometry defect, gauge-quotiented
composition-law testing) lives in Rust; this module only marshals numpy arrays
across the FFI boundary.

Functions
---------
layer_transport_fit
    Fit one transport map ``t_to = h(t_from)`` between two chart
    coordinatizations of the same rows and return the evidence payload.
layer_transport_ladder
    Fit every adjacent and two-hop transport map across a ladder of layers,
    with the composition law ``h_{l->l+2} =? h_{l+1->l+2} o h_{l->l+1}``
    tested per triple.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._binding import rust_module


def _as_coord_array(coords: Any, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(coords, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def layer_transport_fit(
    coords_from: Any,
    coords_to: Any,
    topology_from: str = "circle",
    topology_to: str = "circle",
    *,
    layer_from: int = 0,
    layer_to: int = 1,
) -> dict[str, Any]:
    """Estimate the transport map between two layer charts with evidence.

    ``coords_from[i]`` and ``coords_to[i]`` must coordinatize the same
    observation in the source and target charts (circle coordinates in
    radians, any branch). Topologies are ``"circle"`` or ``"interval"``
    (interval bounds are derived from the coordinate range).

    Returns a dict with the winding ``degree`` (circle->circle),
    ``topology_preserved``, ``isometry_defect`` / ``isometry_defect_se``,
    ``transport_edf``, ``smoothing_lambda``, and friends.
    """
    return rust_module().layer_transport_fit(
        _as_coord_array(coords_from, "coords_from"),
        _as_coord_array(coords_to, "coords_to"),
        topology_from,
        topology_to,
        int(layer_from),
        int(layer_to),
    )


def layer_transport_ladder(
    coords: Sequence[Any],
    topology: str = "circle",
    *,
    layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit a whole ladder of layer charts and test the composition law.

    ``coords`` is a sequence of equal-length 1-D coordinate arrays (one per
    layer, same rows). Returns ``{"adjacent": [...], "two_hop": [...]}``
    where each two-hop report carries ``composition_defect`` /
    ``composition_p_value`` from the gauge-quotiented defect test.
    """
    arrays = [_as_coord_array(c, f"coords[{i}]") for i, c in enumerate(coords)]
    layer_list = None if layers is None else [int(v) for v in layers]
    return rust_module().layer_transport_ladder(arrays, topology, layer_list)
