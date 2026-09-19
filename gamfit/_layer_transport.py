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
fit_transport
    Fit one transport map and return a live, invertible ``FittedTransport``
    object (``eval``/``derivative``/``eval_with_variance``/``invert`` + report)
    rather than the summary dict, so callers can form ``g_B o g_A^-1``.
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


def fit_transport(
    coords_from: Any,
    coords_to: Any,
    topology_from: str = "circle",
    topology_to: str = "circle",
) -> Any:
    """Fit a transport map and return a live, invertible ``FittedTransport``.

    Unlike :func:`layer_transport_fit` (which returns a summary dict), this
    returns the map itself. The returned object exposes ``eval(t)``,
    ``derivative(t)``, ``eval_with_variance(t)``, and ``invert(y)`` (each over a
    1-D coordinate array), a ``report(layer_from=0, layer_to=1)`` dict, and the
    summary properties ``topology_from``/``topology_to``/``topology_preserved``/
    ``degree``/``isometry_defect``. ``invert`` is the piece that lets a caller
    compose ``g_B o g_A^-1`` from two fitted transports.

    ``coords_from[i]`` and ``coords_to[i]`` must coordinatize the same
    observation in the source and target charts; topologies are ``"circle"`` or
    ``"interval"``.
    """
    return rust_module().fit_transport(
        _as_coord_array(coords_from, "coords_from"),
        _as_coord_array(coords_to, "coords_to"),
        topology_from,
        topology_to,
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


def _as_operator_stack(arr: Any, name: str) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3-D [n_tokens, ambient, d], got shape {a.shape}")
    return a


def chart_transfer_operator(
    output_chart_jets: Any,
    ambient_jvps: Any,
    weights: Any = None,
) -> dict[str, Any]:
    """Density-aggregated pulled-back transfer operator ``A_kj`` across layers.

    The mechanistic "equation of motion" for a fitted atom: how the frozen
    model carries the atom's chart from an input layer ``j`` to an output layer
    ``k``. Per token,
    ``A_kj(x) = (J_k^T J_k)^-1 J_k^T J_F(x) J_j(x)`` — the model Jacobian-vector
    product ``J_F(x) J_j(x)`` (supplied already applied, from the torch lane)
    projected onto the output atom's decoder frame ``J_k(x)``.

    ``output_chart_jets`` is ``[n_tokens, ambient, d_out]`` (the output atom's
    decoder frame — for a fitted circle, its 2-D plane frame at each token).
    ``ambient_jvps`` is ``[n_tokens, ambient, d_in]`` (``J_F(x) J_j(x)``).
    ``weights`` is an optional ``[n_tokens]`` non-negative token density.

    Returns ``{"mean", "variance", "effective_n", "token_operators"}``: the
    density-weighted operator, its token-spread band, Kish's effective sample
    size, and the per-token operators.
    """
    jets = _as_operator_stack(output_chart_jets, "output_chart_jets")
    jvps = _as_operator_stack(ambient_jvps, "ambient_jvps")
    w = None
    if weights is not None:
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    return rust_module().chart_transfer_operator(jets, jvps, w)


def certify_chart_transfer(
    operator: Any,
    input_generator: Any,
    output_generator: Any,
) -> dict[str, Any]:
    """Transport / Lie-equivariance certificate for a square transfer operator.

    ``transport_defect = ||A^T A - I||_F`` (near-zero => the layer *carries* the
    atom as a near-isometry; large => it *computes* — rotates/rescales).
    ``equivariance_defect = ||A G_in - G_out A||_F`` tests whether ``A`` commutes
    with the charts' infinitesimal symmetry generators (for a circle,
    ``G = [[0,-1],[1,0]]``); near-zero certifies the cyclic structure is
    preserved. Returns ``{"transport_defect", "equivariance_defect"}``.
    """
    op = np.ascontiguousarray(operator, dtype=np.float64)
    gin = np.ascontiguousarray(input_generator, dtype=np.float64)
    gout = np.ascontiguousarray(output_generator, dtype=np.float64)
    return rust_module().certify_chart_transfer(op, gin, gout)
