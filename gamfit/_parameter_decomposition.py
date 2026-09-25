"""Manifold parameter decomposition (#2951) — a thin adapter over the Rust surface.

``gam_sae::parameter_decomposition::surface`` owns the versioned request
document, its validation, every operation and the report. This module only
serializes the request mapping to JSON text, passes each named array as
contiguous float64, and returns the report with the arrays it names. ``gam
parameter-decomposition`` runs the same Rust entry, so the CLI writes the same
report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from ._binding import rust_module

__all__ = ["ParameterDecompositionReport", "fit_supports", "minimal_support", "run_parameter_decomposition"]


@dataclass(frozen=True)
class ParameterDecompositionReport:
    """A ``gam.mpd-report`` document and the arrays it names.

    Attributes
    ----------
    report
        The parsed report document. Array-valued results appear as ids.
    arrays
        ``{id: ndarray}`` for every array id the report names.
    """

    report: dict[str, Any]
    arrays: dict[str, np.ndarray]


def run_parameter_decomposition(
    request: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> ParameterDecompositionReport:
    """Run one ``gam.mpd-request`` document against its named input arrays.

    Every validation and refusal happens in Rust; a refused request raises
    :class:`gamfit.errors.GamfitError`. JSON has no non-finite numbers, so a request
    holding one is rejected by :func:`json.dumps` before Rust is called.
    """
    report_json, arrays = rust_module().parameter_decomposition_run(
        json.dumps(request, allow_nan=False),
        {name: np.ascontiguousarray(values, dtype=np.float64) for name, values in tensors.items()},
    )
    return ParameterDecompositionReport(report=json.loads(report_json), arrays=dict(arrays))


def minimal_support(
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    positions: int,
    pieces: int,
    eps: float,
    form: str,
    sequence: int,
    start: np.ndarray | None = None,
) -> dict[str, Any]:
    """Smallest per-position supports at fidelity ``eps``, searched by the Rust owner.

    ``evaluate(keep)`` receives a ``positions x pieces`` bool array (False = the piece
    is removed at that position), runs the model, and returns ``(divergence,
    removal_cost, restore_gain)``: float64 ``KL_t`` per position, the predicted increase
    of removing each kept piece and the predicted decrease of restoring each removed one. The search, its acceptance rule and its
    stopping rule are ``gam_sae::parameter_decomposition::minimal_support``'s; this
    function only forwards the call. ``form`` declares what admissible means: ``"per_position"`` (every
    position's divergence at most ``eps``) or ``"mean"`` (the batch mean, the form a batch-mean metric reports).
    Positions come in causal sequences of length ``sequence`` (a position's
    divergence depends on removals at it and at earlier positions of its sequence; 1 for independent examples),
    which decides which proposals a violation halves. ``start`` (``positions x pieces`` bool) warm-starts the search; an inadmissible start is
    first repaired by restoring pieces where violations can come from. Returns ``{"keep", "divergence", "rounds"}``.
    """
    return dict(rust_module().parameter_decomposition_minimal_support(
        evaluate, positions, pieces, float(eps), str(form), int(sequence), None if start is None else np.ascontiguousarray(start, dtype=bool)))


def fit_supports(executor: Any, theta: np.ndarray, positions: int, pieces: int, eps: float, form: str,
                 sequence: int, frames: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    """Alternate minimal supports and piece reshaping, by the Rust owner.

    ``executor`` runs the model only: methods ``supports(theta, keep)`` (as ``minimal_support``'s
    ``evaluate``), ``divergence(theta, keep)`` (``KL_t`` per position), ``weighted_gradient(theta, keep,
    weights)`` (``sum_t w_t grad KL_t``), ``directional(theta, keep, v)`` (``grad KL_t . v`` per position),
    ``weighted_hessian(theta, keep, weights, v)`` (the exact ``sum_t w_t Hess KL_t v``), ``gradient_arithmetic()``
    and ``observe(alternation)``; every derivative is Euclidean, in ``theta``'s coordinates. ``form`` is
    ``minimal_support``'s: ``"per_position"`` (the barrier ``-sum_t log(eps - KL_t)``) or ``"mean"`` (the mean
    divergence itself). ``frames``, when given, is a list of ``(rows, cols)`` blocks covering ``theta`` in order, each
    a matrix with orthonormal columns stored row-major: the pieces are tight frames and ``theta`` moves on that
    product of Stiefel manifolds (``gam_geometry::StiefelFrames``); without it ``theta`` is Euclidean. The trust
    region, the fidelity path from the empty support down to ``eps``, and the stopping rule are
    ``gam_sae::parameter_decomposition::support_fit``'s. Returns ``{"theta", "keep", "divergence",
    "alternations"}``.
    """
    return dict(rust_module().parameter_decomposition_fit_supports(
        executor, np.ascontiguousarray(theta, dtype=np.float64), positions, pieces, float(eps), str(form), int(sequence),
        None if frames is None else [(int(r), int(c)) for r, c in frames]))
