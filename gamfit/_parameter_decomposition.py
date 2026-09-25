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

__all__ = ["ParameterDecompositionReport", "robust_support", "run_parameter_decomposition"]


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


def robust_support(
    objective: Callable[[np.ndarray], tuple[float, float, np.ndarray, float]],
    generators: Any,
    lower: Any,
    upper: Any,
    piece_bits: int,
    epsilon: float,
    gradient_lipschitz: float | None = None,
    ranking: Any = None,
) -> dict[str, Any]:
    """The minimum-code robust support at one input, by counterexample-guided search.

    ``objective(mask)`` executes the teacher at a float64 mask of length ``C`` and returns
    ``(divergence, its roundoff bound, d divergence / d moment (length K), its roundoff
    bound)``. ``generators`` is ``C x K`` (the moment rows ``v_c``), or ``None`` for literal pieces whose
    moment is the deletion vector itself (``K = C``); ``lower``/``upper``
    declare each control's mask interval (containing 1); ``piece_bits`` is the longest decoded body of one piece,
    charged for every kept piece (a local explanation carries what it keeps); ``epsilon`` is the declared
    tolerance. The search, the separation ascent over the moment zonotope and every
    certificate are Rust's (``supports::minimum_code_support`` over
    ``adversary::ZonotopeSeparationOracle``). With ``ranking`` (controls, most important first) the
    search is the shortest sufficient leading run of it, by bisection (``supports::ranked_support``).
    """
    return dict(
        rust_module().parameter_decomposition_robust_support(
            objective,
            None if generators is None else np.ascontiguousarray(generators, dtype=np.float64),
            np.ascontiguousarray(lower, dtype=np.float64),
            np.ascontiguousarray(upper, dtype=np.float64),
            int(piece_bits),
            float(epsilon),
            gradient_lipschitz,
            None if ranking is None else [int(c) for c in ranking],
        )
    )
