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
from typing import Any, Mapping

import numpy as np

from ._binding import rust_module

__all__ = ["ParameterDecompositionReport", "run_parameter_decomposition"]


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
    :class:`gamfit.GamError`. JSON has no non-finite numbers, so a request
    holding one is rejected by :func:`json.dumps` before Rust is called.
    """
    report_json, arrays = rust_module().parameter_decomposition_run(
        json.dumps(request, allow_nan=False),
        {name: np.ascontiguousarray(values, dtype=np.float64) for name, values in tensors.items()},
    )
    return ParameterDecompositionReport(report=json.loads(report_json), arrays=dict(arrays))
