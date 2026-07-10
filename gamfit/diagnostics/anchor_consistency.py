"""Anchor-consistency diagnostic for manifold-SAE atom identifiability.

Reference: the project's own atom-anchor convention (see
``gamfit._sae_manifold`` / ``gam::identifiability::kernel::anchor_consistency_report``).
A manifold-SAE with ``K`` atoms is identified up to permutation of atoms
only when the assignment matrix ``A in R^{N x K}`` contains a sufficient
number of *anchor* rows — rows where one atom dominates
(``a_max / sum|a| >= anchor_dominance``). Too few anchors and the recovered
atoms are only identified up to a linear transformation in atom space.

The typed diagnostic report, including every pass/fail verdict, lives in the
Rust ``gam::identifiability::kernel`` module; this Python file only extracts
the assignment matrix and presents that report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._binding import rust_module
from ._report import IdentifiabilityReport

__all__ = ["check_anchor_consistency"]


def _extract_assignments(model: Any) -> np.ndarray:
    """Return the ``(N, K)`` assignment matrix from a manifold-SAE-like model."""
    if not hasattr(model, "assignments"):
        raise TypeError(
            f"Cannot extract atom assignments from object of type "
            f"{type(model).__name__}; expected a `.assignments` attribute "
            f"with shape (N, K)."
        )
    assignments = np.asarray(model.assignments, dtype=float)
    if assignments.ndim != 2:
        raise ValueError(
            "assignments must be a 2-D (N, K) matrix; "
            f"got shape {assignments.shape}."
        )
    return np.ascontiguousarray(assignments)


def check_anchor_consistency(
    model: Any,
    *,
    anchor_dominance: float | None = None,
) -> IdentifiabilityReport:
    """Check that a fitted manifold-SAE has enough anchor points per atom.

    Parameters
    ----------
    model : object with ``.assignments`` of shape ``(N, K)``
    anchor_dominance : float in ``(0.5, 1]``, optional
        A row is an *anchor* if its largest entry contributes at least this
        fraction of the row's L1 mass. When omitted, the shared Rust core
        convention is used.

    Returns
    -------
    IdentifiabilityReport
    """
    name = "anchor_consistency"
    theorem = "gam manifold-SAE atom-anchor convention"

    A = _extract_assignments(model)

    rust = rust_module()
    threshold = None if anchor_dominance is None else float(anchor_dominance)
    core = rust.diagnostics_anchor_consistency_report(A, threshold)
    return IdentifiabilityReport(
        name=name,
        theorem=theorem,
        preconditions={
            str(key): bool(value)
            for key, value in core["preconditions"].items()
        },
        violations=[str(value) for value in core["violations"]],
        recommendations=[str(value) for value in core["recommendations"]],
        details=dict(core["details"]),
    )
