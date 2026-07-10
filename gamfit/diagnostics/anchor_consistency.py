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
    if hasattr(model, "assignments"):
        A = np.ascontiguousarray(np.asarray(model.assignments, dtype=float))
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        return np.ascontiguousarray(A)
    raise TypeError(
        f"Cannot extract atom assignments from object of type "
        f"{type(model).__name__}; expected a `.assignments` attribute "
        f"with shape (N, K)."
    )


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

    try:
        A = _extract_assignments(model)
    except TypeError as exc:
        return IdentifiabilityReport(
            name=name,
            theorem=theorem,
            preconditions={"assignments_available": False},
            violations=[str(exc)],
            recommendations=[
                "Pass a manifold-SAE model (gamfit.ManifoldSAE) or any object "
                "exposing an (N, K) `assignments` ndarray."
            ],
        )

    if A.ndim != 2 or A.shape[1] < 1:
        return IdentifiabilityReport(
            name=name,
            theorem=theorem,
            preconditions={
                "assignments_available": True,
                "assignments_2d": False,
            },
            violations=[
                "assignments must be 2D with at least one atom column; "
                f"got shape {A.shape}."
            ],
            recommendations=[
                "Refit the model so that `assignments` is an (N, K) ndarray."
            ],
        )

    rust = rust_module()
    threshold = None if anchor_dominance is None else float(anchor_dominance)
    core = rust.diagnostics_anchor_consistency_report(A, threshold)
    preconditions = {
        "assignments_available": True,
        "assignments_2d": True,
        **{str(key): bool(value) for key, value in core["preconditions"].items()},
    }
    return IdentifiabilityReport(
        name=name,
        theorem=theorem,
        preconditions=preconditions,
        violations=[str(value) for value in core["violations"]],
        recommendations=[str(value) for value in core["recommendations"]],
        details=dict(core["details"]),
    )
