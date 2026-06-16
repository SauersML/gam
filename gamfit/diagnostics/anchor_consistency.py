"""Anchor-consistency diagnostic for manifold-SAE atom identifiability.

Reference: the project's own atom-anchor convention (see
``gamfit._sae_manifold`` / ``gam::identifiability::kernel::anchor_consistency_metrics``).
A manifold-SAE with ``K`` atoms is identified up to permutation of atoms
only when the assignment matrix ``A in R^{N x K}`` contains a sufficient
number of *anchor* rows — rows where one atom dominates
(``a_max / sum|a| >= anchor_dominance``). Too few anchors and the recovered
atoms are only identified up to a linear transformation in atom space.

The numeric kernel (anchor counts per atom, total anchor count) lives in
the Rust ``gam::identifiability::kernel`` module; this Python file is a thin
extraction-and-reporting wrapper.
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
    anchor_dominance: float = 0.95,
) -> IdentifiabilityReport:
    """Check that a fitted manifold-SAE has enough anchor points per atom.

    Parameters
    ----------
    model : object with ``.assignments`` of shape ``(N, K)``
    anchor_dominance : float in ``(0, 1]``, default ``0.95``
        A row is an *anchor* if its largest entry contributes at least this
        fraction of the row's L1 mass.

    Returns
    -------
    IdentifiabilityReport
    """
    name = "anchor_consistency"
    theorem = "gam manifold-SAE atom-anchor convention"

    preconditions: dict[str, bool] = {}
    violations: list[str] = []
    recommendations: list[str] = []

    try:
        A = _extract_assignments(model)
    except TypeError as exc:
        preconditions["assignments_available"] = False
        violations.append(str(exc))
        recommendations.append(
            "Pass a manifold-SAE model (gamfit.ManifoldSAE) or any object "
            "exposing an (N, K) `assignments` ndarray."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations,
        )

    if A.ndim != 2 or A.shape[1] < 1:
        preconditions["assignments_2d"] = False
        violations.append(
            f"assignments must be 2D with at least one atom column; got shape {A.shape}."
        )
        recommendations.append(
            "Refit the model so that `assignments` is an (N, K) ndarray."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations,
        )
    preconditions["assignments_available"] = True
    preconditions["assignments_2d"] = True

    n, K = A.shape

    if not (0.0 < float(anchor_dominance) <= 1.0):
        raise ValueError(
            f"anchor_dominance must be in (0, 1]; got {anchor_dominance!r}"
        )

    rust = rust_module()
    metrics = rust.diagnostics_anchor_consistency(A, float(anchor_dominance))
    n_anchors = int(metrics["n_anchors"])
    anchors_per_atom = [int(c) for c in metrics["anchors_per_atom"]]

    if K == 1:
        preconditions["enough_anchors_total"] = True
        preconditions["anchors_cover_all_atoms"] = True
        details = {
            "n_samples": n, "K": K, "n_anchors": n_anchors,
            "anchor_fraction": float(n_anchors) / max(n, 1),
            "anchors_per_atom": anchors_per_atom,
        }
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations, details=details,
        )

    enough_anchors = n_anchors >= K
    preconditions["enough_anchors_total"] = enough_anchors
    if not enough_anchors:
        violations.append(
            f"Only {n_anchors} anchor row(s) (dominance >= {anchor_dominance:.2f}) "
            f"found in a K={K}-atom model; need at least {K}. The recovered "
            f"atoms are identified only up to a linear transformation in atom space."
        )
        recommendations.append(
            f"Reduce K to <= {max(n_anchors, 1)}, sharpen the assignment prior "
            f"(e.g. lower temperature / stronger IBP concentration), or collect "
            f"more anchor-like rows where a single atom dominates."
        )

    uncovered = [int(j) for j, c in enumerate(anchors_per_atom) if c == 0]
    cover_ok = len(uncovered) == 0
    preconditions["anchors_cover_all_atoms"] = cover_ok
    if not cover_ok:
        violations.append(
            f"Atom(s) {uncovered} have zero anchor rows; they are not "
            f"individually identifiable and may be redundant or merged with neighbours."
        )
        recommendations.append(
            f"Prune the {len(uncovered)} uncovered atom(s) (refit with "
            f"K={K - len(uncovered)}) or strengthen the per-atom sparsity prior "
            f"so that each atom acquires a dominant region."
        )

    details = {
        "n_samples": n, "K": K, "n_anchors": n_anchors,
        "anchor_fraction": float(n_anchors) / max(n, 1),
        "anchors_per_atom": anchors_per_atom,
    }
    return IdentifiabilityReport(
        name=name, theorem=theorem,
        preconditions=preconditions, violations=violations,
        recommendations=recommendations, details=details,
    )
