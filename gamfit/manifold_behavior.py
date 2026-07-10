"""Thin public facade for the behavior-anchored manifold-SAE fit (#2015)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._binding import rust_module


def sae_behavior_fit(
    activation: Any,
    probabilities: Any,
    *,
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
) -> Any:
    """Fit joint activation/behavior atoms through one converged REML objective.

    Parameters
    ----------
    activation
        ``(N, P_x)`` activation response (the anchor block).
    probabilities
        Row-aligned ``(N, V)`` behavioral distributions; every row must be a
        probability vector. They are embedded through the exact sphere-tangent
        (nats-unit) chart and fit as the second output block, with the coupling
        weight ``log(lambda_y)`` selected by the same outer REML run.

    Returns
    -------
    ManifoldBehaviorCore
        Rust-owned report. ``to_dict()`` includes ``log_lambda_y``, the weight
        identifiability certificate, fitted probabilities, the honest KL
        summary, and per-atom isometry/behavior-pinned-chart certificates.
    """
    activation_array = np.ascontiguousarray(np.asarray(activation, dtype=np.float64))
    probability_array = np.ascontiguousarray(np.asarray(probabilities, dtype=np.float64))
    return rust_module().sae_behavior_fit(
        activation_array,
        probability_array,
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
    )


__all__ = ["sae_behavior_fit"]
