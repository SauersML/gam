"""Auxiliary-richness diagnostic for iVAE-style identifiability.

Reference: Khemakhem, I., Kingma, D., Monti, R., Hyvarinen, A. (2020).
"Variational autoencoders and nonlinear ICA: a unifying framework."
AISTATS 2020. arXiv:1907.04809; see also arXiv:2107.10098.

All numeric work (constant-column detection, discreteness check, joint
distinct-level counting, empirical Jacobian rank) lives in the Rust
``gam::identifiability::kernel`` module. This Python file is a thin wrapper that
turns the metrics dict into an :class:`IdentifiabilityReport` with
concrete violations and recommendations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._binding import rust_module
from ._report import IdentifiabilityReport

__all__ = ["check_aux_richness"]


def _as_2d(arr: Any, name: str) -> np.ndarray:
    """Coerce array-like to a contiguous f64 2-D array. 1-D becomes (N, 1)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be shape (N,) or (N, dim); got shape {a.shape}"
        )
    return np.ascontiguousarray(a)


def check_aux_richness(
    aux: Any,
    latents: Any,
    *,
    n_mixture_components: int | None = None,
) -> IdentifiabilityReport:
    """Check the iVAE auxiliary-richness preconditions via the Rust kernel.

    Parameters
    ----------
    aux : array-like, shape ``(N,)`` or ``(N, aux_dim)``
        Auxiliary covariate. Missing values are not allowed.
    latents : array-like, shape ``(N, latent_dim)``
        Fitted latents.
    n_mixture_components : int, optional
        If a discrete aux is used as a K-component mixture indicator, the
        diagnostic checks that aux attains ``>= K`` distinct levels.

    Returns
    -------
    IdentifiabilityReport
    """
    name = "aux_richness"
    theorem = "Khemakhem et al. 2020 (iVAE) Theorem 1"

    preconditions: dict[str, bool] = {}
    violations: list[str] = []
    recommendations: list[str] = []

    aux_arr = _as_2d(aux, "aux")
    z_arr = _as_2d(latents, "latents")

    if aux_arr.shape[0] != z_arr.shape[0]:
        preconditions["row_count_matches"] = False
        violations.append(
            f"aux has {aux_arr.shape[0]} rows but latents has {z_arr.shape[0]} rows."
        )
        recommendations.append(
            "Pass aux and latents from the same fit; both must align row-wise."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations,
        )

    rust = rust_module()
    metrics = rust.diagnostics_aux_richness(aux_arr, z_arr)

    aux_dim = int(metrics["aux_dim"])
    latent_dim = int(metrics["latent_dim"])

    # Precondition 1: aux fully observed.
    aux_observed = bool(metrics["aux_observed"])
    preconditions["aux_observed"] = aux_observed
    if not aux_observed:
        n_bad = int(metrics["n_nonfinite_aux"])
        violations.append(
            f"aux contains {n_bad} non-finite entr{'y' if n_bad == 1 else 'ies'}; "
            f"iVAE Theorem 1 requires the auxiliary to be fully observed."
        )
        recommendations.append(
            "Impute or exclude rows where aux is missing/NaN before fitting; "
            "the iVAE prior cannot condition on unobserved auxiliaries."
        )

    # Precondition 2: aux_dim >= latent_dim.
    dim_ok = aux_dim >= latent_dim
    preconditions["aux_dim_at_least_latent_dim"] = dim_ok
    if not dim_ok:
        violations.append(
            f"aux dimension {aux_dim} is less than latent dimension {latent_dim}; "
            f"the iVAE Jacobian-rank precondition cannot hold."
        )
        recommendations.append(
            f"Provide an auxiliary variable with at least {latent_dim} dimensions, "
            f"or reduce factor_dim to <= {aux_dim}."
        )

    # Precondition 3: no constant columns.
    constant_cols = [int(j) for j in metrics["constant_columns"]]
    no_constant = aux_observed and len(constant_cols) == 0
    preconditions["aux_varies_across_rows"] = no_constant
    if aux_observed and constant_cols:
        violations.append(
            f"aux column(s) {constant_cols} are constant across observations; "
            f"a constant aux carries no conditioning information."
        )
        recommendations.append(
            f"Drop constant aux column(s) {constant_cols} or replace with a covariate that varies."
        )

    # Precondition 4: discrete-level count vs n_mixture_components.
    discrete = bool(metrics["aux_is_discrete"])
    n_levels = int(metrics["n_distinct_levels"])
    K = int(n_mixture_components) if n_mixture_components is not None else int(latent_dim)
    if discrete and aux_observed:
        levels_ok = n_levels >= K
        preconditions["aux_has_at_least_K_levels"] = levels_ok
        if not levels_ok:
            violations.append(
                f"aux is discrete with {n_levels} distinct level(s); "
                f"a K={K}-component identifiability precondition requires >= {K}."
            )
            recommendations.append(
                f"Use an auxiliary with at least {K} distinct discrete levels, "
                f"or reduce the number of mixture components to <= {n_levels}."
            )

    # Precondition 5: empirical Jacobian rank == latent_dim.
    rank_estimated = bool(metrics["jacobian_rank_estimated"])
    rank_value = metrics["jacobian_rank"]
    if rank_estimated:
        rank = int(rank_value)
        rank_ok = rank >= latent_dim
        preconditions["jacobian_rank_full"] = rank_ok
        if not rank_ok:
            violations.append(
                f"Empirical Jacobian rank(d z / d aux) = {rank}, expected {latent_dim}; "
                f"the pushforward map is rank-deficient."
            )
            recommendations.append(
                "Add aux columns that drive the deficient latent directions, "
                "or shrink latent_dim to match the achievable Jacobian rank."
            )
    else:
        preconditions["jacobian_rank_full"] = False
        if aux_observed:
            need = max(aux_dim, latent_dim) + 1
            violations.append(
                f"Insufficient rows to estimate the empirical Jacobian rank "
                f"(need N >= {need})."
            )
            recommendations.append(
                f"Collect at least {need} rows before relying on iVAE identifiability."
            )

    details = {
        "aux_dim": aux_dim,
        "latent_dim": latent_dim,
        "discrete": discrete,
        "n_distinct_levels": n_levels,
    }
    return IdentifiabilityReport(
        name=name, theorem=theorem,
        preconditions=preconditions, violations=violations,
        recommendations=recommendations, details=details,
    )
