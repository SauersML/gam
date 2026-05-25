"""Auxiliary-richness diagnostic for iVAE-style identifiability.

Reference: Khemakhem, I., Kingma, D., Monti, R., Hyvarinen, A. (2020).
"Variational autoencoders and nonlinear ICA: a unifying framework."
AISTATS 2020. arXiv:1907.04809; see also arXiv:2107.10098.

The iVAE identifiability theorem (Khemakhem 2020 Theorem 1) requires the
auxiliary covariate ``u`` to induce a sufficiently rich conditional family
``p(z | u)`` over latents ``z``. In practice that requires three things:

1. ``u`` is fully observed across rows (no missing values).
2. The dimension of ``u`` is at least the latent dimension (necessary
   for the Jacobian-rank precondition; not sufficient).
3. If ``u`` is discrete, it must take at least as many distinct values as
   the number of latent components / mixture components.
4. The pushforward Jacobian ``∂ E[z | u] / ∂ u`` evaluated at the fitted
   posterior latents must have rank equal to ``latent_dim``. We estimate
   this rank from the empirical Jacobian of the regression
   ``latents ~ aux``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._report import IdentifiabilityReport

__all__ = ["check_aux_richness"]


def _is_discrete(u: np.ndarray) -> bool:
    """Heuristic: treat aux as discrete if every column has <= 64 unique values
    AND every entry is integer-valued. Otherwise continuous."""
    if u.size == 0:
        return False
    if not np.all(np.isfinite(u)):
        return False
    if not np.all(u == np.round(u)):
        return False
    for j in range(u.shape[1]):
        if len(np.unique(u[:, j])) > 64:
            return False
    return True


def _jacobian_rank(aux: np.ndarray, latents: np.ndarray, tol: float | None = None) -> int:
    """Rank of the linear regression ``latents = aux @ B + c`` Jacobian.

    The fitted slope ``B`` (shape ``(aux_dim, latent_dim)``) *is* the
    ``∂z/∂u`` Jacobian for the linear-Gaussian limit of the iVAE prior.
    For non-linear models this is a first-order surrogate; the diagnostic
    flags rank deficiency that no amount of nonlinear post-processing can
    fix.
    """
    a = aux - aux.mean(axis=0, keepdims=True)
    z = latents - latents.mean(axis=0, keepdims=True)
    # Solve least-squares for B
    b_hat, _residuals, _rank, _sv = np.linalg.lstsq(a, z, rcond=None)
    return int(np.linalg.matrix_rank(b_hat, tol=tol))


def check_aux_richness(
    aux: Any,
    latents: Any,
    *,
    n_mixture_components: int | None = None,
) -> IdentifiabilityReport:
    """Check the iVAE auxiliary-richness preconditions.

    Parameters
    ----------
    aux : array-like, shape ``(N,)`` or ``(N, aux_dim)``
        Auxiliary covariate that was supplied to the identifiable-factor
        recipe. Missing values are not allowed; the diagnostic flags
        non-finite entries.
    latents : array-like, shape ``(N, latent_dim)``
        Fitted latents (e.g. ``T_supervised`` from
        :class:`IdentifiableFactorFitResult`).
    n_mixture_components : int, optional
        If the prior is a finite mixture with ``K`` components, the
        diagnostic also verifies that the discrete aux variable attains at
        least ``K`` distinct levels. Defaults to ``None`` (no mixture
        precondition).

    Returns
    -------
    IdentifiabilityReport
    """
    name = "aux_richness"
    theorem = "Khemakhem et al. 2020 (iVAE) Theorem 1"

    aux_arr = np.asarray(aux, dtype=float)
    if aux_arr.ndim == 1:
        aux_arr = aux_arr.reshape(-1, 1)
    z_arr = np.asarray(latents, dtype=float)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(-1, 1)

    preconditions: dict[str, bool] = {}
    violations: list[str] = []
    recommendations: list[str] = []

    if aux_arr.ndim != 2 or z_arr.ndim != 2:
        # Pathological shape; emit a single violation and bail.
        preconditions["valid_shapes"] = False
        violations.append(
            f"aux must be shape (N,) or (N, aux_dim) and latents must be (N, latent_dim); "
            f"got aux={aux_arr.shape}, latents={z_arr.shape}."
        )
        recommendations.append(
            "Reshape aux to (N, aux_dim) and latents to (N, latent_dim) before calling."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations, recommendations=recommendations,
        )

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
            preconditions=preconditions, violations=violations, recommendations=recommendations,
        )

    aux_dim = aux_arr.shape[1]
    latent_dim = z_arr.shape[1]

    # Precondition 1: aux is fully observed.
    finite_ok = bool(np.all(np.isfinite(aux_arr)))
    preconditions["aux_observed"] = finite_ok
    if not finite_ok:
        n_bad = int(np.sum(~np.isfinite(aux_arr)))
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

    # Precondition 3: variation across rows (no constant axes).
    if finite_ok:
        col_std = aux_arr.std(axis=0)
        constant_cols = [int(j) for j in np.where(col_std <= 1e-12)[0]]
        no_constant = len(constant_cols) == 0
    else:
        no_constant = False
        constant_cols = []
    preconditions["aux_varies_across_rows"] = no_constant
    if not no_constant:
        violations.append(
            f"aux column(s) {constant_cols} are constant across observations; "
            f"a constant aux carries no conditioning information."
        )
        recommendations.append(
            f"Drop constant aux column(s) {constant_cols} or replace with a covariate that varies."
        )

    # Precondition 4: discrete aux has enough distinct levels.
    discrete = _is_discrete(aux_arr) if finite_ok else False
    K = int(n_mixture_components) if n_mixture_components is not None else int(latent_dim)
    if discrete and finite_ok:
        # Use the joint distinct levels across all aux columns.
        rows_as_tuples = {tuple(row) for row in aux_arr.tolist()}
        n_levels = len(rows_as_tuples)
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

    # Precondition 5: empirical Jacobian rank equals latent_dim.
    if finite_ok and np.all(np.isfinite(z_arr)) and aux_arr.shape[0] >= max(aux_dim, latent_dim) + 1:
        rank = _jacobian_rank(aux_arr, z_arr)
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
        if not finite_ok:
            # Already complained about non-finite aux; do not double-flag.
            pass
        else:
            violations.append(
                "Insufficient rows to estimate the empirical Jacobian rank "
                f"(need N >= {max(aux_dim, latent_dim) + 1})."
            )
            recommendations.append(
                f"Collect at least {max(aux_dim, latent_dim) + 1} rows before relying on iVAE identifiability."
            )

    details = {"aux_dim": aux_dim, "latent_dim": latent_dim, "discrete": discrete}
    return IdentifiabilityReport(
        name=name, theorem=theorem,
        preconditions=preconditions, violations=violations, recommendations=recommendations,
        details=details,
    )
