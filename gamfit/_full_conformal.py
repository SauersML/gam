"""Exact full-conformal prediction sets for canonical-link GLMs (issue #942).

Thin wrapper over the Rust core ``gam::inference::full_conformal``: the
certified predictor–corrector homotopy that assembles the *exact*
full-conformal prediction set for a canonical-link GLM (Bernoulli-logit or
Poisson-log). For each candidate response ``z`` the augmented penalized fit is
tracked (or cold-refit when the per-step third-derivative certificate refuses),
the ``n+1`` absolute-residual nonconformity scores are ranked, and ``z`` is
kept iff its conformal p-value exceeds ``alpha``.

The resulting set has finite-sample coverage ``≥ 1 − alpha`` under
exchangeability of the ``n+1`` rows — a guarantee split conformal cannot match
at small calibration ``n``, and one that no mature classification-conformal
tool (MAPIE, glmnet, mgcv) exposes for a GLM with a coverage certificate. The
honest-but-exact arithmetic lives entirely in Rust; this module only marshals
arrays across the FFI boundary.

Smoothing is FROZEN at the supplied penalty ``s_lambda`` (the honest ρ-re-
selection is the engine's separate certified layer), and unit prior weights are
required: a reweighted training row is not exchangeable with the test row, so
the coverage proof would not apply and the engine refuses rather than silently
mis-cover.

Functions
---------
glm_full_conformal
    The exact full-conformal prediction set for one test row of a canonical
    GLM, with the enumerated candidate p-values and the homotopy exactness
    diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def glm_full_conformal(
    design: Any,
    response: Any,
    s_lambda: Any,
    x_star: Any,
    family: str,
    alpha: float,
    *,
    candidates: Any | None = None,
) -> dict[str, Any]:
    """Exact full-conformal prediction set for one test row of a GLM.

    Parameters
    ----------
    design : array (n, p)
        Training design ``X`` in the fitted coefficient basis.
    response : array (n,)
        Training response ``y``: ``{0, 1}`` for Bernoulli, non-negative integer
        counts for Poisson.
    s_lambda : array (p, p)
        Penalty matrix ``S_λ`` (``≥ 0``); pass a zero matrix for an unpenalized
        GLM. Smoothing is frozen at this value.
    x_star : array (p,)
        The test design row ``x_*`` whose prediction set is computed.
    family : {"bernoulli", "poisson"}
        Canonical-link family (logit / log). ``"binomial"`` and ``"logit"`` are
        accepted aliases for ``"bernoulli"``.
    alpha : float
        Target miscoverage in ``(0, 1)``; the set covers at ``≥ 1 − alpha``.
    candidates : array, optional
        Strictly increasing response candidates to test. Defaults to ``[0, 1]``
        (the exhaustive Bernoulli support). Poisson REQUIRES an explicit
        integer count window — there is no honest unbounded enumeration.

    Returns
    -------
    dict
        ``{"members", "p_values", "candidates", "alpha", "n_augmented",
        "refit_fallbacks", "margin_refits", "ties_unresolved",
        "max_beta_error_bound"}``. ``members`` is the retained prediction set;
        ``p_values`` / ``candidates`` are the per-candidate diagnostics;
        ``max_beta_error_bound`` is the largest certified ``‖β − β̂(z)‖`` over
        the tracked candidates — the homotopy exactness witness (``0`` when
        every candidate was cold-fit).
    """
    from ._binding import rust_module

    design_arr = np.ascontiguousarray(design, dtype=np.float64)
    if design_arr.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {design_arr.shape}")
    response_arr = np.ascontiguousarray(response, dtype=np.float64)
    if response_arr.ndim != 1:
        raise ValueError(f"response must be 1-D, got shape {response_arr.shape}")
    s_lambda_arr = np.ascontiguousarray(s_lambda, dtype=np.float64)
    if s_lambda_arr.ndim != 2:
        raise ValueError(f"s_lambda must be 2-D, got shape {s_lambda_arr.shape}")
    x_star_arr = np.ascontiguousarray(x_star, dtype=np.float64)
    if x_star_arr.ndim != 1:
        raise ValueError(f"x_star must be 1-D, got shape {x_star_arr.shape}")
    candidates_list = (
        None
        if candidates is None
        else [float(z) for z in np.asarray(candidates, dtype=np.float64).ravel()]
    )
    return rust_module().glm_full_conformal(
        design_arr,
        response_arr,
        s_lambda_arr,
        x_star_arr,
        str(family),
        float(alpha),
        candidates_list,
    )
