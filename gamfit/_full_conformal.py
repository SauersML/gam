"""Certified full-conformal prediction sets for GLM families (issue #942).

Thin wrapper over the Rust core ``gam::inference::full_conformal_glm``, the
same certified engine the predict route
(``model.predict(interval="conformal", training_data=...)``) uses. For each
candidate response the augmented penalized fit at the frozen penalty is solved
by certified Newton, the ``n + 1`` working-score nonconformity scores are
ranked, and the candidate is retained when its conformal p-value exceeds
``alpha`` or numerical uncertainty prevents certified exclusion.

* Bernoulli: both levels ``{0, 1}`` are tested.
* Poisson and negative binomial: the counts are enumerated up to a tail the
  data certify, beyond which no count can conform (the set is ``[0, inf)`` when
  no tail is provable).
* Gamma: the continuum is walked with certified refits.

The discrete families use one independent random tie variate per inversion.
Under exchangeability of the supplied rows and a fixed, symmetric fitting map,
the ideal smoothed rank set has marginal coverage ``1 - alpha``. The numerical
set is a conservative enclosure, so its coverage can be higher. This is not a
conditional-on-features guarantee. A basis or penalty learned only from the
training rows does not automatically satisfy the symmetry assumption.
Smoothing is frozen at ``s_lambda`` and prior weights are not supported.
All arithmetic lives in Rust; this module only marshals arrays.

Functions
---------
glm_full_conformal
    The certified full-conformal prediction set for one test row of a GLM.
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
    theta: float | None = None,
    offset: Any | None = None,
    offset_star: float = 0.0,
) -> dict[str, Any]:
    """Certified full-conformal prediction set for one test row of a GLM.

    Parameters
    ----------
    design : array (n, p)
        Training design ``X`` in the fitted coefficient basis.
    response : array (n,)
        Training response ``y``: ``{0, 1}`` for Bernoulli, non-negative integer
        counts for Poisson and negative binomial, positive for Gamma.
    s_lambda : array (p, p)
        Positive semidefinite penalty matrix ``S_λ`` in unit-dispersion units;
        pass a zero matrix for an unpenalized GLM. Smoothing is frozen at this
        value.
    x_star : array (p,)
        The test design row ``x_*`` whose prediction set is computed.
    family : {"bernoulli", "poisson", "negative_binomial", "gamma"}
        Family with its canonical/log link (logit for Bernoulli, log
        otherwise). ``"binomial"`` and ``"logit"`` are accepted aliases for
        ``"bernoulli"``.
    alpha : float
        Target miscoverage in ``(0, 1)`` for the conservative enclosure.
    theta : float, optional
        Negative-binomial size parameter; required for that family only.
    offset : array (n,), optional
        Training offsets (default zeros).
    offset_star : float
        The test row's offset (default ``0``).

    Returns
    -------
    dict
        ``{"intervals", "alpha", "n_augmented", "set_kind"}``. ``intervals`` is the sorted,
        disjoint list of ``(lo, hi)`` pieces of the set; endpoints may be
        infinite. ``set_kind`` is ``"conservative_enclosure"``. For the discrete
        families each piece is the integer run ``lo..=hi``.
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
    offset_arr = None
    if offset is not None:
        offset_arr = np.ascontiguousarray(offset, dtype=np.float64)
        if offset_arr.ndim != 1:
            raise ValueError(f"offset must be 1-D, got shape {offset_arr.shape}")
    return rust_module().glm_full_conformal(
        design_arr,
        response_arr,
        s_lambda_arr,
        x_star_arr,
        str(family),
        float(alpha),
        None if theta is None else float(theta),
        offset_arr,
        float(offset_star),
    )
