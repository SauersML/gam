"""Lawley likelihood-ratio Bartlett correction (issue #939).

Thin wrapper over the Rust core ``gam::inference::lawley``: the second-order
Bartlett factor ``c = E[W]/d = 1 + (ε_k − ε_{k−q})/d`` that makes the ``χ²_d``
reference of a likelihood-ratio statistic ``W`` second-order accurate
(``O(n⁻²)`` size error instead of ``O(n⁻¹)``). The full Lawley (1956) expansion
— including the score↔information joint cumulants — lives in Rust; this module
only marshals arrays across the FFI boundary.

This is an EXPLICIT instrument, not an auto-magic rewrite of the summary
table's smooth-term test. That table reports a *Wald* χ²; the Lawley factor
corrects the *likelihood-ratio* statistic, a different quantity. Apply this to
an observed LR statistic from a per-term LR refit, not to the Wald χ².

Functions
---------
lawley_bartlett_factor
    The Bartlett factor for a tested coefficient block, with the optional
    corrected statistic and p-value when an observed LR statistic is supplied.
lawley_bartlett_factor_estimated_lambda
    The same factor plus the ρ-hat sampling-variation contribution from the
    inverse outer Hessian covariance of the smoothing parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def lawley_bartlett_factor(
    design: Any,
    family: str,
    eta: Any,
    tested_start: int,
    tested_end: int,
    ref_df: float,
    *,
    penalty: Any | None = None,
    dispersion: float = 1.0,
    prior_weights: Any | None = None,
    lr_statistic: float | None = None,
) -> dict[str, Any]:
    """Lawley LR Bartlett factor for the block ``design[:, start:end]``.

    Parameters
    ----------
    design : array (n, k)
        Model design; the tested block is columns ``tested_start:tested_end``.
    family : {"gaussian", "poisson", "binomial", "gamma"}
        GLM family with its canonical link (identity / log / logit / log).
    eta : array (n,)
        Per-row linear predictor ``η`` at the NULL fit (Lawley's ε is an
        expectation evaluated at the null).
    tested_start, tested_end : int
        Column range under test (H0: those coefficients are zero).
    ref_df : float
        LR reference degrees of freedom ``d``.
    penalty : array (k, k), optional
        Quadratic penalty ``S_λ`` folded into the information (valid for nulls
        with ``S_λ β0 = 0``).
    dispersion : float
        Family dispersion φ (Gaussian σ², Gamma φ; 1 for Poisson/Binomial).
    prior_weights : array (n,), optional
        Per-row weights (e.g. binomial trial counts).
    lr_statistic : float, optional
        Observed LR statistic to correct; when supplied the result also carries
        ``corrected_statistic = lr_statistic / c`` and the corrected ``χ²_d``
        ``p_value_corrected`` / ``p_value_uncorrected``.

    Returns
    -------
    dict
        ``{"bartlett_factor", "mean_shift", "ref_df"}`` and, when
        ``lr_statistic`` is given, ``{"corrected_statistic",
        "p_value_corrected", "p_value_uncorrected"}``.
    """
    from ._binding import rust_module

    design_arr = np.ascontiguousarray(design, dtype=np.float64)
    if design_arr.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {design_arr.shape}")
    eta_arr = np.ascontiguousarray(eta, dtype=np.float64)
    if eta_arr.ndim != 1:
        raise ValueError(f"eta must be 1-D, got shape {eta_arr.shape}")
    penalty_arr = (
        None if penalty is None else np.ascontiguousarray(penalty, dtype=np.float64)
    )
    weights_arr = (
        None
        if prior_weights is None
        else np.ascontiguousarray(prior_weights, dtype=np.float64)
    )
    return rust_module().lawley_bartlett_factor(
        design_arr,
        str(family),
        eta_arr,
        int(tested_start),
        int(tested_end),
        float(ref_df),
        penalty_arr,
        float(dispersion),
        weights_arr,
        None if lr_statistic is None else float(lr_statistic),
    )


def lawley_bartlett_factor_estimated_lambda(
    design: Any,
    family: str,
    eta: Any,
    tested_start: int,
    tested_end: int,
    ref_df: float,
    *,
    penalty: Any,
    components: Any,
    rho_cov: Any,
    dispersion: float = 1.0,
    prior_weights: Any | None = None,
    lr_statistic: float | None = None,
) -> dict[str, Any]:
    """Lawley LR Bartlett factor including estimated-λ ρ-hat variation.

    ``penalty`` is the fitted total ``S_lambda``. ``components`` is one fitted
    penalty component per log smoothing parameter, each already multiplied by
    its λ. ``rho_cov`` is ``Cov(rho_hat)``, the regularized inverse exact outer
    Hessian from the smoothing-parameter optimizer.
    """
    from ._binding import rust_module

    design_arr = np.ascontiguousarray(design, dtype=np.float64)
    if design_arr.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {design_arr.shape}")
    eta_arr = np.ascontiguousarray(eta, dtype=np.float64)
    if eta_arr.ndim != 1:
        raise ValueError(f"eta must be 1-D, got shape {eta_arr.shape}")
    penalty_arr = np.ascontiguousarray(penalty, dtype=np.float64)
    if penalty_arr.ndim != 2:
        raise ValueError(f"penalty must be 2-D, got shape {penalty_arr.shape}")
    component_arrs = [
        np.ascontiguousarray(component, dtype=np.float64) for component in components
    ]
    for idx, component in enumerate(component_arrs):
        if component.ndim != 2:
            raise ValueError(
                f"components[{idx}] must be 2-D, got shape {component.shape}"
            )
    rho_cov_arr = np.ascontiguousarray(rho_cov, dtype=np.float64)
    if rho_cov_arr.ndim != 2:
        raise ValueError(f"rho_cov must be 2-D, got shape {rho_cov_arr.shape}")
    weights_arr = (
        None
        if prior_weights is None
        else np.ascontiguousarray(prior_weights, dtype=np.float64)
    )
    return rust_module().lawley_bartlett_factor_estimated_lambda(
        design_arr,
        str(family),
        eta_arr,
        int(tested_start),
        int(tested_end),
        float(ref_df),
        penalty_arr,
        component_arrs,
        rho_cov_arr,
        float(dispersion),
        weights_arr,
        None if lr_statistic is None else float(lr_statistic),
    )
