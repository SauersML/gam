"""Bayesian model-evidence comparison across fits.

Rank candidate gamfit fits by their REML / LAML marginal-likelihood scores.
The score extraction, Tierney-Kadane normalization, and evidence ranking live
in Rust; this module is only the Python FFI wall.
"""

from __future__ import annotations

from typing import Any

from ._binding import rust_module


def _extract_reml_score(fit: Any) -> float:
    return float(rust_module().extract_reml_score(fit))


def _extract_reml_score_raw(fit: Any) -> float:
    return float(rust_module().extract_reml_score_raw(fit))


def _tierney_kadane_normalizer_from_null_dim(
    null_dim: float,
    null_hessian_logdet: float | None = None,
) -> float:
    return float(
        rust_module().tierney_kadane_normalized_score(
            0.0,
            float(null_dim),
            null_hessian_logdet,
        )
    )


def _extract_edf(fit: Any) -> float | None:
    edf = rust_module().extract_reml_edf(fit)
    return None if edf is None else float(edf)


def compare_models(
    fits: list[Any] | tuple[Any, ...],
    names: list[str] | tuple[str, ...] | None = None,
    *,
    cv_scores: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Rank candidate fits by Bayesian marginal-likelihood (REML / LAML).

    Returns ``ranking``, ``winner``, ``evidence_summary``, and ``score_table``.
    When ``cv_scores`` is supplied, the Rust ranking also returns
    ``cv_optional`` aligned to the evidence order.
    """
    return dict(rust_module().compare_reml_fits(fits, names, cv_scores))
