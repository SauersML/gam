"""Smoothing-corrected AIC comparison across fits.

Ranking lives in Rust (``compare_saved_models``, the same function behind
``gam compare``); this module is only the Python FFI wall.
"""

from __future__ import annotations

from typing import Any

from ._binding import rust_module


def _extract_reml_score_raw(fit: Any) -> float:
    return float(rust_module().extract_reml_score_raw(fit))


def compare_models(
    fits: list[Any] | tuple[Any, ...],
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Rank fitted models by their smoothing-corrected AIC.

    ``fits`` are ``gamfit.Model`` objects (or their saved bytes) fitted to the
    same data with the same response family. The criterion is
    ``aic_corrected = -2*loglik + 2*(edf_corrected + scale_dof)``, where
    ``edf_corrected`` adds the Wood-Pya-Saefken correction for having
    estimated the smoothing parameters to the conditional EDF. The plain
    conditional AIC ``-2*loglik + 2*(edf + scale_dof)`` treats the smoothing
    parameters as known and so favours over-flexible models; it is reported in
    each row but never ranked on.

    Returns a dict with

    * ``criterion`` -- ``"aic_corrected"``;
    * ``ranking`` -- one dict per fit, best first, with ``name``,
      ``aic_corrected``, ``delta_aic`` (gap to the winner),
      ``evidence_ratio`` (``exp(delta_aic / 2)``, the Akaike evidence ratio of
      the winner over the row; not a Bayes factor), ``aic_conditional``,
      ``edf_corrected`` and ``edf_conditional``;
    * ``winner`` and ``evidence_summary``;
    * ``score_table`` -- each fit's REML/LAML score. Differences in that score
      compare models only when they share the family, the data and the
      unpenalized fixed-effect space (for example nested smooths added to a
      common parametric part); across different fixed effects the REML
      likelihoods are of different transformed data and are not comparable.

    A fit with no corrected AIC (for example a spline-scan fit) is refused
    with the reason its summary records.
    """
    return dict(rust_module().compare_models(fits, names))
